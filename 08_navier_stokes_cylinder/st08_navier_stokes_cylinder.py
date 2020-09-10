from pathlib import Path

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from meshio.xdmf import TimeSeriesWriter

import skfem
from skfem.importers.meshio import from_meshio
from skfem.models.general import divergence
from skfem.models.poisson import laplace, vector_laplace

import pyamgcl


@skfem.BilinearForm
def vector_mass(u, v, w):
    return sum(v * u)


@skfem.LinearForm
def acceleration(v, w):
    """Compute the vector (v, u . grad u) for given velocity u

    passed in via w after having been interpolated onto its quadrature
    points.

    In Cartesian tensorial indicial notation, the integrand is

    .. math::

        u_j u_{i,j} v_i.

    """
    return sum(np.einsum("j...,ij...->i...", w["wind"], w["wind"].grad) * v)


@skfem.BilinearForm
def port_pressure(u, v, w):
    """v is the P2 velocity test-function, u a P1 pressure"""
    return sum(v * (u * w.n))


radius = 0.05
height = 0.41

geom = Geometry()
cylinder = geom.add_circle([0.2, 0.2, 0.0], radius, lcar=radius / 2)
channel = geom.add_rectangle(
    0.0, 2.2, 0.0, height, 0, holes=[cylinder], lcar=radius / 2
)
geom.add_physical(channel.surface, "domain")
geom.add_physical(channel.lines[1], "outlet")
geom.add_physical(channel.lines[3], "inlet")

mesh = from_meshio(generate_mesh(geom, dim=2))

element = {"u": skfem.ElementVectorH1(skfem.ElementTriP2()), "p": skfem.ElementTriP1()}
basis = {
    **{v: skfem.InteriorBasis(mesh, e, intorder=4) for v, e in element.items()},
    "inlet": skfem.FacetBasis(mesh, element["u"], facets=mesh.boundaries["inlet"]),
}
M = skfem.asm(vector_mass, basis["u"])
L = {"u": skfem.asm(vector_laplace, basis["u"]), "p": skfem.asm(laplace, basis["p"])}
B = -skfem.asm(divergence, basis["u"], basis["p"])
P = B.T + skfem.asm(
    port_pressure,
    *(
        skfem.FacetBasis(mesh, element[v], intorder=3, facets=mesh.boundaries["outlet"])
        for v in ["p", "u"]
    ),
)

t_final = 5.0
dt = 0.001

nu = 0.001

K_lhs = M / dt + nu * L["u"] / 2
K_rhs = M / dt - nu * L["u"] / 2

uv_, p_ = (np.zeros(basis[v].N) for v in element.keys())  # penultimate
p__ = np.zeros_like(p_)  # antepenultimate
u = np.zeros_like(uv_)

dirichlet = {
    "u": np.setdiff1d(
        basis["u"].get_dofs().all(),
        basis["u"].get_dofs(mesh.boundaries["outlet"]).all(),
    ),
    "p": basis["p"].get_dofs(mesh.boundaries["outlet"]).all(),
}

uv0 = np.zeros(basis["u"].N)
inlet_dofs = basis["u"].get_dofs(mesh.boundaries["inlet"]).all("u^1")
inlet_y_lim = [p.x[1] for p in channel.lines[3].points[::-1]]
monic = np.polynomial.polynomial.Polynomial.fromroots(inlet_y_lim)
uv0[inlet_dofs] = -6 * monic(basis["u"].doflocs[1, inlet_dofs]) / inlet_y_lim[1] ** 2


def embed(xy: np.ndarray) -> np.ndarray:
    return np.pad(xy, ((0, 0), (0, 1)), "constant")


solvers = [
    pyamgcl.solver(
        pyamgcl.amg(
            skfem.condense(K_lhs, D=dirichlet["u"], expand=False),
            {"coarsening.type": "aggregation", "relax.type": "ilu0"},
        ),
        {"type": "cg"},
    ),
    pyamgcl.solver(
        pyamgcl.amg(skfem.condense(L["p"], D=dirichlet["p"], expand=False)),
        {"type": "cg"},
    ),
    pyamgcl.solver(
        pyamgcl.amg(
            skfem.condense(M / dt, D=dirichlet["u"], expand=False),
            {"relax.type": "gauss_seidel"},
        ),
        {"type": "cg"},
    ),
]

with TimeSeriesWriter(Path(__file__).with_suffix(".xdmf").name) as writer:

    writer.write_points_cells(embed(mesh.p.T), {"triangle": mesh.t.T})

    t = 0.0
    while t < t_final:
        t += dt

        # Step 1: momentum prediction

        uv = skfem.solve(
            *skfem.condense(
                K_lhs,
                K_rhs @ uv_
                - P @ (2 * p_ - p__)
                - skfem.asm(acceleration, basis["u"], wind=basis["u"].interpolate(u)),
                uv0,
                D=dirichlet["u"],
            ),
            solver=solvers[0],
        )

        # Step 2: pressure correction

        dp = skfem.solve(
            *skfem.condense(L["p"], (B / dt) @ uv, D=dirichlet["p"]), solver=solvers[1]
        )

        # Step 3: velocity correction

        p = p_ + dp
        du = skfem.solve(
            *skfem.condense(M / dt, -P @ dp, D=dirichlet["u"]), solver=solvers[2]
        )
        u = uv + du

        uv_ = uv
        p_, p__ = p, p_

        # postprocessing

        writer.write_data(
            t,
            point_data={"pressure": p, "velocity": embed(uv[basis["u"].nodal_dofs].T)},
        )

        print(f"t = {t}, max u = ", u[basis["u"].nodal_dofs].max())
