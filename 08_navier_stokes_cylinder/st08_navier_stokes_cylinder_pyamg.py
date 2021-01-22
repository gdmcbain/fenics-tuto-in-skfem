from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator


from meshio.xdmf import TimeSeriesWriter

import skfem
from skfem.io.json import from_file
from skfem.models.general import divergence
from skfem.models.poisson import laplace, vector_laplace
from skfem.utils import LinearSolver

from pyamg import ruge_stuben_solver


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


def probe(
    basis: skfem.InteriorBasis, points: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:

    finder = basis.mesh.element_finder(mapping=basis.mapping)
    cells = finder(*points)
    pts = basis.mapping.invF(points[:, :, np.newaxis], tind=cells)

    def interpolator(u: np.ndarray):
        w = np.zeros(points.shape[1])
        for k in range(basis.Nbfun):
            phi = basis.elem.gbasis(basis.mapping, pts, k, tind=cells)[0][0].flatten()
            w += u[basis.element_dofs[k, cells]] * phi
        return w

    return interpolator


parser = ArgumentParser()
parser.add_argument("--mesh", type=Path)
args = parser.parse_args()

if args.mesh:

    mesh = from_file(args.mesh)
else:
    try:
        mesh = from_file(Path("cylinder.json"))
    except Exception as e:
        from cylinder_dmsh import CylinderDmsh

        print("Couldn't find cylinder.json; generating in dmsh.")
        mesher = CylinderDmsh()
        mesh = mesher.mesh
        mesher.save()

element = {"u": skfem.ElementVectorH1(skfem.ElementTriP2()), "p": skfem.ElementTriP1()}
basis = {
    **{v: skfem.InteriorBasis(mesh, e, intorder=4) for v, e in element.items()},
    "inlet": skfem.FacetBasis(mesh, element["u"], facets=mesh.boundaries["inlet"]),
}
pressure_probe = probe(basis["p"], np.array([(0.15, 0.2), (0.25, 0.2)]).T)

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
inlet_y = mesh.p[1, mesh.facets[:, mesh.boundaries["inlet"]]]
inlet_y_lim = inlet_y.min(), inlet_y.max()
monic = np.polynomial.polynomial.Polynomial.fromroots(inlet_y_lim)
uv0[inlet_dofs] = -6 * monic(basis["u"].doflocs[1, inlet_dofs]) / inlet_y_lim[1] ** 2


def embed(xy: np.ndarray) -> np.ndarray:
    return np.pad(xy, ((0, 0), (0, 1)), "constant")


solvers = []
for A, v in [(K_lhs, "u"), (L["p"], "p"), (M / dt, "u")]:

    def s(arr: spmatrix, b: np.ndarray) -> LinearSolver:
        ml = ruge_stuben_solver(skfem.condense(arr, D=dirichlet[v], expand=False))

        def f(_: LinearOperator, b: np.ndarray) -> np.ndarray:
            return ml.solve(b)

        return f

    solvers.append(s(A, v))

pressure_history = []

with TimeSeriesWriter(Path(__file__).with_suffix(".xdmf")) as writer:

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

        pressure_history.append([t, *pressure_probe(p)])

        writer.write_data(
            t,
            point_data={"pressure": p, "velocity": embed(uv[basis["u"].nodal_dofs].T)},
        )

        print(f"t = {t}, max u = ", u[basis["u"].nodal_dofs].max())

np.savetxt(Path(__file__).with_suffix(".csv"), pressure_history, delimiter=",")
