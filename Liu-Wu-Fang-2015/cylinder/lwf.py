import logging
from pathlib import Path

from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import bmat, csr_matrix, spmatrix
from scipy.sparse.linalg import gmres, splu
from scipy.sparse.linalg.interface import LinearOperator

import skfem as fem
from skfem.helpers import dot
from skfem.io.meshio import from_file
from skfem.models.general import divergence, rot
from skfem.models.poisson import laplace, mass, vector_laplace
from skfem.visuals.matplotlib import draw, plot, savefig


@fem.BilinearForm
def vector_mass(u, v, w):
    return dot(v, u)


@fem.BilinearForm
def oseen(u, v, w):
    return dot(np.einsum("j...,ij...->i...", w["w"], u.grad), v)


logging.basicConfig(
    format="%(levelname)s %(asctime)s %(message)s",
    filename="log.python." + Path(__file__).stem,
    level=logging.INFO,
)

logging.info("Beginning.")
mesh = from_file("cylinder_pygmsh.msh")
logging.info(f"mesh: {mesh}")
ax = draw(mesh)
ax.get_figure().savefig("mesh.svg")

element = {"u": fem.ElementVectorH1(fem.ElementQuad2()), "p": fem.ElementQuad1()}
basis = {v: fem.InteriorBasis(mesh, e, intorder=3) for v, e in element.items()}
logging.info(f"basis: {({v: b.N for v, b in basis.items()})}")

nu = 1
dt = 0.001
tol_picard = 1e-3
tol_steady = 1e-3

M = fem.asm(vector_mass, basis["u"])
velocity_matrix0 = M / dt + fem.asm(vector_laplace, basis["u"]) * nu
B = -fem.asm(divergence, basis["u"], basis["p"])
Q = fem.asm(mass, basis["p"])
D = basis["u"].find_dofs()

diagQ = Q.diagonal()

dirichlet = {
    "u": np.setdiff1d(
        basis["u"].get_dofs().all(),
        basis["u"].get_dofs(mesh.boundaries["outlet"]).all(),
    ),
    "p": basis["p"].get_dofs(mesh.boundaries["outlet"]).all()
}

uvp = np.zeros(sum(b.N for b in basis.values()))
inlet_dofs = basis["u"].get_dofs(mesh.boundaries["inlet"]).all("u^1")
inlet_y = mesh.p[1, mesh.facets[:, mesh.boundaries["inlet"]]]
inlet_y_lim = inlet_y.min(), inlet_y.max()
monic = np.polynomial.polynomial.Polynomial.fromroots(inlet_y_lim)
uvp[inlet_dofs] = -6 * monic(basis["u"].doflocs[1, inlet_dofs]) / inlet_y_lim[1] **2
logging.info("Assembled basic operators.")


def velocity_matrix(u: np.ndarray) -> csr_matrix:
    return velocity_matrix0 + fem.asm(oseen, basis["u"], w=basis["u"].interpolate(u))


t = 0.0
t_max = 2.0

K0 = bmat([[velocity_matrix0, B.T], [B, 1e-6 * Q]], "csc")
Kint = fem.condense(K0, D=dirichlet["u"], expand=False)
Aint = Kint[: -basis["p"].N, : -basis["p"].N]
Alu = splu(Aint)
Apc = LinearOperator(Aint.shape, Alu.solve, dtype=M.dtype)


def precondition(uvp: np.ndarray) -> np.ndarray:
    uv, p = np.split(uvp, Aint.shape[:1])
    return np.concatenate([Apc @ uv, p / diagQ])


pc = LinearOperator(Kint.shape, precondition, dtype=Q.dtype)
logging.info("Factored Stokes LU preconditioner.")

while True:  # time-stepping
    t += dt

    u_old = uvp[: basis["u"].N]
    f = np.concatenate([M @ u_old / dt, np.zeros(basis["p"].N)])

    iterations_picard = 0
    u = uvp[: basis["u"].N]

    while True:  # Picard
        uvp = fem.solve(
            *fem.condense(
                bmat([[velocity_matrix(u), B.T], [B, 1e-6 * Q]], "csr"), f, uvp, D=dirichlet["u"]
            ),
            solver=fem.solver_iter_krylov(gmres, verbose=True, M=pc),
        )
        iterations_picard += 1
        u_new = uvp[: basis["u"].N]
        if np.linalg.norm(u_new - u) < tol_picard:
            break
        u = u_new
    change = np.linalg.norm(u - u_old)
    logging.info(f"t = {t}, {iterations_picard} Picard iterations, ||Delta u|| = {change}")

    if change < tol_steady:
        break

basis["omega"] = basis["u"].with_element(fem.ElementQuad1())
vorticity = fem.asm(rot, basis["omega"], w=basis["u"].interpolate(u))
logging.info("Computed vorticity.")

ax = draw(mesh)
plot(basis["omega"], vorticity, ax=ax)
ax.tricontour(Triangulation(*mesh.p, mesh.to_meshtri().t.T), uvp[-basis["p"].N:])
savefig(f"cylinder_pygmsh-{Path(__file__).stem}.png"))
logging.info("Plotted vorticity and isobars.")
