import logging
from pathlib import Path

from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import gcrotmk, spilu, LinearOperator

import skfem as fem
from skfem.helpers import dot
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
mesh = fem.MeshQuad.init_tensor(*[np.linspace(0, 1, 1 + 2 ** 5)] * 2)
mesh.define_boundary("lid", lambda x: x[1] == 1.0)
mesh.define_boundary(
    "wall",
    lambda x: np.logical_or(np.logical_or(x[0] == 0.0, x[0] == 1.0), x[1] == 0.0),
)
logging.info(f"mesh: {mesh}")

element = {"u": fem.ElementVectorH1(fem.ElementQuad2()), "p": fem.ElementQuad1()}
basis = {v: fem.InteriorBasis(mesh, e, intorder=3) for v, e in element.items()}
logging.info(f"basis: {({v: b.N for v, b in basis.items()})}")

nu = 1.0 / 2e2
dt = 10.0 * mesh.param()
tol_picard = 1e-3
tol_steady = 1e-3

M = fem.asm(vector_mass, basis["u"])
velocity_matrix0 = M / dt + fem.asm(vector_laplace, basis["u"]) * nu
B = -fem.asm(divergence, basis["u"], basis["p"])
Q = fem.asm(mass, basis["p"])
D = basis["u"].find_dofs()

diagQ = Q.diagonal()

uvp = np.zeros(sum(b.N for b in basis.values()))
uvp[D["lid"].all()[::2]] = 1.0
logging.info("Assembled basic operators.")


def velocity_matrix(u: np.ndarray) -> csr_matrix:
    return velocity_matrix0 + fem.asm(oseen, basis["u"], w=basis["u"].interpolate(u))


t = 0.0
t_max = 2.0


while True:  # time-stepping
    t += dt

    u = u_old = uvp[: basis["u"].N]
    f = np.concatenate([M @ u_old / dt, np.zeros(basis["p"].N)])

    K0 = bmat([[velocity_matrix(u), B.T], [B, 1e-6 * Q]], "csc")
    Kint = fem.condense(K0, D=D, expand=False)
    Aint = Kint[: -basis["p"].N, : -basis["p"].N]
    Alu = spilu(Aint)
    Apc = LinearOperator(Aint.shape, Alu.solve, dtype=M.dtype)

    def precondition(uvp: np.ndarray) -> np.ndarray:
        uv, p = np.split(uvp, Aint.shape[:1])
        return np.concatenate([Apc @ uv, p / diagQ])

    pc = LinearOperator(Kint.shape, precondition, dtype=Q.dtype)

    iterations_picard = 0
    while True:  # Picard
        uvp = fem.solve(
            *fem.condense(
                bmat([[velocity_matrix(u), B.T], [B, 1e-6 * Q]], "csr"), f, uvp, D=D
            ),
            solver=fem.solver_iter_krylov(gcrotmk, verbose=True, M=pc),
        )
        iterations_picard += 1
        u_new = uvp[: basis["u"].N]
        if np.linalg.norm(u_new - u) < tol_picard:
            break
        u = u_new
    change = np.linalg.norm(u - u_old)
    logging.info(
        f"t = {t}, {iterations_picard} Picard iterations, ||Delta u|| = {change}"
    )

    if change < tol_steady:
        break

basis["psi"] = basis["u"].with_element(fem.ElementQuad2())
vorticity = fem.asm(rot, basis["psi"], w=basis["u"].interpolate(u))
logging.info("Computed vorticity.")
psi = fem.solve(
    *fem.condense(fem.asm(laplace, basis["psi"]), vorticity, D=basis["psi"].find_dofs())
)
logging.info("Computed streamfunction.")

ax = draw(mesh)
plot(basis["p"], uvp[-basis["p"].N :], ax=ax)
ax.quiver(*mesh.p, *uvp[basis["u"].nodal_dofs])
ax.tricontour(
    Triangulation(*mesh.p, mesh.to_meshtri().t.T),
    psi[basis["psi"].nodal_dofs.flatten()],
)
savefig(Path(__file__).with_suffix(".png"))
logging.info("Plotted pressure, velocity, and stream-lines.")
