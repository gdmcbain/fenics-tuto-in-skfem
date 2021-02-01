import logging
from pathlib import Path

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import gmres

import skfem
from skfem.helpers import dot
from skfem.models.general import divergence
from skfem.models.poisson import mass, vector_laplace
from skfem.visuals.matplotlib import draw, plot, savefig


@skfem.BilinearForm
def vector_mass(u, v, w):
    return dot(v, u)


@skfem.BilinearForm
def oseen(u, v, w):
    return dot(np.einsum("j...,ij...->i...", w["w"], u.grad), v)


logging.basicConfig(
    format="%(levelname)s %(asctime)s %(message)s",
    filename="log.python." + Path(__file__).stem,
    level=logging.INFO,
)

logging.info("Beginning.")
mesh = skfem.MeshQuad.init_tensor(*[np.linspace(0, 1, 1 + 2 ** 5)] * 2)
mesh.define_boundary("lid", lambda x: x[1] == 1.0)
mesh.define_boundary(
    "wall",
    lambda x: np.logical_or(np.logical_or(x[0] == 0.0, x[0] == 1.0), x[1] == 0.0),
)
logging.info(f"mesh: {mesh}")

element = {"u": skfem.ElementVectorH1(skfem.ElementQuad2()), "p": skfem.ElementQuad0()}
basis = {v: skfem.InteriorBasis(mesh, e, intorder=3) for v, e in element.items()}
logging.info(f"basis: {({v: b.N for v, b in basis.items()})}")

dt = 0.1
nu = 1.0 / 20
tol_picard = 1e-3
tol_steady = 1e-3

M = skfem.asm(vector_mass, basis["u"])
velocity_matrix0 = M / dt + skfem.asm(vector_laplace, basis["u"]) * nu
B = -skfem.asm(divergence, basis["u"], basis["p"])
Q = skfem.asm(mass, basis["p"])
D = basis["u"].find_dofs()

uvp = np.zeros(sum(b.N for b in basis.values()))
uvp[D["lid"].all()[::2]] = 1.0
logging.info("Assembled basic operators.")


def velocity_matrix(u: np.ndarray) -> csr_matrix:
    return velocity_matrix0 + skfem.asm(oseen, basis["u"], w=basis["u"].interpolate(u))


t = 0.0
t_max = 2.0

while True:  # time-stepping
    t += dt

    u_old = uvp[: basis["u"].N]
    f = np.concatenate([M @ u_old / dt, np.zeros(basis["p"].N)])
    K0 = bmat([[velocity_matrix0, B.T], [B, 1e-6 * Q]], "csr")
    Kint, rhs, uint, I = skfem.condense(K0, f, uvp, D=D)

    uvp = skfem.solve(Kint, rhs, uint, I)

    iterations_picard = 0
    u = uvp[: basis["u"].N]
    pc = skfem.build_pc_ilu(Kint)
    while True:  # Picard
        uvp = skfem.solve(
            *skfem.condense(
                bmat([[velocity_matrix(u), B.T], [B, 1e-6 * Q]], "csr"), f, uvp, D=D
            ),
            solver=skfem.solver_iter_krylov(gmres, verbose=True, M=pc)
        )
        iterations_picard += 1
        u_new = uvp[: basis["u"].N]
        if np.linalg.norm(u_new - u) < tol_picard:
            break
        u = u_new
    change = np.linalg.norm(u - u_old)
    logging.info(f"t = {t}, {iterations_picard} Picard iterations, ||u|| = {change}")

    if change < tol_steady:
        break

ax = draw(mesh)
plot(basis["p"], uvp[-basis["p"].N :], ax=ax)
ax.quiver(*mesh.p, *uvp[basis["u"].nodal_dofs])
savefig(Path(__file__).with_suffix(".png"))
