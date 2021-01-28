import numpy as np
from scipy.sparse import bmat, csr_matrix

import skfem
from skfem.helpers import dot
from skfem.models.general import divergence
from skfem.models.poisson import vector_laplace


@skfem.BilinearForm
def vector_mass(u, v, w):
    return dot(v, u)


@skfem.BilinearForm
def oseen(u, v, w):
    return dot(np.einsum("j...,ij...->i...", w["w"], u.grad), v)


mesh = skfem.MeshQuad.init_tensor(*[np.linspace(0, 1, 1 + 2 ** 3)] * 2)
mesh.define_boundary("lid", lambda x: x[1] == 1.0)
mesh.define_boundary(
    "wall",
    lambda x: np.logical_or(np.logical_or(x[0] == 0.0, x[0] == 1.0), x[1] == 0.0),
)
print(mesh)

element = {"u": skfem.ElementVectorH1(skfem.ElementQuad2()), "p": skfem.ElementQuad0()}
basis = {v: skfem.InteriorBasis(mesh, e, intorder=3) for v, e in element.items()}
print({v: b.N for v, b in basis.items()})


dt = 0.1
nu = 1.0 / 20

velocity_matrix0 = (
    skfem.asm(vector_mass, basis["u"]) / dt + skfem.asm(vector_laplace, basis["u"]) * nu
)
B = -skfem.asm(divergence, basis["u"], basis["p"])

uvp0 = np.zeros(sum(b.N for b in basis.values()))
uvp0[basis["u"].find_dofs()["lid"].all()[::2]] = 1.


def velocity_matrix(u: np.ndarray) -> csr_matrix:
    return velocity_matrix0 + skfem.asm(oseen, basis["uu"], w=basis["u"].interpolate(u))


def picard_step(u_old: np.ndarray, u_k: np.ndarray) -> np.ndarray:
    return skfem.solve(
        *skfem.condense(
            bmat([[velocity_matrix(u_k), B.T], [B, None]]),
            np.concatenate(u_old / dt, np.zeros(basis["p"].N)),
            uvp0,
            D=basis["u"].find_dofs()["lid"],
        )
    )


def step(t: float, uvp: np.ndarray) -> tuple[float, np.ndarray]:
    u_old = uvp[: basis["u"].N]
    pass
