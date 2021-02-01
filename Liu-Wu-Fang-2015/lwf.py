from pathlib import Path

import numpy as np
from scipy.sparse import bmat, csr_matrix

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


mesh = skfem.MeshQuad.init_tensor(*[np.linspace(0, 1, 1 + 2 ** 5)] * 2)
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

M = skfem.asm(vector_mass, basis["u"])
velocity_matrix0 = M / dt + skfem.asm(vector_laplace, basis["u"]) * nu
B = -skfem.asm(divergence, basis["u"], basis["p"])
C = skfem.asm(mass, basis["p"])
D = basis["u"].find_dofs()

uvp0 = np.zeros(sum(b.N for b in basis.values()))
uvp0[D["lid"].all()[::2]] = 1.0


def velocity_matrix(u: np.ndarray) -> csr_matrix:
    return velocity_matrix0 + skfem.asm(oseen, basis["uu"], w=basis["u"].interpolate(u))


t = 0.0
uvp = uvp0.copy()
t_max = 1.0

while True:  # time-stepping
    t += dt

    f = np.concatenate([M @ uvp[: basis["u"].N] / dt, np.zeros(basis["p"].N)])
    uvp = skfem.solve(
        *skfem.condense(
            bmat([[velocity_matrix0, B.T], [B, 1e-6 * C]], "csr"), f, uvp, D=D
        )
    )
    print(t, uvp[: basis["u"].N] @ M @ uvp[: basis["u"].N])
    if t > 4 * dt:
        break

ax = draw(mesh)
plot(basis["p"], uvp[-basis["p"].N :], ax=ax)
ax.quiver(*mesh.p, *uvp[basis["u"].nodal_dofs])
savefig(Path(__file__).with_suffix(".png"))
