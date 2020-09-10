from pathlib import Path

import numpy as np

from skfem import (
    MeshTet,
    ElementTetP1,
    ElementVectorH1,
    InteriorBasis,
    asm,
    LinearForm,
    condense,
    solve,
)
from skfem.models.elasticity import linear_elasticity


L = 1.0
W = 0.2
mu = 1.0
rho = 1.0
delta = W / L
gamma = 0.4 * delta ** 2
beta = 1.25
lambda_ = beta
g = gamma

mesh = MeshTet.init_tensor(
    np.linspace(0, L, 10 + 1), np.linspace(0, W, 3 + 1), np.linspace(0, W, 3 + 1)
)

basis = InteriorBasis(mesh, ElementVectorH1(ElementTetP1()))

clamped = basis.get_dofs(lambda x: x[0] == 0.0).all()
free = basis.complement_dofs(clamped)

K = asm(linear_elasticity(lambda_, mu), basis)


@LinearForm
def load(v, w):
    return -rho * g * v.value[2]


f = asm(load, basis)

u = solve(*condense(K, f, I=free))

deformed = MeshTet(mesh.p + u[basis.nodal_dofs], mesh.t)
deformed.save("deformed.xdmf")

u_magnitude = np.linalg.norm(u.reshape((-1, 3)), axis=1)
print("min/max u:", min(u_magnitude), max(u_magnitude))
