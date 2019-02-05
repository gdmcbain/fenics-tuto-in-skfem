from pathlib import Path

import numpy as np

from skfem import (MeshTet,
                   ElementTetP1, ElementVectorH1,
                   InteriorBasis, asm, linear_form,
                   condense, solve)
from skfem.models.elasticity import linear_elasticity


L = 1.; W = 0.2
mu = 1.
rho = 1.
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

mesh = MeshTet.init_tensor(np.linspace(0, L, 10 + 1),
                           np.linspace(0, W, 3 + 1),
                           np.linspace(0, W, 3 + 1))
mesh.save('mesh.xdmf')

basis = InteriorBasis(mesh, ElementVectorH1(ElementTetP1()))

clamped = basis.get_dofs(lambda x, y, z: x==0.).all()
free = basis.complement_dofs(clamped)

K = asm(linear_elasticity(lambda_, mu), basis)

@linear_form
def load(v, dv, w):
    return -rho * g * v[2]

f = asm(load, basis)

u = np.zeros(basis.N)
u[free] = solve(*condense(K, f, u, I=free))

deformed = MeshTet(mesh.p + u[basis.nodal_dofs], mesh.t)
deformed.save('deformed.xdmf')

u_magnitude = np.linalg.norm(u.reshape((-1, 3)), axis=1)
print('min/max u:', min(u_magnitude), max(u_magnitude))
