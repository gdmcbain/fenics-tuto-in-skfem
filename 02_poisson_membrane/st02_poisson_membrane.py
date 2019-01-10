import numpy as np

from skfem import *
from skfem.models.poisson import laplace

import dmsh


mesh = MeshTri(*map(np.transpose,
                    dmsh.generate(dmsh.Circle([0., 0.], 1.), 1/64)))
basis = InteriorBasis(mesh, ElementTriP2())

boundary = basis.get_dofs().all()

@linear_form
def load(v, dv, w):
    beta = 8.
    R0 = .6
    return 4 / np.exp(beta**2 * (w.x[0]**2 + (w.x[1] - R0)**2))

A = asm(laplace, basis)
b = asm(load, basis)

w = np.zeros_like(b)
w[basis.complement_dofs(boundary)] = solve(*condense(A, b, D=boundary))

mesh.save('w.xdmf', w)
