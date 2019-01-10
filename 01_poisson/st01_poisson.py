import numpy as np

from skfem import *
from skfem.models.poisson import laplace, unit_load, mass

mesh = MeshTri()
mesh.refine(3)

V = InteriorBasis(mesh, ElementTriP1())

u_D = 1 + [1, 2] @ mesh.p**2

boundary = mesh.boundary_nodes()

u = np.zeros_like(u_D)
u[boundary] = u_D[boundary]

a = asm(laplace, V)
L = -6.0 * asm(unit_load, V)

u[mesh.interior_nodes()] = solve(*condense(a, L, u, D=boundary))

ax = mesh.plot(u)
ax.get_figure().savefig('poisson.png')

mesh.save('u.xdmf', u)

error = u - u_D
print('error_L2  =', np.sqrt(error.T @ asm(mass, V) @ error))
print('error_max =', np.linalg.norm(error, np.inf))
