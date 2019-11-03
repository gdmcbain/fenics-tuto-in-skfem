from matplotlib.pyplot import subplots
import numpy as np

from skfem import *
from skfem.models.poisson import laplace

import dmsh


mesh = MeshTri(*map(np.transpose,
                    dmsh.generate(dmsh.Circle([0., 0.], 1.), np.sqrt(2)/64)))
basis = InteriorBasis(mesh, ElementTriP2())

boundary = basis.get_dofs().all()

beta = 8.
R0 = .6

def load_f(x):
    return 4 / np.exp(beta**2 * (x[0]**2 + (x[1] - R0)**2))

@linear_form
def load(v, dv, w):
    return v * load_f(w.x)

A = asm(laplace, basis)
b = asm(load, basis)

w = solve(*condense(A, b, D=boundary))

mesh.save('w.xdmf', {'deflexion': w})

y = np.linspace(-1, 1, 103)[1:-1]
yy = np.vstack([np.zeros_like(y), y])

fig, ax = subplots()
ax.plot(load_f(yy), y, marker='x', label='load')
factor = 50
ax.plot(factor * basis.interpolator(w)(yy), y,
        marker='o', label=f'deflexion × {factor}')
ax.set_xlabel('deflexion')
ax.set_ylabel('y/a')
ax.legend()
fig.savefig('curves.png')
