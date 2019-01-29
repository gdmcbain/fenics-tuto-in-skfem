from pathlib import Path

from sympy import symbols
from sympy.vector import CoordSys3D, gradient, divergence
from sympy.utilities.lambdify import lambdify

from skfem import (MeshTri, InteriorBasis, ElementTriP1,
                   linear_form, asm, solve, condense)
from skfem.models.poisson import laplace

output_dir = Path('poisson_nonlinear')
try:
    output_dir.mkdir()
except FileExistsError:
    pass

R = CoordSys3D('R')

u = 1 + R.x + 2*R.y

def q(u):
    """Return nonlinear coefficient"""
    return 1 + u**2

f = -divergence(q(u) * gradient(u))

print('u:', u)
print('f:', f)

mesh = MeshTri()
mesh.refine(3)

V = InteriorBasis(mesh, ElementTriP1())

def funcify(f):
    x, y = symbols('x y')
    return lambdify((x, y), f.subs({R.x: x, R.y: y}))

u_D = funcify(u)

u = u_D(*mesh.p)
mesh.plot(u).get_figure().savefig(str(output_dir.joinpath('initial.png')))

f_f = funcify(f)

@linear_form
def load(v, dv, w):
    return v * f_f(*w.x)

a = asm(laplace, V)
b = asm(load, V)

boundary = V.get_dofs().all()
interior = V.complement_dofs(boundary)

u[interior] = solve(*condense(a, b, u, D=boundary))

mesh.plot(u).get_figure().savefig(str(output_dir.joinpath('solution.png')))
