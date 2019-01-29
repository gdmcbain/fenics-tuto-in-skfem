from pathlib import Path

from sympy import symbols
from sympy.vector import CoordSys3D, gradient, divergence
from sympy.utilities.lambdify import lambdify

from skfem import MeshTri, InteriorBasis, ElementTriP1


output_dir = Path('poisson_nonlinear')
try:
    output_dir.mkdir()
except FileExistsError:
    pass

R = CoordSys3D('R')
x, y = symbols('x y')
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

u_D = lambdify((x, y), u.subs({R.x: x, R.y: y}))

u = u_D(*mesh.p)
mesh.save(str(output_dir.joinpath('dirichlet.xdmf')), u)
