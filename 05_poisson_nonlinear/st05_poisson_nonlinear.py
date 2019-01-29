from sympy import symbols
from sympy.vector import CoordSys3D, gradient, divergence

R = CoordSys3D('R')
u = 1 + R.x + 2*R.y

def q(u):
    """Return nonlinear coefficient"""
    return 1 + u**2

f = -divergence(q(u) * gradient(u))

print('u:', u)
print('f:', f)
