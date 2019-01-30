from pathlib import Path

import numpy as np
from scipy.optimize import root

from sympy import symbols
from sympy.vector import CoordSys3D, gradient, divergence
from sympy.utilities.lambdify import lambdify

from skfem import (MeshTri, InteriorBasis, ElementTriP1,
                   bilinear_form, linear_form, asm, solve, condense)
from skfem.models.poisson import laplace


output_dir = Path('poisson_nonlinear')

try:
    output_dir.mkdir()
except FileExistsError:
    pass


def q(u):
    """Return nonlinear coefficient"""
    return 1 + u**2


R = CoordSys3D('R')


def apply(f, coords):
    x, y = symbols('x y')
    return lambdify((x, y), f.subs({R.x: x, R.y: y}))(*coords)


u_exact = 1 + R.x + 2*R.y                  # exact solution
f = -divergence(q(u_exact) * gradient(u_exact))  # manufactured RHS 

mesh = MeshTri()
mesh.refine(3)

V = InteriorBasis(mesh, ElementTriP1())

boundary = V.get_dofs().all()
interior = V.complement_dofs(boundary)


@linear_form
def load(v, dv, w):
    return v * apply(f, w.x)


b = asm(load, V)


@bilinear_form
def diffusion_form(u, du, v, dv, w):
    return sum(dv * (q(w.w) * du))


def diffusion_matrix(u):
    return asm(diffusion_form, V, w=V.interpolate(u))

dirichlet = apply(u_exact, mesh.p)    # P1 nodal interpolation
mesh.plot(u).get_figure().savefig(str(output_dir.joinpath('exact.png')))


def residual(u):
    r = b - diffusion_matrix(u) @ u
    r[boundary] = 0.
    return r


u = np.zeros(V.N)
u[boundary] = dirichlet[boundary]
result = root(residual, u, method='krylov')

if result.success:
    u = result.x
    print('Success: residual =', np.linalg.norm(residual(u), np.inf))
    print(' or nodally Linf:', np.linalg.norm(u - dirichlet, np.inf))
    mesh.plot(u).get_figure().savefig(str(output_dir.joinpath('solution.png')))
else:
    print(result)
