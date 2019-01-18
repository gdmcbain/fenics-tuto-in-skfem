from matplotlib.pyplot import subplots, pause
import numpy as np

from skfem import *
from skfem.models.poisson import laplace, mass, unit_load

alpha = 3.
beta = 1.2

nx = ny = 2**3

time_end = 2.0
num_steps = 10
dt = time_end / num_steps

mesh = (MeshLine(np.linspace(0, 1, nx + 1))
        * MeshLine(np.linspace(0, 1, ny + 1)))._splitquads()
basis = InteriorBasis(mesh, ElementTriP1())

boundary = basis.get_dofs().all()
interior = basis.complement_dofs(boundary)

M = asm(mass, basis)
A = M + dt * asm(laplace, basis)
f = (beta - 2 - 2 * alpha) * asm(unit_load, basis)

fig, ax = subplots()


def dirichlet(t: float) -> np.ndarray:
    return 1. + [1., alpha] @ mesh.p**2 + beta * t

t = 0.
u = dirichlet(t)

zlim = (0, np.ceil(1 + alpha + beta * time_end))

for i in range(num_steps + 1):
    
    ax.cla()
    ax.axis('off')
    fig.suptitle('t = {:.4f}'.format(t))
    mesh.plot(u, ax=ax, zlim=zlim)
    if t == 0.:
        fig.colorbar(ax.get_children()[0])
    fig.show()
    pause(1.)

    t += dt
    b = dt * f + M @ u
    
    u_D = dirichlet(t)
    u[boundary] = u_D[boundary]
    u[interior] = solve(*condense(A, b, u_D, D=boundary))
    error = np.linalg.norm(u - u_D)
    print('t = %.2f: error = %.3g' % (t, error))
    

