from pathlib import Path

from matplotlib.pyplot import subplots, pause
import numpy as np

from skfem import *
from skfem.models.poisson import laplace, mass

a = 5.

nx = ny = 30

time_end = 2.0
num_steps = 50
dt = time_end / num_steps

mesh = (MeshLine(np.linspace(-2, 2, nx + 1))
        * MeshLine(np.linspace(-2, 2, ny + 1)))._splitquads()
basis = InteriorBasis(mesh, ElementTriP1())

boundary = basis.get_dofs().all()
interior = basis.complement_dofs(boundary)

M = asm(mass, basis)
A = M + dt * asm(laplace, basis)

fig, ax = subplots()

t = 0.
u = np.exp(-a * (np.sum(mesh.p**2, axis=0)))  # initial condition, P1 only

output_dir = Path('heat_gaussian')
try:
    output_dir.mkdir()
except FileExistsError:
    pass

for i in range(num_steps + 1):

    ax.cla()
    ax.axis('off')
    fig.suptitle('t = {:.4f}'.format(t))
    mesh.plot(u, ax=ax, zlim=(0, 1))
    if t == 0.:
        fig.colorbar(ax.get_children()[0])
        fig.savefig('initial.png')
    fig.show()
    pause(.01)

    t += dt
    b = M @ u

    u[boundary] = 0.
    u[interior] = solve(*condense(A, b, u, D=boundary))
    
    mesh.save(str(output_dir.joinpath(f'solution{i:06d}.vtk')), u)
