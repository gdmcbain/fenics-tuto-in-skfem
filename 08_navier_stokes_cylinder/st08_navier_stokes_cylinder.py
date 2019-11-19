from pathlib import Path

from matplotlib.pyplot import pause, subplots
import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from meshio import XdmfTimeSeriesWriter

import skfem
from skfem.importers.meshio import from_meshio
from skfem.models.general import divergence
from skfem.models.poisson import laplace, vector_laplace


@skfem.bilinear_form
def vector_mass(u, du, v, dv, w):
    return sum(v * u)


@skfem.bilinear_form
def port_pressure(u, du, v, dv, w):
    return sum(v * (u * w.n))


radius = .05
height = .41

geom = Geometry()
cylinder = geom.add_circle([0.2, 0.2, 0.0], radius, lcar=radius/2)
channel = geom.add_rectangle(.0, 2.2, .0, height, 0, holes=[cylinder],
                             lcar=radius/2)
geom.add_physical(channel.surface, 'domain')
geom.add_physical(channel.lines[1], 'outlet')
geom.add_physical(channel.lines[3], 'inlet')

mesh = from_meshio(generate_mesh(geom, dim=2))

element = {'u': skfem.ElementVectorH1(skfem.ElementTriP2()),
           'p': skfem.ElementTriP1()}
basis = {**{v: skfem.InteriorBasis(mesh, e, intorder=4)
            for v, e in element.items()},
         'inlet': skfem.FacetBasis(mesh, element['u'],
                                   facets=mesh.boundaries['inlet'])}
M = skfem.asm(vector_mass, basis['u'])
L = {'u': skfem.asm(vector_laplace, basis['u']),
     'p': skfem.asm(laplace, basis['p'])}
B = -skfem.asm(divergence, basis['u'], basis['p'])
port_facets = np.concatenate([mesh.boundaries['inlet'],
                              mesh.boundaries['outlet']])
P = B.T + skfem.asm(port_pressure,
                    *(skfem.FacetBasis(mesh, element[v],
                                       facets=port_facets, intorder=3)
                      for v in ['p', 'u']))

t_final = 5.
dt = t_final / 5000

mu = .001
rho = 1.

K = M / dt + mu * L['u']

uv_, p_ = (np.zeros(basis[v].N) for v in element.keys())  # penultimate
p__ = np.zeros_like(p_)  # antepenultimate

dirichlet = {'u': np.setdiff1d(
    basis['u'].get_dofs().all(),
    basis['u'].get_dofs(mesh.boundaries['outlet']).all()),
             'p': basis['p'].get_dofs(
                 np.concatenate([mesh.boundaries['inlet'],
                                 mesh.boundaries['outlet']]))}
inlet_pressure_dofs = basis['p'].get_dofs(mesh.boundaries['inlet']).all()
p_inlet = 1.                    # XXX
# uv0 = np.zeros(basis['u'].N)
# inlet_dofs = basis['u'].get_dofs(mesh.boundaries['inlet']).all('u^1')
# inlet_y_lim = [p.x[1] for p in channel.lines[3].points[::-1]]
# monic = np.polynomial.polynomial.Polynomial.fromroots(inlet_y_lim)
# uv0[inlet_dofs] = (monic(basis['u'].doflocs[1, inlet_dofs]) /
#                    monic(np.mean(inlet_y_lim)))


def embed(xy: np.ndarray) -> np.ndarray:
    return np.pad(xy, ((0, 0), (0, 1)), 'constant')

with XdmfTimeSeriesWriter(Path(__file__).with_suffix('.xdmf').name) as writer:

    writer.write_points_cells(embed(mesh.p.T), {'triangle': mesh.t.T})

    fig, ax = subplots()
    ax.axis('off')
    
    t = 0.
    while t < t_final:
        t += dt

        # Step 1: momentum prediction

        uv = skfem.solve(*skfem.condense(
            K, (M / dt) @ uv_ - P @ (2 * p_ - p__),
            np.zeros_like(uv_), D=dirichlet['u']))

        # Step 2: pressure correction

        dp = np.zeros(basis['p'].N)
        dp[inlet_pressure_dofs] = p_inlet - p_[inlet_pressure_dofs]
        dp = skfem.solve(*skfem.condense(L['p'], B @ uv, dp,
                                         D=dirichlet['p']))


        # Step 3: velocity correction

        p = p_ + dp
        du = skfem.solve(*skfem.condense(M / dt, -P @ dp, D=dirichlet['u']))
        u = uv + du

        uv_ = uv
        p_, p__ = p, p_

        # postprocessing
        
        writer.write_data(
            t,
            point_data={'pressure': p,
                        'velocity': embed(uv[basis['u'].nodal_dofs].T)})

        print(t, min(u[::2]), '<= u <= ', max(u[::2]),
              ',', min(p), '<= p <=', max(p))

        ax.cla()
        fig.suptitle(f't = {t:4f}')
        mesh.plot(p, ax=ax)
        ax.set_aspect(1.)
        ax.set_axis_off()
        if t == dt:
            fig.colorbar(ax.get_children()[0])
        fig.show()
        pause(.5)
