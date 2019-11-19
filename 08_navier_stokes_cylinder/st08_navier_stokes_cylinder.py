from pathlib import Path

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

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


ax = mesh.draw()
ax.get_figure().savefig(Path(__file__).with_suffix('.png'))
