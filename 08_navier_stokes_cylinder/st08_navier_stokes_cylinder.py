from pathlib import Path

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem.importers.meshio import from_meshio


radius = .05
height = .41

geom = Geometry()
cylinder = geom.add_circle([0.2, 0.2, 0.0], radius, lcar=radius/2)
channel = geom.add_rectangle(.0, 2.2, .0, height, 0, holes=[cylinder],
                             lcar=radius/2)
geom.add_physical(channel.surface, 'domain')
geom.add_physical(channel.lines[3], 'inlet')

mesh = from_meshio(generate_mesh(geom, dim=2))

ax = mesh.draw()
ax.get_figure().savefig(Path(__file__).with_suffix('.png'))
