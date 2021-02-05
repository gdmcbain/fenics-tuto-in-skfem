from functools import cached_property

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry
from skfem import MeshQuad
from skfem.importers.meshio import from_meshio

from cylinder import Cylinder


class CylinderPygmsh(Cylinder):

    @cached_property
    def geometry(self) -> Geometry:
        geom = Geometry()
        cylinder = geom.add_circle([*self.centre, 0.0], self.radius, lcar=self.lcar)
        channel = geom.add_rectangle(
            0.0, self.length, 0.0, self.height, 0, holes=[cylinder], lcar=self.lcar
        )
        geom.add_physical(channel.surface, "domain")
        geom.add_physical(channel.lines[1], "outlet")
        geom.add_physical(channel.lines[3], "inlet")

        geom.add_raw_code("Mesh.RecombinationAlgorithm=2;\n")
        geom.add_raw_code(f"Recombine Surface {{{channel.surface.id}}};\n")

        return geom

    @cached_property
    def mesh(self) -> MeshQuad:
        return from_meshio(generate_mesh(self.geometry, dim=2))


if __name__ == "__main__":

    from pathlib import Path
    
    cylinder = CylinderPygmsh()
    with open(Path(__file__).with_suffix(".geo"), "w") as fgeo:
        fgeo.write(cylinder.geometry.get_code())
