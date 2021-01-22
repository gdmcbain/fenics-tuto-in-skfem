from functools import cached_property

import dmsh
from skfem import MeshTri

from cylinder import Cylinder

class CylinderDmsh(Cylinder):

    @cached_property
    def mesh(self) -> MeshTri:
        geo = dmsh.Difference(
            dmsh.Rectangle(0.0, self.length, 0.0, self.height),
            dmsh.Circle(self.centre, self.radius)
        )

        points, triangles = dmsh.generate(geo, 0.025, tol=1e-9)
        m = MeshTri(points.T, triangles.T)
        m.define_boundary("inlet", lambda x: x[0] == .0)
        m.define_boundary("outlet", lambda x: x[0] == self.length)

        return m


if __name__ == "__main__":
    CylinderDmsh().save()
