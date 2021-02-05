from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from skfem import MeshQuad
from skfem.io.json import to_file


@dataclass(frozen=True)
class Cylinder:
    radius: float = 0.05
    height: float = 0.41
    length: float = 2.2
    centre: tuple[float] = (0.2, 0.2)

    @cached_property
    def lcar(self) -> float:
        return self.radius / 2

    @cached_property
    def mesh(self) -> MeshQuad:
        raise NotImplementedError

    def save(self) -> None:
        to_file(self.mesh, Path(__file__).with_suffix(".json"))
