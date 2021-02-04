from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Tuple

import skfem
from skfem.io.json import to_file


@dataclass(frozen=True)
class Cylinder:
    radius: float = 0.05
    height: float = 0.41
    length: float = 2.2
    centre: Tuple[float] = (0.2, 0.2)

    @cached_property
    def lcar(self) -> float:
        return self.radius / 2

    @cached_property
    def mesh(self) -> skfem.MeshTri:
        raise NotImplementedError

    def save(self) -> None:
        to_file(self.mesh, Path(__file__).with_suffix(".json"))
