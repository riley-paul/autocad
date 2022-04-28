from dataclasses import dataclass
from typing import List

from autocad.geometry import Point

@dataclass
class Text:
  name: str
  position: Point
  height: float = None
  width: float = None
  just: str = None
  style: str = "Standard"
  rot: float = None
  lspace: float = None

  def __post_init__(self):
    self.name = self.name.replace("\n"," ")

  def ACAD(self) -> List[str]:
    command = f"_.-mtext _non {self.position.x},{self.position.y}"
    command += f" J {self.just}" if self.just else ""
    command += f" S {self.style}" if self.style else ""
    command += f" R {self.rot}" if self.rot else ""
    command += f" LS {self.lspace}" if self.lspace else ""
    command += f" H {self.height}" if self.height else ""
    command += f"\nW {self.width}" if self.width else "\nW 0"
    command += f" {self.name}\n"
    return [command]