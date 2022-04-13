from dataclasses import dataclass
from typing import List
import math

@dataclass
class Point:
  x: float
  y: float

  def __repr__(self):
    return f"POINT({self.x:.3f} {self.y:.3f})"

  def move(self, dx=0, dy=0):
    self.x += dx
    self.y += dy

  def copy(self, dx=0, dy=0):
    return Point(self.x + dx, self.y + dy)

  def copy_polar(self,d,theta):
    dx = d*math.cos(math.radians(theta))
    dy = d*math.sin(math.radians(theta))
    return Point(self.x + dx, self.y + dy)

  def pt_to_pt(self, node) -> float:
    delta_x = self.x - node.x
    delta_y = self.y - node.y
    return math.sqrt(delta_x**2 + delta_y**2)

  def ACAD(self) -> List[str]:
    return [f"_.point _non {self.x},{self.y}"]