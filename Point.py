from dataclasses import dataclass
from typing import List
import math

@dataclass
class Point:
  x: float
  y: float
  
  def __repr__(self):
    return f"POINT({self.x:.3f} {self.y:.3f})"

  def __key(self):
    return (self.x, self.y, self.z)

  def __hash__(self):
    return hash(self.__key())

  def __eq__(self, other):
    if not isinstance(other,Point): return NotImplemented
    precision = 5
    c1 = round(self.x,precision) == round(other.x,precision)
    c2 = round(self.y,precision) == round(other.y,precision)
    return c1 and c2
  
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