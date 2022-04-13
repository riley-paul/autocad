from autocad.geometry import Point,Line

from autocad.helpers import QuadTree,find_boundary

from dataclasses import dataclass
from typing import List

@dataclass
class Polygon:
  vertices: List[Point]

  def __post_init__(self):
    assert type(self.vertices) == list, "Vertices must a list of points"
    assert len(self.vertices) >= 3, "Polygon Error: Needs more points"
    
    if self.vertices[0] == self.vertices[-1]: self.vertices = self.vertices[:-1]
    self.segments = [Line(self.vertices[i-1],v) for i,v in enumerate(self.vertices)]
    self.qtree = QuadTree(find_boundary(self.vertices))
    for i in self.vertices: self.qtree.insert(i)

  def __repr__(self):
    points = [f"{i.x} {i.y} {i.z}" for i in self.vertices]
    return f"POLYGON(({', '.join(points)}))"

  def area(self) -> float:
    area = 0
    j = len(self.vertices) - 1
    for i in range(len(self.vertices)):
      p1 = self.vertices[j]
      p2 = self.vertices[i]
      area += (p1.x + p2.x) * (p1.y - p2.y)
      j = i
    return abs(area)/2

  # def vis_center(self):
  #   x,y = polylabel([[[pt.x,pt.y] for pt in self.vertices]])
  #   self.vis_pt = Point(x,y,label=self.label)
  #   return self.vis_pt

  def within(self,node) -> bool:
    outside = Point(min(i.x for i in self.vertices),min(i.y for i in self.vertices)).copy(-10,-10)
    line = Line(outside,node)
    intersections = line.intersects_pl(self)
    return len(intersections) % 2 == 1 # odd -> inside, even -> outside
  
  def envelope(self):
    pt_min = Point(min(i.x for i in self.vertices),min(i.y for i in self.vertices))
    pt_max = Point(max(i.x for i in self.vertices),max(i.y for i in self.vertices))
    return pt_min.pt_to_pt(pt_max)

  def centroid(self):
    x = sum(i.x for i in self.vertices)/len(self.vertices)
    y = sum(i.y for i in self.vertices)/len(self.vertices)
    return Point(x,y)

  def intersects_pl(self,other):
    intersections = [i.intersects(j) for i in self.segments for j in other.segments]
    return [i for i in intersections if i]

  def KML_coords(self):
    vertices = self.vertices + [self.vertices[0]]
    return [(i.x,i.y) for i in vertices]

  def ACAD_solid(self):
    assert len(self.vertices) == 3 or len(self.vertices) == 4, "Wrong number of vertices"
    command = "SOLID "
    command += f"{self.vertices[0].comm()} "
    command += f"{self.vertices[1].comm()} "
    command += f"{self.vertices[-1].comm()}"
    if len(self.vertices) > 3:
      command += f" {self.vertices[2].comm()}"
      command += "\n"
    else:
      command += "\n\n"
    return [command]

  def ACAD_mpoly(self):
    command = "MPOLYGON"
    for pt in self.vertices:
      command += f" {pt.comm()}"
    command += "\n"
    command += "C\nX"
    return [command]

  def ACAD(self):
    """Outputs command list to draw the Polyline in AutoCAD"""
    vertices = self.vertices + [self.vertices[0]]
    command = []
    command.append("_.pline")
    command += [f"_non {i.x},{i.y}" for i in vertices]
    command.append("(SCRIPT-CANCEL)")
    return command

  def offset(self,dist):
    from autocad.geometry import join_lines
    lines = [i.offset(dist) for i in self.segments]
    joined = Polygon(join_lines(lines).vertices)
    return joined

  def plot(self,axis,style="k",**kwargs):
    x = [i.x for i in self.vertices] + [self.vertices[0].x]
    y = [i.y for i in self.vertices] + [self.vertices[0].y]
    axis.plot(x,y,style,**kwargs)