import math
import shapefile
import numpy as np
import os

from pyproj import Proj,CRS
from dataclasses import dataclass
from typing import List

from helpers.quadtree import *
from polylabel import polylabel

import matplotlib.pyplot as plt

@dataclass
class Point:
  x: float
  y: float
  z: float = 0
  label: any = None
  
  def __repr__(self):
    if self.z == 0: return f"POINT({self.x:.3f} {self.y:.3f})"
    else: return f"POINT({self.x:.3f} {self.y:.3f} {self.z:.3f})"

  def __key(self):
    return (self.x, self.y, self.z)

  def __hash__(self):
    return hash(self.__key())

  def __eq__(self,other):
    if not isinstance(other,Point): return NotImplemented
    precision = 5
    c1 = round(self.x,precision) == round(other.x,precision)
    c2 = round(self.y,precision) == round(other.y,precision)
    return c1 and c2
  
  def move(self,dx=0,dy=0,dz=0):
    self.x += dx
    self.y += dy
    self.z += dz

  def copy(self,dx=0,dy=0,dz=0):
    return Point(self.x + dx,self.y + dy,self.z + dz,label=self.label)

  def copy_polar(self,d,theta):
    dx = d*math.cos(math.radians(theta))
    dy = d*math.sin(math.radians(theta))
    return Point(self.x + dx,self.y + dy,self.z,label=self.label)

  def pt_to_pt(self,node) -> float:
    delta_x = self.x - node.x
    delta_y = self.y - node.y
    return math.sqrt(delta_x**2 + delta_y**2)

  def nearest(self,nodes):
    return min(nodes, key=lambda i: self.pt_to_pt(i))

  def comm(self) -> str:
    return f"{self.x},{self.y}"

  def ACAD(self) -> List[str]:
    return [f"_.point _non {self.x},{self.y},{self.z}"]

  def KML_coords(self) -> list:
    return [(self.x,self.y)]

  def plot(self,axis,style="k",**kwargs):
    axis.plot([self.x],[self.y],style,**kwargs)

  def to_utm(self,proj=None,epsg=26910):
    if not proj:
      crs = CRS.from_epsg(epsg)
      proj = Proj(crs)
    coords = proj(self.x,self.y)
    return Point(coords[0],coords[1])
  
  def to_geo(self,proj=None,epsg=26910):
    if not proj:
      crs = CRS.from_epsg(epsg)
      proj = Proj(crs)
    coords = proj(self.x,self.y,inverse=True)
    return Point(coords[0],coords[1])

@dataclass
class Line:
  p1: Point
  p2: Point

  def __repr__(self):
    return f"LINESTRING({self.p1.x:.3f} {self.p1.y:.3f}, {self.p2.x:.3f} {self.p2.y:.3f})"

  def length(self) -> float:
    return self.p1.pt_to_pt(self.p2)
  
  def angle(self) -> float:
    """Angle from the positive horizontal axis in degrees"""
    dist_x = self.p2.x - self.p1.x
    dist_y = self.p2.y - self.p1.y
    angle = math.atan2(dist_y,dist_x)
    angle = math.degrees(angle)
    return angle

  def reverse(self):
    self.p1,self.p2 = self.p2,self.p1
    return self
  
  def equation_abc(self) -> float:
    # Ax + By + C = 0
    A = self.p1.y - self.p2.y
    B = self.p2.x - self.p1.x
    C = self.p1.x * self.p2.y - self.p2.x * self.p1.y
    return A,B,C
  
  def equation_mb(self) -> float:
    # y = mx + b
    m = (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
    b = self.p1.y - m*self.p1.x
    return m,b

  def which_side(self,node) -> int:
    d = (node.x - self.p1.x)*(self.p2.y - self.p1.y) - (node.y - self.p1.y)*(self.p2.x - self.p1.x)
    if d < 0: return -1
    if d > 0: return 1
    if d == 0: return 0

  def dist_to_pt(self,node,signed=False) -> float:
    A,B,C = self.equation_abc()
    num = abs(A*node.x + B*node.y + C)
    num = num*self.which_side(node) if signed else num
    den = math.sqrt(A**2 + B**2)
    return 0 if den == 0 else num/den
  
  def on_line(self,node):
    """Determine if a point is coincident with the line segment"""
    cross_product = (node.y - self.p1.y) * (self.p2.x - self.p1.x) - (node.x - self.p1.x) * (self.p2.y - self.p1.y)
    if abs(cross_product) > 1e-3: return False

    dot_product = (node.x - self.p1.x) * (self.p2.x - self.p1.x) + (node.y - self.p1.y) * (self.p2.y - self.p1.y)
    if dot_product < 0: return False

    squared_length = (self.p2.x - self.p1.x) * (self.p2.x - self.p1.x) + (self.p2.y - self.p1.y) * (self.p2.y - self.p1.y)
    if dot_product > squared_length: return False

    return True
 
  def move_to_ln(self,node):
    if self.on_line(node): return node
    
    A,B,C = self.equation_abc()
    x = (B*(B*node.x - A*node.y) - A*C) / (A**2 + B**2)
    y = (A*(-B*node.x + A*node.y) - B*C) / (A**2 + B**2)
    
    return Point(x,y,node.z) 

  def along(self,dt):
    """Returns the coordinates of a point a certain distance along the length of the line"""
    # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    # assert dt >= 0, "Distance must be positive"
    d = self.length()
    t = dt/d
    x = (1-t)*self.p1.x + t*self.p2.x
    y = (1-t)*self.p1.y + t*self.p2.y
    return Point(x,y)

  def middle(self):
    return self.along(self.length/2)

  def pt_along(self,node):
    """Returns the length along the line to a point. Starting at P1"""
    new_node = self.move_to_ln(node)
    return new_node.pt_to_pt(self.p1)

  def intersects(self,other,within=True):
    assert isinstance(other,Line), "Other must be a line"
    
    x1,x2 = float(self.p1.x),float(self.p2.x)
    x3,x4 = float(other.p1.x),float(other.p2.x)
    y1,y2 = float(self.p1.y),float(self.p2.y)
    y3,y4 = float(other.p1.y),float(other.p2.y)

    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    try:
      t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))/den
      u = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3))/den
    except: return None

    if (t > 1 or t < 0) and within: return None
    if (u > 1 or u < 0) and within: return None

    Px = self.p1.x + t*(self.p2.x - self.p1.x)
    Py = self.p1.y + t*(self.p2.y - self.p1.y)

    return Point(Px,Py)

  def intersects_pl(self,other):
    assert isinstance(other,Polyline) or isinstance(other,Polygon), "Other must be a polyline"
    intersections = [self.intersects(i) for i in other.segments]
    return [i for i in intersections if i]

  def offset(self,dist):
    angle = self.angle() + 90
    return Line(self.p1.copy_polar(dist,angle),self.p2.copy_polar(dist,angle))

  def buffer(self,dist,rounded=True):
    dist = abs(dist)
    if rounded:
      num_sides = 12
      angle = self.angle() + 90
      angles = [angle + 180/num_sides*i for i in range(num_sides+1)]
      vertices = []
      vertices += [self.p1.copy_polar(dist,a) for a in angles]
      vertices += [self.p2.copy_polar(dist,a+180) for a in angles]
    else:
      l1 = self.offset(dist)
      l2 = self.offset(-dist).reverse()    
      vertices = [l1.p1,l1.p2,l2.p1,l2.p2]
    return Polygon(vertices)

  def compass(self):
    angle = (90 - self.angle()) % 360
    val = int(angle/22.5 + 0.5)
    arr = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return arr[val % 16]
    
  def ACAD(self):
    return [f"_.line _non {self.p1.x},{self.p1.y} _non {self.p2.x},{self.p2.y}\n"]

  def plot(self,axis,style="k",**kwargs):
    axis.plot([self.p1.x,self.p2.x],[self.p1.y,self.p2.y],style,**kwargs)

@dataclass
class Polyline:
  vertices: List[Point]
  KPs: List[Point] = None

  def __post_init__(self):
    assert len(self.vertices) > 1, "Need vertices to make a Polyline"
    self.vertices = list(dict.fromkeys(self.vertices))
    self.segments = [Line(self.vertices[i-1],v) for i,v in enumerate(self.vertices) if i > 0]
    
    self.qtree = QuadTree(find_boundary(self.vertices))
    self.q_radius = max(i.length() for i in self.segments)
    for i in self.vertices: self.qtree.insert(i)
    
    if self.KPs:
      # self.KPs = sorted(self.KPs,key=lambda i: i.label)
      self.qtree_KP = QuadTree(find_boundary(self.KPs))
      for i in self.KPs: self.qtree_KP.insert(i)

  def __repr__(self):
    points = [f"{i.x} {i.y} {i.z}" for i in self.vertices]
    return f"LINESTRING({', '.join(points)})"

  def nearest_vertex(self,node,radius=None):
    radius = self.q_radius if not radius else radius
    points = []
    self.qtree.query_radius(node,radius,points)
    if not points:
      raise ValueError(f"No vertices within a {radius:.2f} radius of {node}")
    return node.nearest(points)

  def remove_vertex(self,index):
    new_vertices = self.vertices.copy()
    del new_vertices[index]
    return Polyline(new_vertices,self.KPs)

  def insert_vertex(self,index,node):
    new_vertices = self.vertices.copy()
    new_vertices.insert(index,node)
    return Polyline(new_vertices,self.KPs)

  def length(self):
    return sum(seg.length() for seg in self.segments)

  def move_to_ln(self,node,radius=None):
    """Returns the coordinates of node if it were moved to be coincident with the line along the shortest possible path"""
    # Determine nearest vertex to the node
    nearest_pt = self.nearest_vertex(node,radius=radius)
    nearest_pt_dist = nearest_pt.pt_to_pt(node)

    # Sort segments by distance to node
    for segment,distance in sorted(zip(self.segments,[i.dist_to_pt(node) for i in self.segments]), key=lambda i: i[1]):
      # If a vertex is the nearest part of the polyline to the node, return vertex
      if distance >= nearest_pt_dist: return nearest_pt
      
      # Move pt to line segment and see if on line
      moved_pt = segment.move_to_ln(node)
      if segment.on_line(moved_pt):
        return moved_pt

    return nearest_pt

  def reverse(self):
    new_vertices = self.vertices
    new_vertices.reverse()
    return Polyline(new_vertices)

  def which_segment(self,node,radius=None):
    """Returns the line segment in the polyline which is both closest to the given point and which the point can be moved perpindicularly to be coincident with"""
    new_pt = self.move_to_ln(node,radius=radius)
    return next((i for i in self.segments if i.on_line(new_pt)),False)

  def along(self,dt):
    """Returns a point at a certain distance along the polyline"""
    assert dt <= self.length(), "Distance must be less than polyline length"
    assert dt >= 0, "Distance must be positive"
    length = 0
    for segment in self.segments:
      length += segment.length()
      if length >= dt:
        return segment.along(dt - (length - segment.length()))
  
  def on_line(self,node):
    """Determine if node is coincident with Polyline"""
    segment = self.which_segment(node)
    if segment: return segment.on_line(node)
    return False

  def dist_to_ln(self,node,signed=False):
    segment = self.which_segment(node)
    distance = segment.dist_to_pt(node)
    if signed: distance = distance * segment.which_side(node)
    return distance

  def which_side(self,node):
    segment = self.which_segment(node)
    return segment.which_side(node)

  def pt_along(self,node,radius=None):
    """Returns distance from the start of the Polyline to a point when moving along it"""
    segment = self.which_segment(node,radius=radius)
    index = self.segments.index(segment)
    length = sum([seg.length() for seg in self.segments[:index]])
    length += segment.pt_along(node)
    return length

  def pt_to_pt_along(self,p1,p2):
    """Returns the distance between two points when moving along the Polyline"""
    d1 = self.pt_along(p1)
    d2 = self.pt_along(p2)
    return abs(d1 - d2)

  def splice(self,p1,p2,radius=None):
    assert isinstance(p1,Point), "p1 not a point"
    assert isinstance(p2,Point), "p2 not a point"

    p1.segment = self.which_segment(p1,radius=radius)
    p2.segment = self.which_segment(p2,radius=radius)

    p1.moved = p1.segment.move_to_ln(p1)
    p2.moved = p2.segment.move_to_ln(p2)

    # if p1.moved == p2.moved: return None

    p1.index = self.segments.index(p1.segment)
    p2.index = self.segments.index(p2.segment)

    if p1.index <= p2.index: pt_beg,pt_end = p1,p2
    else: pt_end,pt_beg = p1,p2

    vertices = []
    vertices.append(pt_beg.moved)
    index = pt_beg.index

    while index < pt_end.index:
      vertices.append(self.segments[index].p2)
      index += 1

    vertices.append(pt_end.moved)
    return Polyline(vertices)

  def splice_circle(self,center,radius):
    circle = Circle(center,radius)
    points = circle.intersects_pl(self)
    num_pts = len(points)

    if num_pts == 0: return None
    elif num_pts >= 2: return self.splice(points[0],points[-1])
    elif num_pts == 1 and circle.within(self.vertices[0]): return self.splice(self.vertices[0],points[0])
    elif num_pts == 1 and circle.within(self.vertices[-1]): return self.splice(points[0],self.vertices[-1])
    # else:
    #   print("Weird number of circle intersections")
    #   return None

  def avg_angle(self):
    """Flawed"""
    tans = [math.tan(math.radians(i.angle())) for i in self.segments]
    avg = sum(tans)/len(tans)
    return math.degrees(math.atan(avg))

  def find_KP(self,node,radius=1000):
    """Returns the exact chainage of a point near the Polyline"""
    assert isinstance(node,Point), "Node must be a Point"

    points = []
    self.qtree_KP.query_radius(node,radius,points) # Determine the nearest KP point
    if points:
      nearest_KP = node.nearest(points)
    else:
      print(f"Point is further than {radius}m from CL")
      return None
    
    along_pt = self.pt_along(node,radius=radius)
    along_KP = self.pt_along(nearest_KP,radius=radius)
    diff = along_pt - along_KP # Find the distance between the nearest KP and the node
    chainage = nearest_KP.label + diff # Add the distance to the nearest KP
    return chainage

  def from_KP(self,chainage,extend=False,radius=1000):
    """Returns a point on the Polyline at the given chainage"""
    max_KP = max(i.label for i in self.KPs)
    min_KP = min(i.label for i in self.KPs)

    assert type(chainage) == float or int, "KP must be a number"

    if chainage > max_KP or chainage < min_KP:
      if extend == False:
        print(f"{format_KP(chainage)} not in project")
        return None
      elif chainage < min_KP:
        old_line = self.segments[0]
        length = old_line.length() + abs(chainage - min_KP)
        return self.vertices[1].copy_polar(-length,old_line.angle())
      elif chainage > max_KP:
        old_line = self.segments[-1]
        length = old_line.length() + abs(chainage - max_KP)
        return self.vertices[-2].copy_polar(length,old_line.angle())
  
    nearest_KP = min(self.KPs, key=lambda x: abs(x.label-chainage))
    nearest_i = self.KPs.index(nearest_KP)
    
    if nearest_KP.label < chainage or nearest_KP.label == min_KP:
      lower_KP = nearest_KP
      upper_KP = self.KPs[nearest_i + 1]
    else:
      lower_KP = self.KPs[nearest_i - 1]
      upper_KP = nearest_KP

    for entry in self.KPs:
      if entry.label == chainage:
        return entry
    
    # Create a polyline between the upper and lower bounding points
    # Determine the point as a percentage of the distance between the bounds

    temp_pl = self.splice(lower_KP,upper_KP,radius=radius)

    KP_div = abs(upper_KP.label - lower_KP.label)
    length = temp_pl.length()
    delta = chainage - lower_KP.label
    node = temp_pl.along(delta / KP_div * length)
    node.label = chainage

    return node

  def perp_angle(self,node):
    if node in self.vertices:
      index = self.vertices.index(node)
      if node == self.vertices[0]: return self.segments[0].angle() + 90
      if node == self.vertices[-1]: return self.segments[0].angle() + 90
      s1 = self.segments[index-1]
      s2 = self.segments[index]
      angle3 = avg_lines([s1,s2]).angle() + 90
      return angle3
    return self.which_segment(node).angle() + 90    

  def intersects_pl(self,other):
    inter = []
    for segment in self.segments:
      intersections = [segment.intersects(i) for i in other.segments]
      intersections = [i for i in intersections if i]
      intersections.sort(key=lambda i: segment.pt_along(i))
      inter += intersections
    return inter
  
  def offset_simple(self,dist):
    if dist == 0: return self
    offset_lines = [i.offset(dist) for i in self.segments]
    return join_lines(offset_lines)

  def offset(self,dist):
    if dist == 0: return self
    offset = self.offset_simple(dist)

    inter = [offset.vertices[0]]
    inter += offset.intersects_pl(offset)
    inter += [offset.vertices[-1]]

    inter_self = [i for i in inter if i not in offset.vertices]
    overlaps = []
    for i in inter_self:
      i1 = inter.index(i)
      i2 = inter.index(i,i1+1)
      overlap = (i1,i2)
      if overlap not in overlaps: overlaps.append(overlap)

    overlaps = [i for i in overlaps if not any((i[0] > j[0] and i[1] < j[1]) for j in overlaps)]
    inter = [i for index,i in enumerate(inter) if not any((index >= j[0] and index < j[1]) for j in overlaps)]
    return Polyline(inter)

  def buffer(self,dist):
    p1 = self.offset(dist).vertices
    p2 = self.offset(-dist).vertices
    p2.reverse()
    return Polygon(p1+p2)

  def lin_divide(self,num_div):
    alongs = np.linspace(0,self.length(),num_div)
    return [self.along(i) for i in alongs]

  def splice_KP(self,KP_beg,KP_end):
    pt_beg = self.from_KP(KP_beg,True)
    pt_end = self.from_KP(KP_end,True)
    new_KPs = [i for i in self.KPs if (i.label <= KP_end and i.label >= KP_beg)]
    new_vertices = self.splice(pt_beg,pt_end).vertices
    return Polyline(new_vertices,new_KPs)

  def elevation_at_pt(self,node):
    segment = self.which_segment(node)
    new_node = segment.move_to_ln(node)
    if not segment.on_line(new_node): return None
    
    tot_length = segment.length()
    pt_length = segment.pt_along(node)
    tot_vert = segment.p2.z - segment.p1.z
    pt_vert = pt_length * tot_vert / tot_length
    
    return pt_vert + segment.p1.z

  def ACAD(self,inc_elevation=False):
    """Outputs command list to draw the Polyline in AutoCAD"""
    command = []
    command.append("_.pline")
    for index,vertex in enumerate(self.vertices):
      if index == 0 and inc_elevation:
        command.append(f"_non {vertex.x},{vertex.y},{vertex.z}")
        continue
      command.append(f"_non {vertex.x},{vertex.y}")
    command.append("(SCRIPT-CANCEL)")
    return command

  def KML_coords(self):
    return [(i.x,i.y) for i in self.vertices]

  def plot(self,axis,style="k",**kwargs):
    axis.plot([i.x for i in self.vertices],[i.y for i in self.vertices],style,**kwargs)

  def to_geo(self,proj=None,epsg=26910):
    if not proj:
      crs = CRS.from_epsg(epsg)
      proj = Proj(crs)
    coords = [proj(i.x,i.y,inverse=True) for i in self.vertices]
    points = [Point(i[0],i[1]) for i in coords]
    return Polyline(points)

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

  def vis_center(self):
    x,y = polylabel([[[pt.x,pt.y] for pt in self.vertices]])
    self.vis_pt = Point(x,y,label=self.label)
    return self.vis_pt

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

  def save(self):
    pts = self.vertices
    if pts[0] != pts[-1]: pts.append(pts[0])
    pts_text = [f"{pt.x} {pt.y}" for pt in pts]
    return ", ".join(pts_text)

  def to_geo(self,proj=None,epsg=26910):
    if not proj:
      crs = CRS.from_epsg(epsg)
      proj = Proj(crs)
    coords = [proj(i.x,i.y,inverse=True) for i in self.vertices]
    points = [Point(i[0],i[1]) for i in coords]
    return Polygon(points)

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
    lines = [i.offset(dist) for i in self.segments]
    joined = Polygon(join_lines(lines).vertices)
    return joined

  def plot(self,axis,style="k",**kwargs):
    x = [i.x for i in self.vertices] + [self.vertices[0].x]
    y = [i.y for i in self.vertices] + [self.vertices[0].y]
    axis.plot(x,y,style,**kwargs)

@dataclass
class Circle:
  origin: Point
  radius: float

  def within(self,node):
    return self.origin.pt_to_pt(node) < self.radius

  def to_poly(self,res=10):
    # res = approximate distance between points
    circumference = math.pi * 2 * self.radius
    num_points = int(circumference/res)
    angle = 360/num_points
    pts = [self.origin.copy_polar(self.radius,angle*i) for i in range(num_points)]
    return Polygon(pts)
 
  def intersects(self,line,segment=True):
    m,d = line.equation_mb()
    a,b,r = self.origin.x,self.origin.y,self.radius

    delta = r**2*(1 + m**2) - (b - m*a - d)**2

    if delta < 0: return []

    x1 = (a + b*m - d*m + math.sqrt(delta)) / (1 + m**2)
    x2 = (a + b*m - d*m - math.sqrt(delta)) / (1 + m**2)
    y1 = (d + a*m + b*m**2 + m*math.sqrt(delta)) / (1 + m**2)
    y2 = (d + a*m + b*m**2 - m*math.sqrt(delta)) / (1 + m**2)

    points = [Point(x1,y1),Point(x2,y1),Point(x1,y2),Point(x2,y2)]
    points = [i for i in points if i.pt_to_pt(self.origin) <= r*1.01 and i.pt_to_pt(self.origin) >= r*0.99]
    points = [i for i in points if line.on_line(i)] if segment else points
    return points

  def intersects_pl(self,other):
    assert isinstance(other,Polyline), "Other must be a polyline"
    intersections = []
    for seg in other.segments:
      intersections += self.intersects(seg,segment=True)
    return intersections

  def ACAD(self):
    command = []
    command += [f"_.circle _non {self.origin.comm()} {self.radius}"]
    return command

  def plot(self,axis,style="k",**kwargs):
    circle = plt.Circle((self.origin.x,self.origin.y),self.radius,color=style,fill=False,**kwargs)
    axis.add_patch(circle)

@dataclass
class Layer:
  name: str
  colour: any = None
  ltype: str = None
  lweight: str = None

  def __post_init__(self):
    bad_characters = ["\\","/"]
    for i in bad_characters:
      self.name = self.name.replace(i,"_")

  def ACAD(self) -> List[str]:
    """Produces a list of commands to create a layer in AutoCAD"""
    command = []
    command.append("-LAYER")
    command.append(f"M \"{self.name}\"")
    if self.colour: 
      try:
        if type(self.colour) == int or type(self.colour) == float:
          self.colour = int(self.colour)
          if self.colour <= 255: 
            command.append(f"C {self.colour} \"{self.name}\"")
          else:
            command.append(f"C T {self.colour:,} \"{self.name}\"")

        if type(self.colour) == str:
          if len(self.colour) <= 3:
            command.append(f"C {self.colour} \"{self.name}\"")
          else:
            command.append(f"C T {self.colour} \"{self.name}\"")
      
      except Exception as e:
        print(e)

    if self.ltype: command.append(f"L {self.ltype} \"{self.name}\"")
    if self.lweight: command.append(f"LW {self.lweight} \"{self.name}\"")
    command.append("(SCRIPT-CANCEL)") 
    return command

  def to_front(self) -> List[str]:
    return [f"(ssget \"X\" '((8 . \"{self.name}\")))\nDRAWORDER P\n\nF"]
  
  def to_back(self) -> List[str]:
    return [f"(ssget \"X\" '((8 . \"{self.name}\")))\nDRAWORDER P\n\nB"]

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

@dataclass
class Page:
  name: str = None
  angle: float = 0
  scale: float = 1
  origin: Point = Point(0,0)
  x_offset: float = 0
  y_offset: float = 0

  def __post_init__(self):
    # Define transformation matrix of viewport
    # 
    angle = math.radians(self.angle)
    t1 = np.array([[1,0,-self.origin.x],[0,1,-self.origin.y],[0,0,1]])
    t2 = np.array([[1,0,self.x_offset],[0,1,self.y_offset],[0,0,1]])
    r = np.array([[math.cos(angle),math.sin(angle),0],[-math.sin(angle),math.cos(angle),0],[0,0,1]])
    s = np.array([[self.scale,0,0],[0,self.scale,0],[0,0,1]])
    self.T = t2 @ s @ r @ t1

  def transform(self,point,inverse=False):
    T = np.linalg.inv(self.T) if inverse else self.T
    P1 = np.array([[point.x],[point.y],[1]])
    P2 = T @ P1
    return Point(P2[0][0],P2[1][0])

def import_pages(data):
  pages = []
  for _,row in data.iterrows():
    origin = import_WKT(row["ORIGIN"])
    shape = import_WKT(row["GEOMETRY"])
    shape_vp = import_WKT(row["GEOMETRY_VP"])

    page = Page(
      name = row["NAME"],
      angle = row["ANGLE"],
      scale = row["SCALE"],
      origin = origin,
      x_offset = shape_vp[0].vertices[0].x,
      y_offset = shape_vp[0].vertices[0].y,
    )
    
    page.KP_beg = row["KP_beg"]
    page.KP_end = row["KP_end"]
    page.shape = shape
    page.shape_vp = shape_vp[0]
    page.section = row["SECTION"]

    pages.append(page)

  return pages

def import_WKT(text):
  if "MULTIPOINT" in text:
    if "((" in text: result = text[text.index("((")+2:text.index("))")].split("), (")
    else: result = text[text.index("(")+1:text.index(")")].split(", ")
    coords = [i.strip().split(" ") for i in result]
    points = [Point(float(i[0]),float(i[1])) for i in coords]
    return points

  elif "MULTILINESTRING" in text:
    result = text[text.index("((")+2:text.index("))")].split("), (")
    result = [i.strip().split(", ") for i in result]
    coords = [[i.split(" ") for i in j] for j in result]
    lines = [Polyline(vertices=[Point(float(i[0]),float(i[1])) for i in j]) for j in coords]
    return lines
  
  elif "MULTIPOLYGON" in text:
    text = text[text.index("(((")+3:text.index(")))")]
    result = text.split(")),((")
    result = [i.split("),(") for i in result]
    result = [i.split(",") for j in result for i in j]
    polygons = []
    for pg in result:
      coords = [Point(float(i.strip().split(" ")[0]),float(i.strip().split(" ")[1])) for i in pg]
      polygons.append(Polygon(vertices=coords))
    return polygons

  elif "POINT" in text:
    result = text[text.index("(")+1:text.index(")")].split(" ")
    coords = [float(i.strip()) for i in result if i]
    if len(coords) == 2: return Point(coords[0],coords[1])
    else: return Point(coords[0],coords[1],coords[2])

  elif "LINESTRING" in text:
    result = text[text.index("(")+1:text.index(")")].split(", ")
    coords = [i.strip().split(" ") for i in result]
    points = [Point(float(i[0]),float(i[1])) if len(i) == 2 else Point(float(i[0]),float(i[1]),float(i[2])) for i in coords]
    return Polyline(vertices=points)

  elif "POLYGON" in text:
    result = text[text.index("((")+2:text.index("))")].split("), (")
    result = [i.strip().split(", ") for i in result]
    coords = [[i.split(" ") for i in j] for j in result]
    polygons = [Polygon(vertices=[Point(float(i[0]),float(i[1])) for i in j]) for j in coords]
    # text = text[text.index("((")+2:text.index("))")]
    # result = text.split("),(")
    # polygons = []
    # for pg in result:
    #   coords = [Point(float(i.strip().split(" ")[0]),float(i.strip().split(" ")[1])) for i in pg.split(",")]
    #   polygons.append(Polygon(vertices=coords))
    # # polygons = [i for j in polygons for i in j if type(j)==list]
    return polygons

  print("Not a valid WKT format")
  return None

def avg_points(nodes):
  x = [i.x for i in nodes]
  y = [i.y for i in nodes]
  return Point(sum(x)/len(x),sum(y)/len(y))

def avg_lines(lines):
  p1 = [i.p1 for i in lines]
  p2 = [i.p2 for i in lines]
  return Line(avg_points(p1),avg_points(p2))

def join_lines(lines) -> Polyline:
  pts = []
  pts.append(lines[0].p1)
  for i in range(1,len(lines)):
    intersection = lines[i-1].intersects(lines[i],within=False)
    pts.append(intersection)
  pts.append(lines[-1].p2)
  return Polyline(pts)

def pts_to_pl(pts,thresh) -> Polyline:
  # Assumes that points are already ordered correctly
  polylines = []
  current = [pts.pop(0)]
  while pts:
    if pts[0].pt_to_pt(current[-1]) < thresh:
      current.append(pts.pop(0))
    else:
      polylines.append(current)
      current = [pts.pop(0)]
  polylines.append(current)
  polylines = [Polyline(vertices=i) for i in polylines if len(i) > 1]
  return polylines

def center_rect(center:Point,angle:float,width:float,height:float) -> Polygon:
  p1 = center.copy_polar(height/2,angle)
  p2 = center.copy_polar(-height/2,angle)
  l1 = Line(p1,p2).offset(width/2)
  l2 = Line(p1,p2).offset(-width/2).reverse()
  return Polygon([l1.p1,l1.p2,l2.p1,l2.p2])

def print_to_file(fname,data):
  """Print list to text file"""
  with open(fname, "w", encoding="utf-16") as file:
    for item in data:
      file.write("%s\n" % item)
 
def format_KP(number,comma=False) -> str:
  """Formats chainage number as '#+###' e.g. 34032.43 to 34+032"""
  if type(number) == int or float:
    post_plus = number % 1000
    pre_plus = (number - post_plus)/1000
    return f"{pre_plus:,.0f}+{post_plus:03.0f}" if comma else f"{pre_plus:.0f}+{post_plus:03.0f}"
  else:
    return number

def import_CL(which="TM5B_006") -> Polyline:
  # points.shp and line.shp in same directory
  # points.shp must have field named KP with chainages in m
  centerlines = {
    "TM5B_006": {
      "name": "Trans Mountain Expansion - Spread 5B",
      "folder": "TMEP_S5_5.24.006",
      "version": "5.24.006",
    },
    "TM5B_005": {
      "name": "Trans Mountain Expansion - Spread 5B",
      "folder": "TMEP_S5_5.24.005",
      "version": "5.24.005",
    },
    "TM5B_002": {
      "name": "Trans Mountain Expansion - Spread 5B",
      "folder": "TMEP_S5_5.24.002",
      "version": "5.24.005",
    },
    "TML1": {
      "name": "Trans Mountain Pipeline",
      "folder": "TMPL_ED41.35628",
      "version": "ED41.35628",
    },
    "CGL34": {
      "name": "Coastal Gaslink - Spreads 3 & 4",
      "folder": "CGL_S34_R10",
      "version": "R10",
    },
  }

  path = os.path.join("/Users","riley","Google Drive","CODE","DATA","Pipeline Centerlines",centerlines[which]["folder"])

  print(f"\nimporting {centerlines[which]['name']} centerline")
  fname = os.path.join(path,"points")
  with shapefile.Reader(fname) as shp:
    KP_index = [i[0] for i in shp.fields].index("KP") - 1
    KPs = []
    for shp_rcd in shp.iterShapeRecords():
      coords_x = shp_rcd.shape.points[0][0]
      coords_y = shp_rcd.shape.points[0][1]
      label = float(shp_rcd.record[KP_index])
      KPs.append(Point(coords_x,coords_y,label=label))
    KPs.sort(key = lambda i: i.label)
    print(f"{len(KPs)} kilometer points imported")

  fname = os.path.join(path,"line")
  with shapefile.Reader(fname) as shp:
    points = [Point(i[0],i[1]) for i in shp.shapes()[0].points]
    print(f"{len(points)} vertices imported","\n")
  
  pl = Polyline(points,KPs)
  pl.version = centerlines[which]["version"]
  pl.name = centerlines[which]["name"]
  return pl