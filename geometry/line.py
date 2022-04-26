from autocad.geometry import Point
from dataclasses import dataclass
import math

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
    
    return Point(x,y) 

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
    intersections = [self.intersects(i) for i in other.segments]
    return [i for i in intersections if i]

  def offset(self,dist):
    angle = self.angle() + 90
    return Line(self.p1.copy_polar(dist,angle),self.p2.copy_polar(dist,angle))

  def compass(self):
    angle = (90 - self.angle()) % 360
    val = int(angle/22.5 + 0.5)
    arr = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return arr[val % 16]
    
  def ACAD(self):
    return [f"_.LINE _NON {self.p1.x},{self.p1.y} _NON {self.p2.x},{self.p2.y}\n"]