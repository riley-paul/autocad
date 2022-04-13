from dataclasses import dataclass

@dataclass
class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""
    cx: float # Center x coordinate
    cy: float # Center y coordinate
    w: float # Total width
    h: float # Total height

    def __post_init__(self):
        self.west_edge = self.cx - self.w/2
        self.east_edge = self.cx + self.w/2
        self.north_edge = self.cy + self.h/2
        self.south_edge = self.cy - self.h/2

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)

    def contains(self,point):
        return (point.x >= self.west_edge and
                point.x < self.east_edge and
                point.y >= self.south_edge and
                point.y < self.north_edge)
    
    def intersects(self,other):
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge < self.south_edge or
                    other.south_edge > self.north_edge)
    
    def plot(self,ax,c="k",lw=0.5,**kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)

@dataclass
class QuadTree:
    boundary: Rect
    capacity: int = 4
    depth: int = 0
    
    def __post_init__(self):
        self.points = []
        self.divided = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + '\n'
        s += sp + ', '.join(str(point) for point in self.points)
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def subdivide(self):
        cx,cy = self.boundary.cx,self.boundary.cy      
        w,h = self.boundary.w/2,self.boundary.h/2

        nw = Rect(cx - w/2, cy + h/2, w, h)
        ne = Rect(cx + w/2, cy + h/2, w, h)
        sw = Rect(cx - w/2, cy - h/2, w, h)
        se = Rect(cx + w/2, cy - h/2, w, h)

        self.nw = QuadTree(nw,self.capacity,self.depth + 1)
        self.ne = QuadTree(ne,self.capacity,self.depth + 1)
        self.sw = QuadTree(sw,self.capacity,self.depth + 1)
        self.se = QuadTree(se,self.capacity,self.depth + 1)
        self.divided = True

    def insert(self,point):
        if not self.boundary.contains(point):
            return False
        
        if len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if not self.divided:
            self.subdivide()
        
        self.nw.insert(point)
        self.ne.insert(point)
        self.sw.insert(point)
        self.se.insert(point)

    def query(self,boundary,found_points):
        if not self.boundary.intersects(boundary):
            return False
        
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)

        if self.divided:
            self.nw.query(boundary, found_points)
            self.ne.query(boundary, found_points)
            self.se.query(boundary, found_points)
            self.sw.query(boundary, found_points)
        return found_points

    def query_circle(self, boundary, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre.

        boundary is a Rect object (a square) that bounds the search circle.
        There is no need to call this method directly: use query_radius.

        """

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary
        # and also lie within a circle of given radius around the centre point.
        for point in self.points:
            if (boundary.contains(point) and point.pt_to_pt(centre) <= radius):
                found_points.append(point)

        # Recurse the search into this node's children.
        if self.divided:
            self.nw.query_circle(boundary, centre, radius, found_points)
            self.ne.query_circle(boundary, centre, radius, found_points)
            self.se.query_circle(boundary, centre, radius, found_points)
            self.sw.query_circle(boundary, centre, radius, found_points)
        return found_points

    def query_radius(self, center, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre."""

        # First find the square that bounds the search circle as a Rect object.
        boundary = Rect(center.x, center.y, 2*radius, 2*radius)
        return self.query_circle(boundary, center, radius, found_points)

    def __len__(self):
        """Returns number of points in quadtree."""
        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints

    def plot(self,ax,style="k",**kwargs):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""
        self.boundary.plot(ax,style,**kwargs)
        if self.divided:
            self.nw.plot(ax,style,**kwargs)
            self.ne.plot(ax,style,**kwargs)
            self.sw.plot(ax,style,**kwargs)
            self.se.plot(ax,style,**kwargs)

def find_boundary(points):
    x_min = min(points,key=lambda i: i.x).x
    x_max = max(points,key=lambda i: i.x).x
    y_min = min(points,key=lambda i: i.y).y
    y_max = max(points,key=lambda i: i.y).y

    cx,cy = (x_max + x_min)/2,(y_min + y_max)/2
    w,h = abs(x_max - x_min)*1.01,abs(y_max - y_min)*1.01
    return Rect(cx,cy,w,h)