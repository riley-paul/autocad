from autocad.geometry import Point
from autocad.geometry import Polyline
from autocad.geometry import Polygon

import math
from dataclasses import dataclass


@dataclass
class Circle:
    origin: Point
    radius: float

    def within(self, node):
        return self.origin.pt_to_pt(node) < self.radius

    def to_poly(self, res=10):
        # res = approximate distance between points
        circumference = math.pi * 2 * self.radius
        num_points = int(circumference / res)
        angle = 360 / num_points
        pts = [
            self.origin.copy_polar(self.radius, angle * i) for i in range(num_points)
        ]
        return Polygon(pts)

    def intersects(self, line, segment=True):
        m, d = line.equation_mb()
        a, b, r = self.origin.x, self.origin.y, self.radius

        delta = r**2 * (1 + m**2) - (b - m * a - d) ** 2

        if delta < 0:
            return []

        x1 = (a + b * m - d * m + math.sqrt(delta)) / (1 + m**2)
        x2 = (a + b * m - d * m - math.sqrt(delta)) / (1 + m**2)
        y1 = (d + a * m + b * m**2 + m * math.sqrt(delta)) / (1 + m**2)
        y2 = (d + a * m + b * m**2 - m * math.sqrt(delta)) / (1 + m**2)

        points = [Point(x1, y1), Point(x2, y1), Point(x1, y2), Point(x2, y2)]
        points = [
            i
            for i in points
            if i.pt_to_pt(self.origin) <= r * 1.01
            and i.pt_to_pt(self.origin) >= r * 0.99
        ]
        points = [i for i in points if line.on_line(i)] if segment else points
        return points

    def intersects_pl(self, other):
        assert isinstance(other, Polyline), "Other must be a polyline"
        intersections = []
        for seg in other.segments:
            intersections += self.intersects(seg, segment=True)
        return intersections

    def ACAD(self):
        command = []
        command += [f"_.circle _non {self.origin.comm()} {self.radius}"]
        return command
