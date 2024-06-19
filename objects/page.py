from dataclasses import dataclass
import numpy as np
import math

from autocad.geometry import Point
from autocad.helpers import import_WKT


@dataclass
class Page:
    name: str = None
    angle: float = 0
    scale: float = 1
    origin: Point = Point(0, 0)
    x_offset: float = 0
    y_offset: float = 0

    def __post_init__(self):
        # Define transformation matrix of viewport
        #
        angle = math.radians(self.angle)
        t1 = np.array([[1, 0, -self.origin.x], [0, 1, -self.origin.y], [0, 0, 1]])
        t2 = np.array([[1, 0, self.x_offset], [0, 1, self.y_offset], [0, 0, 1]])
        r = np.array(
            [
                [math.cos(angle), math.sin(angle), 0],
                [-math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        s = np.array([[self.scale, 0, 0], [0, self.scale, 0], [0, 0, 1]])
        self.T = t2 @ s @ r @ t1

    def transform(self, point, inverse=False):
        T = np.linalg.inv(self.T) if inverse else self.T
        P1 = np.array([[point.x], [point.y], [1]])
        P2 = T @ P1
        return Point(P2[0][0], P2[1][0])

    def select(self):
        return [f'_.-LAYOUT S "{self.name}"']


def import_pages(data):
    pages = []
    for _, row in data.iterrows():
        origin = import_WKT(row["origin"])
        shape = import_WKT(row["geometry"])
        shape_vp = import_WKT(row["geometry_vp"])

        page = Page(
            name=row["name"],
            angle=row["angle"],
            scale=row["scale"],
            origin=origin,
            x_offset=shape_vp[0].vertices[0].x,
            y_offset=shape_vp[0].vertices[0].y,
        )

        page.KP_beg = row["KP_beg"]
        page.KP_end = row["KP_end"]
        page.shape = shape
        page.shape_vp = shape_vp[0]

        pages.append(page)

    return pages
