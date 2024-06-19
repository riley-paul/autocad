from autocad.geometry import Point, Line, Polyline, Polygon


def avg_points(nodes):
    x = [i.x for i in nodes]
    y = [i.y for i in nodes]
    return Point(sum(x) / len(x), sum(y) / len(y))


def avg_lines(lines):
    p1 = [i.p1 for i in lines]
    p2 = [i.p2 for i in lines]
    return Line(avg_points(p1), avg_points(p2))


def join_lines(lines) -> Polyline:
    pts = []
    pts.append(lines[0].p1)
    for i in range(1, len(lines)):
        intersection = lines[i - 1].intersects(lines[i], within=False)
        pts.append(intersection)
    pts.append(lines[-1].p2)
    return Polyline(pts)


def pts_to_pl(pts, thresh) -> Polyline:
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


def center_rect(center: Point, angle: float, width: float, height: float) -> Polygon:
    p1 = center.copy_polar(height / 2, angle)
    p2 = center.copy_polar(-height / 2, angle)
    l1 = Line(p1, p2).offset(width / 2)
    l2 = Line(p1, p2).offset(-width / 2).reverse()
    return Polygon([l1.p1, l1.p2, l2.p1, l2.p2])
