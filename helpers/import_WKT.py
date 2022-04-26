
def import_WKT(text):
  from autocad.geometry import Point,Polyline,Polygon
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