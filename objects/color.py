def color(color="bylayer"):
  if color == "bylayer": return ["_.-COLOR BYLAYER"]
  color = [str(int(i)) for i in color]
  return [f"_.-COLOR T {','.join(color)}"]
