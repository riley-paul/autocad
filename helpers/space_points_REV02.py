import pandas as pd

def space_points(points,spacing,iterations=5):
  """Function to intelligently space out a list of points 
  such that they are as close to their original positions 
  as possible while maintaining a certain buffer from one
  another
  
  These points are on a linear scale
  """

  def space(group,spacing):
    """Function to evenly space points around the
    centroid of the original group"""
    center = sum(group) / len(group) # determine center of group
    final_width = (len(group)-1) * spacing
    beg = center - final_width/2
    return [beg + i * spacing for i in range(len(group))]

  def clump(points,spacing):
    """Function to group points together based on 
    their proximity to one another"""
    points.sort()
    groups = [[pt] for pt in points]
    clumped_groups = []
    cand = groups.pop(0)
    while groups:
      spaced_cand = space(cand,spacing)
      if spaced_cand[-1] + spacing > groups[0][0]:
        cand += groups.pop(0)
      else:
        clumped_groups.append(cand)
        cand = groups.pop(0)
    clumped_groups.append(cand)
    return clumped_groups

  for _ in range(iterations):
    clumps = clump(points,spacing)
    result = []
    for item in clumps:
      result += space(item,spacing)
    points = result
  return result

def space_ranges(ranges,spacing=1,sort_by=0):
  """ sort_by = 0 for smallest
    sort_by = 1 for largest
    sort_by = 2 for center """ 
  ranges = [(min(i),max(i),sum(i)/len(i),max(i)-min(i)) for i in ranges]
  # (start,end,center,width)

  def space(group,spacing):
    """Function to evenly space points around the
    centroid of the original group"""
    center = sum([i[2] for i in group])/len(group)
    width = sum([i[3] for i in group]) + spacing * (len(group) - 1)
    group.sort(key=lambda i: i[sort_by])
    beg = center - width/2
    new_rngs = []
    for rng in group:
      new_rngs.append((beg,beg+rng[3],beg+rng[3]/2,rng[3]))
      beg += rng[3] + spacing
    return new_rngs

  def clump(ranges,spacing):
    """Function to group points together based on 
    their proximity to one another"""
    ranges.sort(key=lambda i: i[sort_by])
    groups = [[i] for i in ranges]
    clumped_groups = []
    cand = groups.pop(0)
    while groups:
      spaced_cand = space(cand,spacing)
      if spaced_cand[-1][1] + spacing > groups[0][0][0]:
        cand += groups.pop(0)
      else:
        clumped_groups.append(cand)
        cand = groups.pop(0)
    clumped_groups.append(cand)
    return clumped_groups

  for _ in range(50):
    clumps = clump(ranges,spacing)
    result = []
    for item in clumps:
      result += space(item,spacing)
    ranges = result

  return [(i[0],i[1]) for i in result]

def space_ranges(
  data: pd.DataFrame,
  spacing: int = 1,
  sort_by: str = "beg",
  col_beg = "beg",
  col_end = "end",
):
  # Store old beg and end
  data["beg_old"] = data[[col_beg,col_end]].min(axis=1)
  data["end_old"] = data[[col_beg,col_end]].max(axis=1)

  data["beg"] = data["beg_old"]
  data["end"] = data["end_old"]
  data["mid"] = data[["beg","end"]].mean(axis=1)
  data["len"] = data.apply(lambda r: r.end - r.beg,axis=1)
  data = data.sort_values(sort_by).reset_index(drop=True)

  def space_incrementally(group: pd.DataFrame):
    # Determine final destination and move a percentage of that
    
    group_center = group.mid.mean()
    group_width = group.len.sum() + (len(group) - 1) * spacing
    group_begin = group_center - group_width/2

    for index,entry in group.iterrows():
      group.at[index,"beg_pos"] = group_begin
      group.at[index,"end_pos"] = group_begin + entry.len
      group_begin += entry.len + spacing

    # group["diff"] = group["beg_pos"] - group["beg"]
    # group["beg"] = group.apply(lambda r: r.beg + r["diff"]/2 if r["diff"] > spacing/3 else r.beg_pos,axis=1)
    # group["end"] = group.apply(lambda r: r.end + r["diff"]/2 if r["diff"] > spacing/3 else r.end_pos,axis=1)

    group["beg"] = group["beg_pos"]
    group["end"] = group["end_pos"]

    return group

  def group(data):
    # Create 
    groups = pd.DataFrame()
    groups["i1"] = data.index[:-1]
    groups["i2"] = data.index[1:]
    groups["gap"] = groups.apply(lambda r: data.at[r.i2,"beg"] - data.at[r.i1,"end"],axis=1)
    groups["prob"] = groups.gap < spacing
    groups.drop("gap",axis=1,inplace=True)
    

    # Aggregate groups to start and stop indices
    index = 1
    length = len(groups)
    droppable = ["i1","i2"]
    while index < length:
      first = groups.loc[index-1].drop(droppable)
      second = groups.loc[index].drop(droppable)

      i1 = groups.at[index,"i1"]
      i2 = groups.at[index-1,"i2"]

      if first.equals(second) and i1 == i2:
        groups.at[index-1,"i2"] = groups.at[index,"i2"]
        groups = groups.drop(index).reset_index(drop=True)
        index -= 1
      length = len(groups)
      index += 1

    return groups.loc[groups.prob]

  print(data)

  groups = group(data)
  iters = 0
  while groups.prob.any() and iters < 50:
    print(groups)
    data_groups = [data]
    for _,g in groups.iterrows():
      temp = data.iloc[g.i1:g.i2+1].copy()
      temp = space_incrementally(temp)
      data_groups.append(temp)

    for data_group in data_groups: data.update(data_group)
    # print(data)
    groups = group(data)
    iters += 1

  return data