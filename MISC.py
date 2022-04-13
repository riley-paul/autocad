import os
import numpy as np

import math
import datetime as dt

def round_up(num,divisor):
  return num + divisor - (num%divisor)

def round_down(num,divisor):
  return num - (num%divisor)

def latest_file(path,date_format="%Y %m %d",prefix="",suffix="",silent=False) -> tuple:
  files = os.listdir(path)
  files = [i for i in files if not (i.startswith(".") or i.startswith("~"))]
  file_date = []
  for file in files:
    name,_ = os.path.splitext(file)
    try:
      if prefix and suffix: date_str = name[name.index(prefix) + len(prefix):name.index(suffix)]
      elif prefix: date_str = name[name.index(prefix) + len(prefix):]
      elif suffix: date_str = name[:name.index(suffix)]
      else: date_str = name
    except ValueError as e:
      if not silent: print(f"ERROR: {prefix} or {suffix} is not contained within {name}")
      continue
    
    try:
      date = dt.datetime.strptime(date_str,date_format)
    except ValueError as e:
      if not silent: print(f"ERROR: {date_str} does not match the format of {date_format}")
      continue
    
    file_date.append((file,date))

  latest = max(file_date,key=lambda i: i[1])
  fname = os.path.join(path,latest[0])
  date = latest[1]
  return fname,date

def file_date(fname,date_format="%Y %m %d",prefix="",suffix="") -> dt.datetime:
  name,_ = os.path.splitext(os.path.basename(fname))
  try:
    if prefix and suffix: date_str = name[name.index(prefix) + len(prefix):name.index(suffix)]
    elif prefix: date_str = name[name.index(prefix) + len(prefix):]
    elif suffix: date_str = name[:name.index(suffix)]
    else: date_str = name
  except ValueError as e:
    print(f"ERROR: {prefix} or {suffix} is not contained within {name}")
    return False
  
  try:
    date = dt.datetime.strptime(date_str,date_format)
  except ValueError as e:
    print(f"ERROR: {date_str} does not match the format of {date_format}")
    return False

  return date

def format_kml_table(data,**kwargs) -> str:
  fname = os.path.join("/Users/riley/Google Drive/CODE/DATA/KML_desc_style.html")
  with open(fname,"r") as file: style = file.read()
  
  html = data.to_html(header=False,na_rep="-",**kwargs)
  desc = style + html
  desc = desc.replace(" 00:00:00","")
  return desc

def percent_exaggeration(p,n=4):
  assert p <= 1 and p >= 0, "p must be between 0 and 1"
  for _ in range(n):
    p = math.log2(p+1)
  return p

def alpha_to_num(col_alpha):
  """Returns the number corresponding to an alphabetic index"""
  numbers = [ord(i) - 97 for i in col_alpha.lower()]
  return sum([i + index*26 for index,i in enumerate(numbers)])

# def colour_scale(c1,c2,num_col=None,percentile=None,exaggeration=3):
#   if num_col != None:
#     return np.linspace(c1,c2,num_col)
#   elif percentile != None:
#     percentile = percent_exaggeration(percentile,exaggeration)
#     return tuple([percentile*(b-a)+a for a,b in zip(c1,c2)])

def colour_scale(c1,c2,num_col=2):
  colours = np.linspace(c1,c2,num_col)
  colours = [[int(i) for i in j] for j in colours]
  return colours

def replace_bool(data):
  for column in data.columns:
    temp = data[column].drop_duplicates().dropna()
    if len(temp) <= 2 and all(i == 1 or i == 0 for i in temp):
      data[column] = data[column].replace(1,True).replace(0,False)

def colour_rainbow(length):
  frequency = 0.8
  p1,p2,p3 = 0,2,4
  width = 127
  center = 128

  red = [math.sin(frequency*i + p1)*width + center for i in range(length)]
  grn = [math.sin(frequency*i + p2)*width + center for i in range(length)]
  blu = [math.sin(frequency*i + p3)*width + center for i in range(length)]

  return [(r,g,b) for r,g,b in zip(red,grn,blu)]

def remove_adjacent(data):
  assert all(i in data.columns for i in ["KP_beg","KP_end"]), "KP_beg and KP_end columns must be in data"
  if len(data) <= 1: return data
  
  data = (data
    .sort_values("KP_beg")
    .reset_index(drop=True))

  index = 1
  length = len(data)

  old_rows = length

  while index < length:
    first = data.loc[index-1].drop(["KP_beg","KP_end"])
    second = data.loc[index].drop(["KP_beg","KP_end"])

    KP_beg = data.at[index,"KP_beg"]
    KP_end = data.at[index-1,"KP_end"]

    if first.equals(second) and KP_beg == KP_end:
      data.at[index-1,"KP_end"] = data.at[index,"KP_end"]
      data = data.drop(index).reset_index(drop=True)
      index -= 1
    length = len(data)
    index += 1
  print(f"Rows reduced from {old_rows} to {len(data)}")
  return data

def convert_chainage_string(string:str):
  if type(string) != str: return float(string)
  result = string.replace(" ","").split("+")
  try: result = [float(i) for i in result]
  except: return None
  result[0] *= 1000
  result = sum(result)
  return result

def write_prj_file(fname:str,EPSG:int=26910):
  import urllib
  
  fname = os.path.splitext(fname)[0] + ".prj"
  url = f"http://spatialreference.org/ref/epsg/{str(EPSG)}/prettywkt/"
  
  with urllib.request.urlopen(url) as request:
    wkt = (request.read().decode("UTF-8")
      .replace(" ","").replace("\n", "")
      .encode("UTF-8"))

  with open(fname,"wb") as prj:
    prj.write(wkt)