# Pythonic AutoCAD Script Generation

Package for recreating AutoCAD objects in Python and exporting them as command line arguments for the generation of drawings.

At the time of making this package there was no easy way (that I could find) to interact with AutoCAD using Python. AutoCAD comes with many data consumption features but these slow the program past it's already glacial pace. I wanted a solution that would generate lightweight, 2D elements anywhere and in any style and allow for some manipulation of layouts and layers.

To create a red layer called _POINTS_ and add a point to it is as simple as:

```python
from autocad.geometry import Point
from autocad.objects import Layer
from autocad.helpers import  print_to_file

# Instantiate layer and point objects
lyr = Layer(name="POINTS",colour=1)
pt = Point(3,4)

command = [] # Instantiate empty command array
command += lyr.ACAD() # Add commands to create and select new layer
command += pt.ACAD() # Add commands to generate point

# Helper function to print list to file
print_to_file("test.scr",command)
```

_Further documentation coming soon_
