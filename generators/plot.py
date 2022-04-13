from old.ACAD import *
from random import randint
from helpers.space_points import space_points
from dataclasses import dataclass
from typing import List

def round_up(num,divisor): return num + divisor - (num%divisor)
def round_down(num,divisor): return num - (num%divisor)

def dec_to_dms(angle:float):
    degree = int(angle)
    minute = int((angle - degree)*60)
    second = int((angle - degree - minute/60) * 3600)
    return f"{degree}°{minute}'{second}\""

@dataclass
class Plot:
    datasets: List[List[Point]]
    origin: Point = Point(0,0)
    colours: List[any] = None
    scale: float = 1
    is_KP: bool = False
    x_ext: int = None
    y_ext: int = None
    x_lab: str = "SSEID 05.24.2 CL"
    y_lab: str = "ELEVATION"
    p_lab: str = "TITLE"
    just: str = "MC" # [T,M,B] [L,C,R]
    PS_y_div: int = 5
    PS_x_div: int = 50

    def __post_init__(self):
        PS_text_height = 1.8

        PS_over = 1

        self.text_height = PS_text_height/self.scale
        self.y_div = round(self.PS_y_div/self.scale,0)
        self.x_div = round(self.PS_x_div/self.scale,0)
        self.over = PS_over/self.scale
        
        self.ranges = []
        self.range_height = self.text_height + self.over

        self.datasets = [sorted(i,key=lambda j: j.x) for i in self.datasets]
        self.polylines = [Polyline(i) for i in self.datasets]
        datasets_flattened = [item for sublist in self.datasets for item in sublist]
        pt_avg = avg_points(datasets_flattened)

        if self.x_ext:
            if self.just[-1] == "L":
                self.x_min = round_down(min(i.x for i in datasets_flattened),self.x_div)
                self.x_max = self.x_min + self.x_ext

            elif self.just[-1] == "C":
                self.x_min = round(pt_avg.x - self.x_ext/2,-1)
                self.x_max = round(pt_avg.x + self.x_ext/2,-1)

            elif self.just[-1] == "R":
                self.x_max = round_up(max(i.x for i in datasets_flattened),self.x_div)
                self.x_min = self.x_max - self.x_ext

        else:
            self.x_min = round_down(min(i.x for i in datasets_flattened),self.x_div)
            self.x_max = round_up(max(i.x for i in datasets_flattened),self.x_div)
            self.x_ext = self.x_max - self.x_min

        if self.y_ext:
            if self.just[0] == "T":
                self.y_max = round_up(max(i.y for i in datasets_flattened),self.y_div)
                self.y_min = self.y_max - self.y_ext
                
            elif self.just[0] == "M":
                self.y_min = round(pt_avg.y - self.y_ext/2,-1)
                self.y_max = round(pt_avg.y + self.y_ext/2,-1)

            if self.just[0] == "B":
                self.y_min = round_down(min(i.y for i in datasets_flattened),self.y_div)
                self.y_max = self.y_min + self.y_ext

        else:
            self.y_min = round_down(min(i.y for i in datasets_flattened),self.y_div)
            self.y_max = round_up(max(i.y for i in datasets_flattened),self.y_div)
            self.y_ext = self.y_max - self.y_min

    def transform(self,x=None,y=None) -> float or Point:
        if x == None and y != None: return self.origin.y - self.y_min + y
        if x != None and y == None: return self.origin.x - self.x_min + x
        if x != None and y != None: return Point(x,y).copy(self.origin.x - self.x_min,self.origin.y - self.y_min)

    def draw_grid(self):
        print(f"x_ext = {self.x_max} - {self.x_min} = {self.x_ext}")
        print(f"y_ext = {self.y_max} - {self.y_min} = {self.y_ext}")
        
        command = []

        lay_axes1 = Layer("PLOT - Axes 1")
        lay_axes2 = Layer("PLOT - Axes 2",colour=253,ltype="ACAD_ISO02W100")
        lay_label = Layer("PLOT - Labels")
        
        # Create vertical lines
        i = self.origin.x
        while i <= self.x_ext + self.origin.x:
            p1 = Point(i,self.origin.y + self.y_ext)
            p2 = Point(i,self.origin.y)
            p3 = p2.copy(0,-len(self.ranges)*self.range_height)
            p4 = p3.copy(0,-self.over)
            pt_txt = p4.copy(0,-self.over)
            
            value = self.x_min + i - self.origin.x

            if self.is_KP: txt_label = format_KP(value,False)
            else: txt_label = f"{value:.0f}"

            if i == self.origin.x or i == self.origin.x + self.x_ext:
                command += lay_axes1.ACAD()
                command += Line(p1,p4).ACAD()

            else:
                command += lay_axes2.ACAD()
                command += Line(p1,p2).ACAD()
                command += lay_axes1.ACAD()
                command += Line(p3,p4).ACAD()

            command += lay_label.ACAD()
            command += Text(txt_label,pt_txt,height=self.text_height,just="TC").ACAD()

            i += self.x_div

        # Create horizontal lines
        i = self.origin.y
        while i <= self.y_ext + self.origin.y:
            pt_min = Point(self.origin.x,i)
            pt_max = Point(self.origin.x + self.x_ext,i)
            pt_mid1 = pt_min.copy(self.over*4,0)
            pt_mid2 = pt_max.copy(-self.over*4,0)

            
            txt = f"{self.y_min + i - self.origin.y:,.0f}"

            if i == self.origin.y:
                command += lay_axes1.ACAD()
                command += Line(pt_min,pt_max).ACAD()

            else:
                command += lay_axes1.ACAD()
                command += Line(pt_min,pt_mid1).ACAD()
                command += Line(pt_max,pt_mid2).ACAD()
                command += lay_axes2.ACAD()
                command += Line(pt_mid1,pt_mid2).ACAD()
                
            command += lay_label.ACAD()
            command += Text(txt,position=pt_min.copy(self.over,self.over),height=self.text_height,just="BL").ACAD()
            command += Text(txt,position=pt_max.copy(-self.over,self.over),height=self.text_height,just="BR").ACAD()

            i += self.y_div

        for index,rng in enumerate(self.ranges):
            command += self.draw_range(rng,self.origin.y - index*self.range_height)

        # Create axis labels
        if self.x_lab:
            pos = self.origin.copy(self.x_ext/2,-(self.over*2+self.text_height+self.over)-self.range_height*len(self.ranges))
            command += lay_label.ACAD()
            command += Text(self.x_lab,position=pos,height=self.text_height/1.2,just = "TC").ACAD()

        if self.y_lab:
            command += lay_label.ACAD()
            command += Text(self.y_lab,position=self.origin.copy(-self.over,self.y_ext/2),rot=90,height=self.text_height/1.2,just = "BC").ACAD()
            command += Text(self.y_lab,position=self.origin.copy(self.x_ext+self.over,self.y_ext/2),rot=90,height=self.text_height/1.2,just = "TC").ACAD()

        if self.p_lab:
            pos = self.origin.copy(self.x_ext/2,-(self.over*3+self.text_height+self.text_height/1.2+self.over*2)-self.range_height*len(self.ranges))
            command += lay_label.ACAD()
            command += Text(self.p_lab,position=pos,height=self.text_height*1.5,just = "TC").ACAD()

        command += Layer("0").ACAD()
        # command.append(f"LTself.SCALE {.25/self.self.scale}")
        return command

    def draw_lines(self):
        if self.colours: assert len(self.datasets) == len(self.colours), "Need a colour for every dataset"

        command = []
        inset = self.over*5

        # Crop data to fit within plot
        p1_min = self.origin.copy(inset,0)
        p2_min = p1_min.copy(0,self.y_ext)
        p1_max = self.origin.copy(self.x_ext-inset,0)
        p2_max = p1_max.copy(0,self.y_ext)

        min_ln = Line(p1_min,p2_min)
        max_ln = Line(p1_max,p2_max)

        for index,data in enumerate(self.datasets):
            pl = Polyline([self.transform(i.x,i.y) for i in data])
            p1 = min_ln.intersects_pl(pl)
            p1 = p1[0] if p1 else pl.vertices[0]
            p2 = max_ln.intersects_pl(pl)
            p2 = p2[0] if p2 else pl.vertices[-1]
            pl = pl.splice(p1,p2)

            # Draw polylines
            colour = self.colours[index] if self.colours else f"{randint(25,225)},{randint(25,225)},{randint(25,225)}"
            command += Layer(f"PLOT - Data {index + 1}",colour=colour,lweight=0.35).ACAD()
            command += pl.ACAD()
    
        command += Layer("0").ACAD()
        return command

    def mirror(self):
        command = []
        buffer = 50

        sel_p1 = self.origin.copy(-buffer,-buffer)
        sel_p2 = self.origin.copy(self.x_ext + buffer,self.y_ext + buffer*3)
        mir_p1 = self.origin.copy(self.x_ext/2,0)
        mir_p2 = self.origin.copy(self.x_ext/2,self.y_ext)
        
        command.append("MIRROR " + sel_p1.comm() + " " + sel_p2.comm() + "\n")
        command.append(mir_p1.comm() + " " + mir_p2.comm())
        command.append("Y")
        return command
    
    def on_line(self,x,data_index=0):
        if x > self.x_max or x < self.x_min:
            print(x,"out of range")
            return None
        p1 = Point(x,self.y_min)
        p2 = Point(x,self.y_max)
        line = Line(p1,p2)
        intersections = line.intersects_pl(self.polylines[data_index])
        if intersections: return self.transform(intersections[0].x,intersections[0].y)
        else: return None

    def add_range(self,item):
        self.ranges.append(item)

    def draw_range(self,ranges,top):
        # Accepts ranges as a tuple in the form (KP_beg,KP_end,text,layer,colour)
        
        bot = top - self.range_height
        lay_rng = Layer("PLOT - Ranges")
        command = []
        command += lay_rng.ACAD()
        command += Line(Point(self.transform(self.x_max),bot),Point(self.transform(self.x_min),bot)).ACAD()

        for entry in ranges:
            KP_beg = min(entry[:2])
            KP_end = max(entry[:2])

            if KP_beg >= self.x_max or KP_end <= self.x_min: continue

            if KP_beg >= self.x_min and KP_beg <= self.x_max:
                # Start of range within plot
                h1 = self.transform(KP_beg)
                command += lay_rng.ACAD()
                command += Line(Point(h1,top),Point(h1,bot)).ACAD()
            
            else:
                h1 = self.transform(self.x_min)

            if KP_end >= self.x_min and KP_end <= self.x_max:
                # End of range within plot
                h2 = self.transform(KP_end)
                command += lay_rng.ACAD()
                command += Line(Point(h2,top),Point(h2,bot)).ACAD()
            
            else:
                h2 = self.transform(self.x_max)
            
            lay_col = Layer(f"PLOT - Ranges - {entry[3]}",entry[4])
            center = Point((h1+h2)/2,(bot+top)/2)
            command += Layer("PLOT - Ranges - Text","255,255,255").ACAD()
            command += Text(entry[2],center,self.text_height/1.2,just="MC").ACAD()
            command += lay_col.ACAD()
            command += Polygon([Point(h1,bot),Point(h1,top),Point(h2,top),Point(h2,bot)]).ACAD_solid()
            command += lay_col.to_back()
        return command

    def draw_labels(self,labels,data_index=0,inc_x=False):
        # Accepts labels as a tuple in the form (x,text)
        labels = sorted(labels,key=lambda i: i[0])
        
        command = []
        command += Layer("PLOT - DIM",252).ACAD()
        ldr_pts = space_points([i[0] for i in labels],self.text_height*2)
        for label,ldr_pt in zip(labels,ldr_pts):
            text = f"{format_KP(label[0])} {label[1]}" if inc_x else label[1]

            p1 = self.on_line(label[0],data_index)
            if not p1: continue
            p2 = self.transform(label[0],self.y_max-self.y_div)
            p3 = self.transform(ldr_pt,self.y_max)
            p4 = p3.copy(0,self.over)
            command += Polyline([p1,p2,p3,p4]).ACAD()
            command += Text(text,p4.copy(0,self.over),height=self.text_height/1.2,rot=90,just="ML").ACAD()
        command += Layer("0").ACAD()
        return command

    def ACAD(self):
        command = []
        command += self.draw_grid()
        command += self.draw_lines()
        command += Layer("0").ACAD()
        return command