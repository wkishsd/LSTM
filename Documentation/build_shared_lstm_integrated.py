"""One connected, left-to-right LSTM illustration (two inputs, two units)."""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
S = 2
im = Image.new('RGB', (2200*S, 1330*S), 'white')
d = ImageDraw.Draw(im)
INK = '#193443'
SOFT = '#c3ced4'
GREEN = '#167866'
BLUE = '#286c9b'


def text(x,y,s,size=23,bold=False,color=INK,anchor='lt'):
    font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size*S)
    d.text((x*S,y*S),s,font=font,fill=color,anchor=anchor)


def path(points,color=INK,width=2,arrow=False):
    d.line([(x*S,y*S) for x,y in points], fill=color,width=width*S,joint='curve')
    if arrow:
        x,y=points[-1]; px,py=points[-2]
        a=math.atan2(y-py,x-px)
        d.polygon([(x*S,y*S),((x-10*math.cos(a-.45))*S,(y-10*math.sin(a-.45))*S),
                   ((x-10*math.cos(a+.45))*S,(y-10*math.sin(a+.45))*S)],fill=color)


def box(x,y,w,h,lines,fill='#f0f5f6',size=23):
    d.rounded_rectangle((x*S,y*S,(x+w)*S,(y+h)*S),radius=8*S,fill=fill)
    for i,s in enumerate(lines):
        text(x+w/2,y+h/2+(i-(len(lines)-1)/2)*28,s,size,anchor='mm')


def node(x,y,label,color=INK,r=23):
    d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill='white',outline=color,width=2*S)
    text(x,y,label,25,True,color,anchor='mm')


text(55,30,'One observation through one shared LSTM',40,True)
text(55,88,'Illustration: 2 new inputs, 2 memory units. Each stream carries 2 hidden values and 2 cell values.',26)
text(55,135,'INPUTS + PREVIOUS HIDDEN VALUES',22,True)
text(585,135,'LEARNED SUMS',22,True)
text(830,135,'GATES',22,True)
text(1180,135,'UPDATE THE CELL MEMORY',22,True)
text(1810,135,'OUTPUT + NEXT STATE',22,True)

# All four source nodes connect to all eight learned sums.
sources=[(310,470,'New input 1'),(310,555,'New input 2'),
         (310,720,'Previous hidden 1'),(310,805,'Previous hidden 2')]
ys=[285,385,485,585,765,865,965,1065]
for _,sy,_ in sources:
    for ty in ys:
        path([(331,sy),(650,ty)],SOFT,1)
for _,sy,_ in sources:
    path([(331,sy),(650,285)],GREEN,2)
for x,y,label in sources:
    text(x-38,y,label,24,anchor='rm')
    node(x,y,'',BLUE)
text(55,875,'All four values feed every sum.',23)
text(55,910,'Each line has its own weight.',23)
text(55,945,'Green: one complete weighted sum.',23,color=GREEN)

# Two lanes are memory coordinates in the SAME layer, not stacked layers.
for unit,base,color in [(1,0,GREEN),(2,480,BLUE)]:
    old=220+base
    forget,write,candidate,expose=[y+base for y in (285,385,485,585)]
    text(1165,175+base,f'MEMORY UNIT {unit}',22,True,color)
    # Gates include separate sum and activation nodes, with no symbolic names.
    for y,name,activation in [(forget,'Keep','sigmoid'),(write,'Write','sigmoid'),
                              (candidate,'New content','tanh'),(expose,'Expose','sigmoid')]:
        node(675,y,'+',color)
        path([(699,y),(810,y)],arrow=True)
        box(815,y-29,190,58,[f'{name} {unit}',activation],size=22)
    # The cell highway and the weighted incoming gates are one connected graph.
    text(1015,old-33,f'Previous cell {unit}',22)
    path([(1015,old),(1201,old)],color,3,True)
    node(1225,old,'×',color)
    path([(1005,forget),(1225,forget),(1225,old+25)],color,2,True)
    path([(1250,old),(1476,old)],color,3,True)
    node(1500,old,'+',color)

    # Write amount times candidate content, then add into the cell.
    product=435+base
    path([(1005,write),(1155,write),(1155,product),(1200,product)],color,2,True)
    path([(1005,candidate),(1225,candidate),(1225,product+25)],color,2,True)
    node(1225,product,'×',color)
    path([(1250,product),(1500,product),(1500,old+25)],color,2,True)
    text(1257,product+13,'New content to store',22)

    # Updated cell continues unchanged to the saved state; a branch is bounded.
    path([(1525,old),(1955,old)],color,3,True)
    box(1960,old-32,195,64,[f'Updated cell {unit}','Save for next call'],fill='#e9f2f3',size=21)
    node_y=350+base
    path([(1630,old),(1630,node_y-32)],color,2,True)
    d.ellipse(((1630-4)*S,(old-4)*S,(1630+4)*S,(old+4)*S),fill=color)
    box(1565,node_y-29,130,58,['tanh'])
    path([(1695,node_y),(1795,node_y)],color,2,True)
    node(1820,node_y,'×',color)
    path([(1005,expose),(1820,expose),(1820,node_y+25)],color,2,True)
    path([(1845,node_y),(1955,node_y)],color,3,True)
    box(1960,node_y-45,195,90,[f'Updated hidden {unit}','Output AND','save for next call'],fill='#e9f2f3',size=21)

path([(55,1140),(2155,1140)],SOFT,1)
text(55,1170,'1  Mix the inputs',24,True)
text(55,1210,'Each learned sum adds four weighted values',22)
text(55,1243,'and two learned biases.',22)
text(590,1170,'2  Form the gate values',24,True)
text(590,1210,'Sigmoid: control amount from 0 to 1.',22)
text(590,1243,'Tanh: proposed content from −1 to 1.',22)
text(1130,1170,'3  Update each cell',24,True)
text(1130,1210,'Keep × old cell + write × new content.',22)
text(1130,1243,'The two units keep separate cell values.',22)
text(1690,1170,'4  Expose the output',24,True)
text(1690,1210,'Expose × tanh(updated cell).',22)
text(1690,1243,'Carry both cell and hidden values onward.',22)
text(55,1290,'One new observation per call. No lookback window. Diagram sizes are illustrative; Python defaults remain 4 inputs and 4 memory units.',22,True)
im.save(ROOT/'shared_lstm_integrated.png')
