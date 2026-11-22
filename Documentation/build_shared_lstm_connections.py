"""Draw the standalone LSTM's connections and memory update, without model changes."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
im = Image.new('RGB', (2000, 1840), 'white')
d = ImageDraw.Draw(im)
ink, muted, wire, accent = '#172e3b', '#4b606b', '#cbd5da', '#167766'
font_path = 'C:/Windows/Fonts/arial.ttf'
bold_path = 'C:/Windows/Fonts/arialbd.ttf'


def text(x, y, value, size=25, bold=False, color=ink, anchor='lt'):
    d.text((x, y), value, font=ImageFont.truetype(bold_path if bold else font_path, size), fill=color, anchor=anchor)


def line(points, color=ink, width=3, arrow=False):
    d.line(points, fill=color, width=width, joint='curve')
    if arrow:
        x, y = points[-1]
        px, py = points[-2]
        import math
        angle = math.atan2(y-py, x-px)
        d.polygon([(x,y), (x-13*math.cos(angle-.45),y-13*math.sin(angle-.45)),
                   (x-13*math.cos(angle+.45),y-13*math.sin(angle+.45))], fill=color)


def box(x, y, w, h, labels, fill='#eff4f5', size=24):
    d.rounded_rectangle((x,y,x+w,y+h), radius=9, fill=fill, outline=wire, width=2)
    for n, label in enumerate(labels):
        text(x+w/2, y+h/2+(n-(len(labels)-1)/2)*30, label, size, anchor='mm')


def circle(x, y, label, r=22, fill='white'):
    d.ellipse((x-r,y-r,x+r,y+r), fill=fill, outline=accent, width=3)
    text(x,y,label,25,True,anchor='mm')


text(60,35,'Inside one shared LSTM layer',44,True)
text(60,95,'Default sizes: 4 new input values + 4 previous hidden values; 4 memory units. One observation per call.',27)
text(60,152,'1  Each incoming value connects to every weighted-sum node',30,True)
text(60,197,'Incoming values',24,True)
text(895,197,'16 weighted sums',24,True)
text(1215,197,'Activation',24,True)
text(1530,197,'16 resulting values',24,True)

source_y = [300+i*65 for i in range(4)] + [625+i*65 for i in range(4)]
target_y = [265+g*174+j*37 for g in range(4) for j in range(4)]
for sy in source_y:
    for ty in target_y:
        line([(325,sy),(930,ty)],wire,2)
# Highlight one complete weighted sum without hiding the other connections.
for sy in source_y:
    line([(325,sy),(930,target_y[0])],accent,3)
for n, y in enumerate(source_y):
    label = f'New input {n+1}' if n<4 else f'Previous hidden {n-3}'
    text(275,y,label,26,anchor='rm')
    circle(303,y,str(n+1 if n<4 else n-3))

groups = [('Forget gate','sigmoid','Keep amount'),('Input gate','sigmoid','Write amount'),
          ('Candidate memory','tanh','New content'),('Output gate','sigmoid','Expose amount')]
for g,(name,activation,result) in enumerate(groups):
    y0=target_y[g*4]
    for j in range(4):
        y=target_y[g*4+j]
        circle(952,y,'+',16, '#e8f3ef' if g==0 and j==0 else 'white')
        line([(976,y),(1205,y)],arrow=True)
        box(1210,y-16,145,32,[activation],size=22)
        line([(1358,y),(1510,y)],arrow=True)
        text(1540,y,f'{result} {j+1}',24,anchor='lm')
    # Keep the gate-name label clear of the wires.
    d.rectangle((1000,y0+128,1470,y0+153),fill='white')
    text(1010,y0+128,f'{name}: four separate calculations',21,color=muted)

text(60,956,'READ THE CONNECTIONS',24,True)
text(60,995,'Every line has its own learned weight. Each + node adds eight weighted values and two learned biases.',26)
text(60,1034,'Green lines show all eight contributions to one sum. The other sums have their own weights: 128 weighted connections total.',26)
text(60,1073,'Sigmoid produces control amounts between 0 and 1. Tanh produces candidate values between -1 and 1.',26)

d.line((60,1134,1940,1134),fill=wire,width=2)
text(60,1165,'2  Update one memory unit — the same operations happen for each of the four units',30,True)

# Data moves left to right; the old cell path is kept separate from the dense inputs.
box(60,1240,270,55,['Previous cell value'])
box(60,1330,270,55,['Keep amount'])
line([(330,1268),(580,1268)],arrow=True)
line([(330,1358),(530,1358),(530,1295),(580,1295)],arrow=True)
box(585,1248,140,70,['Multiply'])
line([(725,1283),(1020,1283)],arrow=True)
text(785,1240,'Retained memory',23)

box(60,1430,270,55,['Write amount'])
box(60,1520,270,55,['New content'])
line([(330,1458),(580,1458)],arrow=True)
line([(330,1548),(530,1548),(530,1485),(580,1485)],arrow=True)
box(585,1438,140,70,['Multiply'])
line([(725,1473),(955,1473),(955,1306),(1020,1306)],arrow=True)
text(765,1500,'Selected new content',23)
box(1025,1253,100,70,['Add'])
line([(1125,1288),(1195,1288)],arrow=True)
box(1200,1240,240,95,['Updated cell value','Save for next call'],fill='#e8f3ef')
line([(1440,1288),(1490,1288)],arrow=True)
box(1495,1253,100,70,['tanh'])
line([(1595,1288),(1665,1288)],arrow=True)
box(1668,1253,140,70,['Multiply'])
box(1200,1438,240,70,['Expose amount'])
line([(1440,1473),(1615,1473),(1615,1360),(1738,1360),(1738,1326)],arrow=True)
line([(1808,1288),(1870,1288),(1870,1390)],arrow=True)
box(1650,1395,290,160,['Updated hidden value','Current output AND','saved for next call'],fill='#e8f3ef')

text(60,1620,'READ THE OPERATIONS',24,True)
text(60,1660,'1. Keep part of the old cell value.   2. Select part of the proposed new content.   3. Add them to obtain the updated cell value.',25)
text(60,1700,'4. Apply tanh to a copy of that value, then multiply by the expose amount to obtain the updated hidden value.',25)
text(60,1740,'The cell value itself is not bounded by tanh. These operations pair matching unit numbers; they do not mix units again.',25)
text(60,1790,'One layer, four memory units, eight saved state values per stream. No stored input window. All parameters are reused on the next call.',24,True)

im.save(ROOT / 'shared_lstm_connections.png')
