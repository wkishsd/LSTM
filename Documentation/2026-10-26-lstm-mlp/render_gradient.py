"""Render one equation for Google Docs with its explicit sum over decision times."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
P=Path(__file__).resolve().parent
font=ImageFont.truetype('C:/Windows/Fonts/cambria.ttc',64)
small=ImageFont.truetype('C:/Windows/Fonts/cambria.ttc',38)
canvas=Image.new('RGB',(2000,260),'white');d=ImageDraw.Draw(canvas)
def extent(base,sub=''):
    return d.textlength(base,font=font)+(d.textlength(sub,font=small) if sub else 0)
def term(cx,y,base,sub=''):
    width=extent(base,sub);x=cx-width/2;d.text((x,y),base,font=font,fill='black')
    if sub:d.text((x+d.textlength(base,font=font),y+31),sub,font=small,fill='black')
def fraction(x,top,bot):
    width=max(extent(*top),extent(*bot))+20;cx=x+width/2
    term(cx,10,*top);d.line((x,94,x+width,94),fill='black',width=3);term(cx,104,*bot)
    return x+width
x=10
x=fraction(x,('∂ℒ',''),('∂θ','LSTM'))
d.text((x+18,54),'=',font=font,fill='black');x+=85
d.text((x,38),'Σ',font=font,fill='black');d.text((x+19,111),'t',font=small,fill='black');x+=85
x=fraction(x,('∂ℒ',''),('∂m','t'))+30
x=fraction(x,('∂m','t'),('∂θ','LSTM'))
canvas=canvas.crop((0,0,int(x)+12,198))
path=P/'assets/gradient_sum.png';canvas.save(path,dpi=(300,300))
size={'path':str(path),'width_pt':round(canvas.width*11.5/64,2),'height_pt':round(canvas.height*11.5/64,2)}
(P/'assets/gradient_size.json').write_text(json.dumps(size),encoding='utf8')
print(json.dumps(size))
