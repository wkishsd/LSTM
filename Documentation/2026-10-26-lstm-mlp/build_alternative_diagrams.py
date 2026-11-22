from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math

root = Path(__file__).parent
assets = root / 'assets'
BLUE = '#476e88'
ORANGE = '#b77a25'
INK = '#17394b'

def build(lang, variant):
    en = lang == 'EN'
    im = Image.new('RGB', (1600, 1430), 'white')
    d = ImageDraw.Draw(im)
    fontfile = 'C:/Windows/Fonts/arial.ttf'
    boldfile = 'C:/Windows/Fonts/arialbd.ttf'
    def font(size=34, bold=False):
        return ImageFont.truetype(boldfile if bold else fontfile, size)
    def line(x, y, text, size=34, bold=False, color=INK):
        d.text((x,y), text, font=font(size,bold), fill=color, anchor='mt')
    def box(rect,title,lines,fill='#eaf2fa',size=34):
        d.rounded_rectangle(rect,radius=16,fill=fill,outline='#bdcdd7',width=2)
        cx=(rect[0]+rect[2])/2
        line(cx,rect[1]+16,title,size,True)
        for i,text in enumerate(lines):
            line(cx,rect[1]+65+i*40,text,size-2)
    def arrow(points,color=BLUE,width=5):
        d.line(points,fill=color,width=width,joint='curve')
        x,y=points[-1];px,py=points[-2]
        a=math.atan2(y-py,x-px)
        pts=[(x,y),(x-18*math.cos(a)+9*math.sin(a),y-18*math.sin(a)-9*math.cos(a)),(x-18*math.cos(a)-9*math.sin(a),y-18*math.sin(a)+9*math.cos(a))]
        d.polygon(pts,fill=color)
    d.rounded_rectangle((40,10,1560,65),radius=14,fill='#fff0cf')
    banner = ('UNDER DISCUSSION — B starts with 1 unit per feature; expandable' if en else 'ПОД ВОПРОСОМ — Б: 1 ячейка на признак; можно расширить') if variant == 'B' else ('UNDER DISCUSSION — illustrative sizes; neither option selected' if en else 'ПОД ВОПРОСОМ — размеры для примера; вариант ещё не выбран')
    line(800,20,banner,32,True,color='#825515')
    box((40,95,1560,220),'Continuous minute timeline' if en else 'Непрерывный ряд по минутам',
        ['54 features per row: 36 market values + 18 anchor event flags' if en else 'В строке 54 признака: 36 рыночных значений + 18 флагов событий якорей'])
    arrow([(800,220),(800,245),(530,245),(530,280)])
    arrow([(800,220),(800,245),(1335,245),(1335,280)])
    box((40,280,1020,420),'Older branch: rows through 12:05' if en else 'Ранняя ветка: строки до 12:05 включительно',
        ['Process every minute; carry state from initialization' if en else 'Каждая минута; состояния сохраняются с инициализации',
         'No fixed history cutoff; memory can still forget' if en else 'Нет фиксированной давности; память может забывать'],size=32)
    box((1100,280,1570,900),'Recent branch' if en else 'Недавняя ветка',
        ['Decision rows' if en else 'Строки решений',
         '12:06–13:05',
         'recent_window',
         '60 × 54',
         '↓',
         'Flatten oldest → newest' if en else 'Развернуть по времени',
         '↓',
         '3,240 values' if en else '3 240 значений',
         'All selected values kept' if en else 'Все выбранные значения',
         'at fixed lag positions' if en else 'на позициях своих лагов'],fill='#e8f5ed',size=31)
    if variant == 'A':
        box((40,480,400,680),'Previous state' if en else 'Прошлое состояние',
            ['hidden_state: 64','cell_state: 64'],fill='#fff3df',size=31)
        box((470,480,1020,680),'One shared LSTM' if en else 'Одна общая LSTM',
            ['54 inputs, distinct input weights' if en else '54 входа, разные входные веса',
             'Memory combines features' if en else 'Память объединяет признаки'],size=30)
        arrow([(745,420),(745,480)])
        arrow([(400,580),(470,580)])
        box((470,750,1020,900),'Updated state' if en else 'Новое состояние',
            ['hidden_state: 64','cell_state: 64'],fill='#fff3df',size=31)
        arrow([(745,680),(745,750)])
        arrow([(470,825),(220,825),(220,680)],color=ORANGE)
        line(245,858,'Next minute' if en else 'Следующая минута',29,color=ORANGE)
        arrow([(745,900),(745,1020),(800,1020),(800,1050)])
        line(500,945,'Only hidden: 64' if en else 'Только hidden: 64',32,True)
        total='3,240 + 64 = 3,304 inputs' if en else '3 240 + 64 = 3 304 входа'
    else:
        line(530,438,'54 independent feature branches; 3 examples shown' if en else '54 независимые ветки признаков; показаны 3 примера',29,True)
        examples=[('RSI','RSI'),('Volume','Объём'),('Anchor flag','Флаг якоря')]
        outputs=[]
        for i,(a,b) in enumerate(examples):
            left=40+i*335;right=left+310;cx=(left+right)/2
            box((left+18,535,right,710),a if en else b,
                ['Own LSTM + weights' if en else 'Своя LSTM и веса',
                 'New value + prior h, c' if en else 'Новое значение + h, c'],size=27)
            arrow([(cx,480),(cx,535)])
            if i==0:d.line([(80,480),(865,480)],fill=BLUE,width=5)
            box((left+18,760,right,900),'Updated h, c' if en else 'Новые h, c',
                ['1 hidden + 1 cell', 'Carry both forward' if en else 'Оба сохраняются'],fill='#fff3df',size=26)
            arrow([(cx,710),(cx,760)])
            arrow([(left+18,835),(left+4,835),(left+4,625),(left+18,625)],color=ORANGE,width=4)
            arrow([(cx,900),(cx,920)])
            outputs.append(cx)
        arrow([(80,420),(80,480)])
        box((40,920,1020,995),'Collect hidden only: 54 × 1 = 54' if en else 'Собрать только hidden: 54 × 1 = 54',[],fill='#eef2f5',size=30)
        arrow([(530,995),(530,1020),(800,1020),(800,1050)])
        total='3,240 + 54 = 3,294 inputs' if en else '3 240 + 54 = 3 294 входа'
    arrow([(1335,900),(1335,1020),(800,1020),(800,1050)])
    box((210,1050,1390,1175),'Join inputs → final MLP' if en else 'Соединить входы → итоговая MLP',[total],fill='#eef2f5')
    arrow([(800,1175),(800,1220)])
    box((210,1220,1390,1395),'profit_probability → entry decision' if en else 'profit_probability → решение о входе',
        ['One series target trains all branches jointly' if en else 'Одна цель серии обучает все ветки совместно',
         'Prediction cutoff + series availability; exit rules stay fixed' if en else 'Порог прогноза + занятость серии; правила выхода фиксированы'],size=32)
    suffix = '_1unit' if variant == 'B' else ''
    im.save(assets / f'architecture_{variant.lower()}{suffix}_{lang.lower()}.png')

for lang in ['EN','RU']:
    for variant in ['A','B']:
        build(lang,variant)
