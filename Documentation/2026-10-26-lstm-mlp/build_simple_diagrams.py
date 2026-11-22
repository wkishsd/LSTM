from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

root = Path(__file__).parent
assets = root / 'assets'
assets.mkdir(exist_ok=True)
fontpath = 'C:/Windows/Fonts/arial.ttf'
boldpath = 'C:/Windows/Fonts/arialbd.ttf'

def build(lang):
    im = Image.new('RGB', (1500, 850), '#ffffff')
    d = ImageDraw.Draw(im)
    regular = ImageFont.truetype(fontpath, 28)
    bold = ImageFont.truetype(boldpath, 31)
    small = ImageFont.truetype(fontpath, 25)
    en = lang == 'EN'
    def box(rect, title, lines, fill):
        d.rounded_rectangle(rect, radius=18, fill=fill, outline='#b8cad5', width=2)
        cx = (rect[0]+rect[2])/2
        d.text((cx, rect[1]+20), title, font=bold, fill='#123447', anchor='mt')
        for i,line in enumerate(lines):
            d.text((cx,rect[1]+65+i*38),line,font=regular,fill='#233e4c',anchor='mt')
    def arrow(points):
        d.line(points, fill='#567e90', width=5, joint='curve')
        x,y = points[-1]
        d.polygon([(x,y),(x-10,y-18),(x+10,y-18)], fill='#567e90')
    box((25,20,475,195), 'Older history' if en else 'Более ранняя история',
        ['Through 12:04' if en else 'До свечи 12:04', 'LSTM → older_summary', '64 values' if en else '64 значения'], '#e9f1fb')
    box((525,20,975,195), 'Recent window' if en else 'Последнее окно',
        ['12:05–13:04', 'recent_window: 60 × 36', '2,160 individual values' if en else '2 160 отдельных значений'], '#e8f6ee')
    box((1025,20,1475,195), 'Current anchors' if en else 'Текущие якоря',
        ['Decision at 13:05' if en else 'Решение в 13:05', 'anchor_flags', '18 values: 0 or 1' if en else '18 значений: 0 или 1'], '#fff3df')
    arrow([(250,195),(250,235),(750,235),(750,280)])
    arrow([(750,195),(750,280)])
    arrow([(1250,195),(1250,235),(750,235),(750,280)])
    box((320,280,1180,440), 'Join the three blocks' if en else 'Соединить три блока',
        ['mlp_input', '2,160 + 64 + 18 = 2,242 values' if en else '2 160 + 64 + 18 = 2 242 значения'], '#eef3f6')
    arrow([(750,440),(750,485)])
    box((320,485,1180,645), 'MLP → profit_probability',
        ['Uses all three blocks together' if en else 'Использует все три блока вместе',
         'Probability of exceeding the net profit threshold' if en else 'Вероятность превысить порог чистой доходности'], '#e9f1fb')
    arrow([(750,645),(750,690)])
    box((320,690,1180,825), 'Entry decision' if en else 'Решение о входе',
        ['Apply probability_cutoff and the series availability rule' if en else 'Применить probability_cutoff и правило занятости серии'], '#f2f5f7')
    im.save(assets / f'pipeline_simple_{lang.lower()}.png')

for lang in ['EN', 'RU']:
    build(lang)
