from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

root = Path(__file__).parent
assets = root / 'assets'
fontpath = 'C:/Windows/Fonts/arial.ttf'
boldpath = 'C:/Windows/Fonts/arialbd.ttf'

def build(lang):
    im = Image.new('RGB', (1500, 850), '#ffffff')
    d = ImageDraw.Draw(im)
    regular = ImageFont.truetype(fontpath, 28)
    bold = ImageFont.truetype(boldpath, 31)
    en = lang == 'EN'
    def box(rect, title, lines, fill):
        d.rounded_rectangle(rect, radius=18, fill=fill, outline='#b8cad5', width=2)
        cx = (rect[0]+rect[2])/2
        d.text((cx, rect[1]+18), title, font=bold, fill='#123447', anchor='mt')
        for i,line in enumerate(lines):
            d.text((cx,rect[1]+63+i*36),line,font=regular,fill='#233e4c',anchor='mt')
    def arrow(points):
        d.line(points, fill='#567e90', width=5, joint='curve')
        x,y = points[-1]
        d.polygon([(x,y),(x-10,y-18),(x+10,y-18)], fill='#567e90')
    box((25,20,725,235), 'Older history → LSTM' if en else 'Ранняя история → LSTM',
        ['Rows through 12:05; no fixed lookback' if en else 'Строки до 12:05; без фиксированного окна',
         'Carry 64 hidden + 64 cell values' if en else 'Сохраняются 64 hidden + 64 cell значения',
         'Send hidden state only:' if en else 'В MLP передаётся только hidden state:',
         'older_summary: 64 values' if en else 'older_summary: 64 значения'], '#e9f1fb')
    box((775,20,1475,235), 'Recent window → flatten' if en else 'Последнее окно → разворачивание',
        ['Decision rows 12:06–13:05' if en else 'Строки решений 12:06–13:05',
         'recent_window: 60 × 54',
         '36 market values + 18 flags in every row' if en else 'В каждой строке 36 признаков + 18 флагов',
         '3,240 individual values' if en else '3 240 отдельных значений'], '#e8f6ee')
    arrow([(375,235),(375,255),(750,255),(750,285)])
    arrow([(1125,235),(1125,255),(750,255),(750,285)])
    box((320,285,1180,440), 'Join the two blocks' if en else 'Соединить два блока',
        ['mlp_input', '3,240 + 64 = 3,304 values' if en else '3 240 + 64 = 3 304 значения'], '#eef3f6')
    arrow([(750,440),(750,485)])
    box((320,485,1180,645), 'MLP → profit_probability',
        ['Learns relationships across lags and older context' if en else 'Учит связи между лагами и ранним контекстом',
         'Probability of exceeding the net profit threshold' if en else 'Вероятность превысить порог чистой доходности'], '#e9f1fb')
    arrow([(750,645),(750,690)])
    box((320,690,1180,825), 'Entry decision' if en else 'Решение о входе',
        ['Apply probability_cutoff and the series availability rule' if en else 'Применить probability_cutoff и правило занятости серии'], '#f2f5f7')
    im.save(assets / f'pipeline_current_{lang.lower()}.png')

for lang in ['EN', 'RU']:
    build(lang)
