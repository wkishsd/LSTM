"""Author the paired architecture reference. Does not import or edit trading code."""
from pathlib import Path
from copy import deepcopy
import json, math, re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)

# Native editable Word math. Vectors are columns in the single-example equations.
def r(text):
    n=OxmlElement('m:r'); p=OxmlElement('m:rPr'); st=OxmlElement('m:sty'); st.set(qn('m:val'),'p'); p.append(st); n.append(p)
    t=OxmlElement('m:t'); t.text=str(text); n.append(t); return n
def seq(*nodes):
    result=[]
    for n in nodes:
        result.extend(n if isinstance(n,list) else [r(n) if isinstance(n,str) else n])
    return result
def script(base,index,kind='sub'):
    n=OxmlElement('m:sSub' if kind=='sub' else 'm:sSup')
    for name,value in [('e',base),('sub' if kind=='sub' else 'sup',index)]:
        child=OxmlElement('m:'+name)
        for item in seq(value):child.append(deepcopy(item))
        n.append(child)
    return n
def sub(b,i):return script(b,i)
def sup(b,i):return script(b,i,'sup')
def frac(a,b):
    n=OxmlElement('m:f')
    for tag,x in [('num',a),('den',b)]:
        e=OxmlElement('m:'+tag)
        for y in seq(x):e.append(deepcopy(y))
        n.append(e)
    return n
def mat(rows):
    d=OxmlElement('m:d'); pr=OxmlElement('m:dPr')
    for tag,v in [('begChr','['),('endChr',']')]:
        z=OxmlElement('m:'+tag);z.set(qn('m:val'),v);pr.append(z)
    d.append(pr);e=OxmlElement('m:e');m=OxmlElement('m:m')
    for row in rows:
        mr=OxmlElement('m:mr')
        for val in row:
            me=OxmlElement('m:e')
            for x in seq(val):me.append(deepcopy(x))
            mr.append(me)
        m.append(mr)
    e.append(m);d.append(e);return d

EQ={
 'split':seq('s = t − L,  ',sub('R','t'),' = ',sup(seq('[',sub('x','s+1'),' … ',sub('x','t'),']'),'T'),' ∈ ',sup('ℝ','L×F')),
 'state':seq(sub('h','j'),', ',sub('c','j'),' ∈ ',sup('ℝ','H'),',  ',sub('x','j'),' ∈ ',sup('ℝ','F')),
 'sigmoid':seq('σ(v) = ',frac('1',seq('1 + ',sup('e','−v'))),',    ReLU(v) = max(0,v)'),
 'gate_i':seq(sub('i','j'),' = σ(',sub('W','xi'),sub('x','j'),' + ',sub('W','hi'),sub('h','j−1'),' + ',sub('b','i'),')'),
 'gate_f':seq(sub('f','j'),' = σ(',sub('W','xf'),sub('x','j'),' + ',sub('W','hf'),sub('h','j−1'),' + ',sub('b','f'),')'),
 'gate_g':seq(sub('g','j'),' = tanh(',sub('W','xg'),sub('x','j'),' + ',sub('W','hg'),sub('h','j−1'),' + ',sub('b','g'),')'),
 'gate_o':seq(sub('o','j'),' = σ(',sub('W','xo'),sub('x','j'),' + ',sub('W','ho'),sub('h','j−1'),' + ',sub('b','o'),')'),
 'cell':seq(sub('c','j'),' = ',sub('f','j'),' ⊙ ',sub('c','j−1'),' + ',sub('i','j'),' ⊙ ',sub('g','j')),
 'hidden':seq(sub('h','j'),' = ',sub('o','j'),' ⊙ tanh(',sub('c','j'),')'),
 'gate_dims':seq(sub('W','x*'),' ∈ ',sup('ℝ','H×F'),',  ',sub('W','h*'),' ∈ ',sup('ℝ','H×H'),',  ',sub('b','*'),' ∈ ',sup('ℝ','H')),
 'summary':seq(sub('m','t'),' = ',sub('h','s'),' = ',sub('h','t−L'),' ∈ ',sup('ℝ','H')),
 'flat':seq(sub('r','t'),' = ',sub('vec','row'),'(',sub('R','t'),') ∈ ',sup('ℝ','LF')),
 'index':seq(sub('r','t, (j−1)F+q'),' = ',sub('x','s+j,q'),',   j = 1,…,L; q = 1,…,F'),
 'concat':seq(sub('z','t'),' = ',sup(seq('[',sup(sub('r','t'),'T'),'  ',sup(sub('m','t'),'T'),'  ',sup(sub('a','t'),'T'),']'),'T'),' ∈ ',sup('ℝ','D'),',   D = LF + H + A'),
 'numbers':seq('D = 60·36 + 64 + 18 = 2242'),
 'mlp1':seq(sub('u','1'),' = ReLU(',sub('W','1'),sub('z','t'),' + ',sub('b','1'),') ∈ ',sup('ℝ','128')),
 'mlp2':seq(sub('u','2'),' = ReLU(',sub('W','2'),sub('u','1'),' + ',sub('b','2'),') ∈ ',sup('ℝ','64')),
 'mlp3':seq(sub('ℓ','t'),' = ',sub('w','3'),sub('u','2'),' + ',sub('b','3'),',    ',sub('p','t'),' = σ(',sub('ℓ','t'),') ∈ (0,1)'),
 'blocks':seq(sub('W','1'),' = [',sub('W','R'),'  ',sub('W','M'),'  ',sub('W','A'),']'),
 'influence':seq(sub('q','1'),' = ',sub('W','R'),sub('r','t'),' + ',sub('W','M'),sub('m','t'),' + ',sub('W','A'),sub('a','t'),' + ',sub('b','1')),
 'batched':seq(sub('U','1'),' = ReLU(Z',sup(sub('W','1'),'T'),' + ',sub('b','1'),') ∈ ',sup('ℝ','B×128')),
 'direct':seq(frac(seq('∂',sub('q','1,k')),seq('∂',sub('r','t,n'))),' = ',sub('W','R,k,n')),
 'compressed':seq(frac(seq('∂',sub('q','1,k')),seq('∂',sub('x','j,q'))),' = ',sub('W','M,k,:'),' ',frac(seq('∂',sub('h','s')),seq('∂',sub('x','j,q'))),',   j ≤ s'),
 'move':seq('(',sub('h','s+1'),',',sub('c','s+1'),') = LSTMCell(',sub('x','s+1'),', (',sub('h','s'),',',sub('c','s'),'))'),
 'target':seq(sub('y','t'),' = 𝟙[',sub('R','net,t'),' > τ]'),
 'loss':seq('ℒ = −',frac('1',seq('Σ',sub('v','t'))),'Σ',sub('v','t'),'[',sub('y','t'),' log(',sub('p','t'),') + (1−',sub('y','t'),') log(1−',sub('p','t'),')]'),
 'grad':seq(frac('∂ℒ',seq('∂',sub('θ','LSTM'))),' = ',sub('Σ','t'),' ',frac('∂ℒ',seq('∂',sub('m','t'))),' ',frac(seq('∂',sub('m','t')),seq('∂',sub('θ','LSTM')))),
 'update':seq('θ ← θ − η ',sub('∇','θ'),'ℒ'),
 'extra':seq(sub('z','t'),' = [',sup(sub('r','t'),'T'),', ',sup(sub('m','t'),'T'),', ',sup(sub('a','t'),'T'),', ',sup(sub('e','t'),'T'),sup(']','T'),',   D = LF + H + A + E'),
}

def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)

def diagram(lang):
    im=Image.new('RGB',(1500,1040),'white'); d=ImageDraw.Draw(im)
    ink='#20252b';line='#69737c';fill='#eef2f5'
    def text_center(box,lines,bold=False):
        x0,y0,x1,y1=box; f=font(29,bold);dy=39; y=(y0+y1-len(lines)*dy)/2
        for s in lines:
            d.text(((x0+x1-d.textlength(s,font=f))/2,y),s,font=f,fill=ink);y+=dy
    def box(b,lines,shade=False):
        d.rounded_rectangle(b,radius=9,fill=fill if shade else 'white',outline=line,width=2);text_center(b,lines)
    def arrow(a,b):
        d.line([a,b],fill=line,width=3)
        x,y=b;dx=b[0]-a[0];dy=b[1]-a[1];l=math.hypot(dx,dy);ux,uy=dx/l,dy/l
        d.polygon([(x,y),(x-14*ux+6*uy,y-14*uy-6*ux),(x-14*ux-6*uy,y-14*uy+6*ux)],fill=line)
    en=lang=='en'
    boxes=[(25,20,475,145),(525,20,975,145),(1025,20,1475,145)]
    texts=[['Earlier feature rows','through 12:04','[N, 36]'],['Recent feature rows','12:05 to 13:04','[60, 36]'],['Current anchor flags','known at 13:05','[18]']] if en else [['Предыдущие признаки','до 12:04 включительно','[N, 36]'],['Недавние признаки','с 12:05 по 13:04','[60, 36]'],['Текущие флаги якорей','известны в 13:05','[18]']]
    for b,t in zip(boxes,texts):box(b,t)
    for x in [250,750,1250]:arrow((x,145),(x,210))
    box((25,210,475,335),['LSTM','h and c: [64] each'] if en else ['LSTM','h и c по [64]'])
    box((525,210,975,335),['Flatten in time order','[2160]'] if en else ['Развернуть по времени','[2160]'])
    box((1025,210,1475,335),['Direct input','[18]'] if en else ['Прямой вход','[18]'])
    for x in [250,750,1250]:arrow((x,335),(x,415))
    box((25,415,475,525),['Summary h at 12:04','[64]'] if en else ['Сводка h в 12:04','[64]'])
    box((525,415,975,525),['Recent values preserved','[2160]'] if en else ['Значения сохранены','[2160]'])
    box((1025,415,1475,525),['Anchor values preserved','[18]'] if en else ['Флаги сохранены','[18]'])
    arrow((250,525),(600,610));arrow((750,525),(750,610));arrow((1250,525),(900,610))
    box((300,610,1200,710),['Concatenate: 2160 + 64 + 18 = 2242'] if en else ['Объединить 2160 + 64 + 18 = 2242'],True)
    arrow((750,710),(750,770))
    box((300,770,1200,875),['MLP: 2242 → 128 → 64 → 1'] if en else ['MLP: 2242 → 128 → 64 → 1'],True)
    arrow((750,875),(750,930))
    text_center((50,930,1450,1020),['Probability that net series return exceeds the threshold'] if en else ['Вероятность превышения порога чистой доходности серии'])
    path=ASSETS/f'forward_{lang}.png';im.save(path);return path

def init_doc(lang):
    doc=Document();sec=doc.sections[0];sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=Cm(1.65);sec.bottom_margin=Cm(1.65);sec.left_margin=Cm(1.8);sec.right_margin=Cm(1.8);sec.footer_distance=Cm(.7)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption','List Bullet']:
        st=doc.styles[name];st.font.name='Arial';st.font.color.rgb=RGBColor(0,0,0)
        st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Arial')
    normal=doc.styles['Normal'];normal.font.size=Pt(11);normal.paragraph_format.line_spacing=1.12;normal.paragraph_format.space_after=Pt(7)
    for name,size in [('Title',24),('Subtitle',11),('Heading 1',17),('Heading 2',12)]:
        st=doc.styles[name];st.font.size=Pt(size);st.paragraph_format.space_before=Pt(10);st.paragraph_format.space_after=Pt(7)
    for name in ['Heading 1','Heading 2']:doc.styles[name].paragraph_format.keep_with_next=True
    doc.styles['Caption'].font.size=Pt(9);doc.styles['Caption'].font.italic=False
    doc.core_properties.title='LSTM lookback window and MLP forward pass' if lang=='en' else 'Прямой проход через LSTM окно истории и MLP'
    doc.core_properties.subject='Architecture and mathematical reference'
    doc.core_properties.author='';doc.core_properties.keywords='LSTM MLP forward pass architecture'
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    return doc

class Writer:
    def __init__(self,lang):self.lang=lang;self.d=init_doc(lang);self.used=[]
    def pick(self,en,ru):return en if self.lang=='en' else ru
    def p(self,en,ru,bold=False):
        p=self.d.add_paragraph();p.add_run(self.pick(en,ru)).bold=bold;return p
    def h(self,en,ru,level=1):return self.d.add_heading(self.pick(en,ru),level)
    def bullet(self,en,ru):return self.d.add_paragraph(self.pick(en,ru),'List Bullet')
    def page(self,en,ru):
        self.d.add_page_break();self.h(en,ru)
    def eq(self,key):
        p=self.d.add_paragraph();p.paragraph_format.space_after=Pt(7)
        om=OxmlElement('m:oMath')
        for x in EQ[key]:om.append(deepcopy(x))
        p._p.append(om);self.used.append(key)
        return p
    def table(self,headers,rows,widths):
        t=self.d.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for col,width in zip(t.columns,widths):col.width=Cm(width)
        for c,h,w in zip(t.rows[0].cells,headers,widths):c.width=Cm(w);c.text=h
        for row in rows:
            cells=t.add_row().cells
            for c,txt,w in zip(cells,row,widths):c.width=Cm(w);c.text=txt
        for i,row in enumerate(t.rows):
            for c in row.cells:
                c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
                for edge in ['top','left','bottom','right']:
                    x=OxmlElement('w:'+edge);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');b.append(x)
                tcpr.append(b);m=OxmlElement('w:tcMar')
                for edge in ['top','left','bottom','right']:
                    x=OxmlElement('w:'+edge);x.set(qn('w:w'),'85');x.set(qn('w:type'),'dxa');m.append(x)
                tcpr.append(m)
                if i==0:
                    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8EEF3');tcpr.append(sh)
                for p in c.paragraphs:
                    p.paragraph_format.space_after=Pt(2);p.paragraph_format.line_spacing=1.05
                    for run in p.runs:run.font.size=Pt(10);run.bold=i==0
            trpr=row._tr.get_or_add_trPr();cs=OxmlElement('w:cantSplit');trpr.append(cs)
            if i==0:
                repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
        self.d.add_paragraph().paragraph_format.space_after=Pt(1)
        return t
    def code(self,en,ru):
        for line in self.pick(en,ru).splitlines():
            p=self.d.add_paragraph();p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1
            run=p.add_run(line);run.font.name='Consolas';run.font.size=Pt(9)

def build(lang):
    w=Writer(lang);P=w.p;H=w.h;E=w.eq;T=w.table;B=w.bullet
    w.d.add_paragraph(w.pick('LSTM lookback window and MLP forward pass','Прямой проход через LSTM окно истории и MLP'),'Title')
    w.d.add_paragraph(w.pick('Architecture reference with equations and tensor dimensions','Описание архитектуры с формулами и размерностями тензоров'),'Subtitle')
    P('The MLP receives individual values from the recent window and a compressed LSTM summary of the preceding history. A recent outlier remains a separate input if its feature is included. An older value has no separate input here; its influence can survive in the LSTM summary and in recent derived indicators.',
      'MLP получает отдельные значения признаков из недавнего окна и сжатую сводку LSTM по предшествующей истории. Недавний выброс остаётся отдельным входом, если признак включён. Старое значение не имеет отдельного входа; его влияние может сохраниться в сводке LSTM и в недавних производных индикаторах.',True)
    P('The following pages define the two branches, their timing, the LSTM recurrence, the exact join and every MLP matrix multiplication. The binary output describes the complete series under the fixed entry, stop, exit and re-entry rules.',
      'Далее описаны две ветви данных, их временные границы, рекуррентные вычисления LSTM, точка объединения и все матричные операции MLP. Бинарный выход относится ко всей серии при заданных правилах входа, стопа, выхода и повторного входа.')
    p=w.d.add_paragraph();p.add_run().add_picture(str(diagram(lang)),width=Cm(17.1))
    w.d.add_paragraph(w.pick('Figure 1  Values for one decision at 13:05 UTC+3. Layer sizes and a 60-minute window are illustrative.','Рисунок 1  Значения для одного решения в 13:05 UTC+3. Размеры слоёв и окно 60 минут приведены для примера.'),'Caption')
    P('This is a proposed network specification. The existing Python script supplies data and targets; this document does not establish that the network has been implemented or trained.',
      'Это описание предлагаемой сети. Существующий Python скрипт подготавливает данные и целевые значения; документ не означает, что такая сеть уже реализована или обучена.')

    w.page('1 Inputs and time boundaries','1 Входные данные и временные границы')
    P('Let d be the decision time, and t the index of the latest completed M1 candle available at d. The candle index refers to opening time; availability occurs at its close. At d = 13:05, t is the candle opening at 13:04. Set L = 60 and s = t − L. The older branch ends at 12:04; the recent window contains 12:05 through 13:04, inclusive.',
      'Пусть d — время решения, а t — индекс последней закрытой свечи M1, доступной в момент d. Индекс свечи относится ко времени открытия; доступность наступает при её закрытии. При d = 13:05 индекс t соответствует свече с открытием в 13:04. Зададим L = 60 и s = t − L. Старая история заканчивается на 12:04; недавнее окно содержит свечи с 12:05 по 13:04 включительно.')
    T(w.pick(['Symbol','Meaning','Example'],['Символ','Значение','Пример']),[
      ['F',w.pick('Market features per minute','Рыночных признаков на минуту'),'36'],
      ['L',w.pick('Detailed recent M1 candles','Недавних свечей M1 в деталях'),'60'],
      ['H',w.pick('LSTM hidden and cell width','Размер скрытого состояния и памяти LSTM'),'64'],
      ['A',w.pick('Current main-anchor flags','Текущих флагов основных якорей'),'18'],
      ['B',w.pick('Examples or aligned streams per batch','Примеров или согласованных потоков в пакете'),'32'],
      ['K',w.pick('Older recurrent steps in one processing chunk','Старых рекуррентных шагов в одном блоке'),'256'],
      ['N',w.pick('All older available steps since initialization','Всех старых шагов с момента инициализации'),w.pick('Variable','Переменно')],
    ],[2.0,10.7,4.5])
    P('The current saved model schema contains 36 normalized market columns: nine features for each of M1, M5, M15 and M30. The nine are log price, log candle volume, RSI, relative ATR, MACD/ATR, signal/ATR, ADX, DI+ and DI−. It also contains 18 current main-anchor flags. Entry IDs, timestamps, raw OHLCV and future execution fields are excluded from that numeric model schema.',
      'Текущая сохранённая схема входов содержит 36 нормализованных рыночных колонок: по девять признаков для M1, M5, M15 и M30. Это логарифм цены, логарифм объёма свечи, RSI, относительный ATR, MACD/ATR, signal/ATR, ADX, DI+ и DI−. Дополнительно есть 18 текущих флагов основных якорей. ID входа, время, сырые OHLCV и будущие параметры исполнения исключены из этой числовой схемы.')
    P('In this specification x_j is the 36-value market vector at minute j. Current anchor flags form a separate vector a_t. Historical anchor flags are not part of x_j in this example. Adding them requires explicitly changing F and defining their availability times.',
      'В этом описании x_j — вектор из 36 рыночных значений на минуте j. Текущие флаги якорей образуют отдельный вектор a_t. Исторические флаги якорей в данном примере не входят в x_j. Их добавление требует явного изменения F и определения времени доступности.')
    P('Every higher-timeframe feature must use only a candle closed by the feature availability time. Normalization and any fitted scalers must be causal. Real zero values must not be confused with padding; missing values need an explicit policy.',
      'Каждый признак старшего таймфрейма использует только свечу, закрытую ко времени доступности признаков. Нормализация и любые обучаемые масштабирования должны быть причинными. Реальные нули нельзя путать с дополнением последовательности; для пропусков нужна явная политика.')
    P('The branches use different feature-row timestamps. Their underlying information can overlap: an RSI or ATR in the recent window may already reflect candles before s.',
      'Ветви используют строки признаков с разными временами. Исходная информация может пересекаться: RSI или ATR недавнего окна уже может учитывать свечи до s.')

    w.page('2 What the LSTM calculates','2 Что вычисляет LSTM')
    P('Use one unidirectional LSTM layer with no projection. For each older minute j, it consumes x_j and the preceding states h_(j−1), c_(j−1). The hidden state h is the exposed summary; c is the internal cell memory. The following uses column vectors and one combined bias per gate. [1]',
      'Используется один однонаправленный слой LSTM без проекции. На каждой старой минуте j он получает x_j и предыдущие состояния h_(j−1), c_(j−1). Скрытое состояние h служит выходной сводкой; c — внутренней памятью ячейки. Ниже используются векторы столбцы и одно объединённое смещение на вентиль. [1]')
    E('state');E('sigmoid')
    P('The input gate i controls new information, forget gate f controls retained cell memory, and output gate o controls exposed memory. The candidate g supplies proposed new content. Each has H components. Multiplication marked ⊙ acts element by element.',
      'Входной вентиль i управляет новой информацией, вентиль забывания f — сохранением памяти, выходной вентиль o — её доступной на выходе долей. Кандидат g задаёт новое содержимое. Каждый вектор содержит H компонент. Знак ⊙ означает поэлементное умножение.')
    for key in ['gate_i','gate_f','gate_g','gate_o','cell','hidden']:E(key)
    E('gate_dims')
    P('Here * denotes i, f, g or o. With F = 36 and H = 64, each input matrix is [64,36], each recurrent matrix [64,64], and each bias [64]. Stacking four gates gives W_x [256,36], W_h [256,64], b [256]. These weights are shared across older minutes.',
      'Здесь * означает i, f, g или o. При F = 36 и H = 64 каждая входная матрица имеет размер [64,36], рекуррентная — [64,64], смещение — [64]. Четыре вентиля вместе дают W_x [256,36], W_h [256,64], b [256]. Одни и те же веса применяются на всех старых минутах.')
    E('summary')
    P('Only h_s enters the MLP in the baseline design. Both h_s and c_s are retained to advance the LSTM. The 64 numbers are learned coordinates, not 64 predefined indicators. Initialization defines the start of available memory; there is no guarantee of lossless or unlimited recall.',
      'В базовой схеме в MLP поступает только h_s. Оба состояния h_s и c_s сохраняются для следующего шага LSTM. Эти 64 числа являются обученными координатами, а не 64 заранее заданными индикаторами. Инициализация задаёт начало доступной памяти; точное и неограниченное запоминание не гарантируется.')

    w.page('3 The recent window and the join','3 Недавнее окно и объединение')
    P('Construct the recent matrix directly from market feature rows. Rows run from oldest to newest; columns follow a fixed feature manifest. This is a parallel input branch: the recent matrix is not produced from the LSTM summary.',
      'Недавняя матрица строится непосредственно из строк рыночных признаков. Строки идут от старых к новым; колонки следуют фиксированному списку признаков. Это параллельная входная ветвь: недавняя матрица не строится из сводки LSTM.')
    E('split')
    P('Each row contains all F features for one minute. Flattening places the first row into the vector, followed by the second row, and so on. It preserves every supplied value and its position. It does not average the window or compress it.',
      'Каждая строка содержит все F признаков одной минуты. Развёртка помещает в вектор первую строку, затем вторую и так далее. Все поданные значения и их позиции сохраняются. Усреднения или сжатия окна на этом этапе нет.')
    E('flat');E('index')
    P('In the index equation j identifies a row within the recent window and q a feature column; indices start at 1. A single volume spike therefore occupies a specific vector position. The chosen feature may already be transformed: log volume preserves a positive volume value through an invertible transform, whereas an indicator alone does not reconstruct all raw candle fields.',
      'В индексной формуле j задаёт строку недавнего окна, q — колонку признака; индексация начинается с 1. Поэтому отдельный всплеск объёма занимает конкретную позицию в векторе. Сам признак уже может быть преобразован: логарифм положительного объёма обратим, а один индикатор не восстанавливает все поля исходной свечи.')
    P('Concatenate the recent vector, the older summary and the current anchor vector in that fixed order. Concatenation only joins values. Learned mixing starts in the first dense layer. [2]',
      'Объединим в фиксированном порядке недавний вектор, старую сводку и текущие флаги якорей. Объединение лишь соединяет значения. Обучаемое смешивание начинается в первом полносвязном слое. [2]')
    E('concat');E('numbers')
    T(w.pick(['Part','Positions in z with zero based indexing','Size'],['Часть','Позиции в z при индексации с нуля','Размер']),[
      [w.pick('Recent detail','Недавние данные'),'0 … 2159','2160'],
      [w.pick('Older LSTM summary','Старая сводка LSTM'),'2160 … 2223','64'],
      [w.pick('Current anchor flags','Текущие флаги якорей'),'2224 … 2241','18'],
    ],[5,9,3.2])
    P('The current one-row feature export has 54 inputs. This proposed windowed network has 2242 inputs to its first MLP layer. Producing that input requires the complete minute timeline, not only the irregularly spaced anchor rows.',
      'Текущий экспорт одной строки признаков содержит 54 входа. У предлагаемой сети с окном первый слой MLP получает 2242 входа. Для их формирования нужна полная минутная история, а не только нерегулярно расположенные строки якорей.')

    w.page('4 The MLP forward pass','4 Прямой проход через MLP')
    P('For a concrete example, use two hidden layers of widths 128 and 64, followed by one logit. ReLU is applied in hidden layers. Sigmoid converts the final logit into a number between 0 and 1. The architecture produces an estimated probability; calibration must be evaluated separately.',
      'Для конкретного примера возьмём два скрытых слоя размером 128 и 64, затем один логит. В скрытых слоях применяется ReLU. Сигмоида преобразует итоговый логит в число от 0 до 1. Это оценка вероятности; её калибровку нужно проверять отдельно.')
    E('mlp1');E('mlp2');E('mlp3')
    T(w.pick(['Parameter','Shape','Role'],['Параметр','Размер','Назначение']),[
      ['W₁ / b₁','[128,2242] / [128]',w.pick('Mix the combined input','Смешать объединённый вход')],
      ['W₂ / b₂','[64,128] / [64]',w.pick('Transform the first hidden activations','Преобразовать активации первого слоя')],
      ['w₃ / b₃','[1,64] / [1]',w.pick('Produce one logit','Получить один логит')],
    ],[2.4,5.4,9.4])
    P('Partition the first layer by the three input blocks. Its pre-activation q₁ shows exactly where the LSTM output enters the calculation.',
      'Разобьём первый слой по трём блокам входа. Его значение до активации q₁ показывает точное место участия выхода LSTM.')
    E('blocks');E('influence')
    P('W_R has shape [128,2160], W_M [128,64], and W_A [128,18]. Thus each memory value can influence every first-layer neuron through a learned weight. It changes the activation values passed to subsequent layers. During prediction all weights remain fixed; the input values and LSTM states change. [3]',
      'Размер W_R равен [128,2160], W_M — [128,64], W_A — [128,18]. Поэтому каждое значение памяти может влиять на каждый нейрон первого слоя через обученный вес. Оно меняет активации, поступающие в последующие слои. При прогнозе все веса фиксированы; меняются входные значения и состояния LSTM. [3]')
    P('For a batch, examples are stored as rows of Z [B,2242]. This requires a transpose of the column-vector weights. Biases broadcast across the batch rows.',
      'В пакете примеры хранятся строками матрицы Z [B,2242]. Поэтому веса из записи для векторов столбцов транспонируются. Смещения добавляются к каждой строке пакета.')
    E('batched')
    P('For B = 32 the layer outputs are [32,128], [32,64], [32,1] and [32,1]. One output represents two classes: class 1 has estimated probability p and class 0 has estimated probability 1 − p.',
      'При B = 32 выходы слоёв имеют размеры [32,128], [32,64], [32,1] и [32,1]. Один выход представляет два класса: оценка вероятности класса 1 равна p, класса 0 — 1 − p.')

    w.page('5 A numerical example of memory influence','5 Числовой пример влияния памяти')
    P('Consider two samples with identical recent features and current anchor flags. Only one LSTM memory coordinate differs. The following chosen weights illustrate the arithmetic; they are not fitted model parameters or measured trading results.',
      'Рассмотрим два примера с одинаковыми недавними признаками и текущими флагами якорей. Различается только одна координата памяти LSTM. Выбранные ниже веса показывают арифметику; это не параметры обученной модели и не результаты торговли.')
    P('Let the recent features, anchor flags and bias together contribute 0.3 to one first-layer neuron. Give memory coordinate m₁ weight 0.8. Pass this one active coordinate through an identity second-layer connection, and let the final logit be 2u − 0.2. All other relevant contributions are zero in this illustration.',
      'Пусть недавние признаки, флаги якорей и смещение вместе дают вклад 0.3 в один нейрон первого слоя. Вес координаты памяти m₁ равен 0.8. Передадим эту единственную активную координату через связь второго слоя с весом 1 и зададим итоговый логит 2u − 0.2. Остальные соответствующие вклады в примере равны нулю.')
    T(w.pick(['Quantity','Sample A','Sample B'],['Величина','Пример A','Пример B']),[
      ['m₁','0.5','−0.5'],
      ['q = 0.3 + 0.8m₁','0.7','−0.1'],
      ['u = ReLU(q)','0.7','0'],
      ['ℓ = 2u − 0.2','1.2','−0.2'],
      ['p = σ(ℓ)','0.7685','0.4502'],
    ],[9.2,4,4])
    P('The recent pattern is identical, but the older context changes the probability. The size and direction of this effect are learned. If the memory block weights are zero, its contribution to this layer is zero.',
      'Недавний паттерн одинаков, но старый контекст меняет вероятность. Размер и направление влияния определяются обучением. При нулевых весах блока памяти его вклад в этот слой равен нулю.')
    H('Direct values and compressed values','Прямые и сжатые значения',2)
    E('direct');E('compressed')
    P('These partial derivatives hold the other supplied inputs fixed. A recent position n has its own first-layer weight. An older x_(j,q) acts through the LSTM summary. A raw-market change could also alter recent indicators through preprocessing. Later nonlinear layers can suppress either contribution. Direct access does not guarantee use of an outlier.',
      'Эти частные производные рассматриваются при фиксированных остальных входах. У недавней позиции n есть собственный вес первого слоя. Старый x_(j,q) действует через сводку LSTM. Изменение исходного рынка также может изменить недавние индикаторы через предобработку. Последующие нелинейные слои способны подавить любой вклад. Прямой доступ не гарантирует использования выброса.')
    P('The LSTM cell equation also explains forgetting. Holding gate values fixed, the direct contribution of an old cell component is multiplied by successive forget gates. For example, 120 multiplications by 0.95 leave about 0.00212 of that contribution. This is an isolated arithmetic path, not the complete derivative of an actual LSTM.',
      'Уравнение памяти LSTM также объясняет забывание. Если считать значения вентилей фиксированными, прямой вклад старой компоненты памяти умножается на последовательность вентилей забывания. Например, 120 умножений на 0.95 оставляют около 0.00212 исходного вклада. Это отдельный арифметический путь, а не полная производная реальной LSTM.')

    w.page('6 Moving the boundary through time','6 Перемещение границы во времени')
    P('A simple causal implementation maintains the LSTM exactly L minutes behind the latest completed candle. The recent window stores the most recent L feature rows. Each completed minute transfers the oldest recent row into the older branch.',
      'Простая причинная реализация поддерживает LSTM ровно на L минут позади последней закрытой свечи. Недавнее окно хранит L последних строк признаков. С закрытием каждой минуты самая старая строка окна передаётся в старую ветвь.')
    T(w.pick(['Decision time','LSTM includes candles through','Detailed window'],['Время решения','LSTM включает свечи до','Подробное окно']),[
      ['13:05','12:04','12:05–13:04'],['13:06','12:05','12:06–13:05'],['13:10','12:09','12:10–13:09'],
    ],[3.6,6.8,6.8])
    E('move')
    P('After this update the new summary is h_(s+1). Remove x_(s+1) from the detailed window and append x_(t+1). The new MLP input still has 2242 values. No earlier candle is processed twice by this delayed recurrent stream.',
      'После обновления новой сводкой становится h_(s+1). Удаляем x_(s+1) из подробного окна и добавляем x_(t+1). Новый вход MLP по-прежнему содержит 2242 значения. В таком отложенном рекуррентном потоке старая свеча не обрабатывается повторно.')
    P('An equivalent inference design processes the LSTM up to the latest minute and keeps snapshots. At decision t it selects the snapshot from t − L. With identical initialization and fixed weights this produces the same older summary. It must not accidentally supply h_t instead, because h_t also contains the recent window.',
      'Эквивалентная схема прогноза обрабатывает LSTM до последней минуты и хранит снимки состояний. Для решения t она выбирает снимок на t − L. При одинаковой инициализации и фиксированных весах старая сводка получается той же. Нельзя случайно подставить h_t: оно также включает недавнее окно.')
    H('State continuity','Непрерывность состояния',2)
    B('Advance the minute timeline even when no main anchor is active and while a series is open. Only the prediction and execution decisions are gated by anchors and position rules.',
      'Обновлять минутную историю и без основных якорей, и во время открытой серии. Якоря и правила позиции ограничивают принятие решений, а не обновление рыночной памяти.')
    B('Save the model version, feature order, normalization settings, boundary index, h, c and the recent buffer for reproducible inference. After a model-weight change, rebuild states with that model.',
      'Для воспроизводимого прогноза сохранять версию модели, порядок признаков, настройки нормализации, индекс границы, h, c и недавний буфер. После изменения весов состояния нужно пересчитать новой моделью.')
    B('Initialize from a defined history start and warm up before eligible predictions. Reset state at independent streams or replay from allowed past data; never carry a state computed from later candles backward in time.',
      'Инициализировать сеть с определённого начала истории и прогреть до допустимых прогнозов. Для независимых потоков сбрасывать состояние или восстанавливать его по доступному прошлому; не переносить назад состояние, вычисленное по будущим свечам.')

    w.page('7 Training and the binary target','7 Обучение и бинарная цель')
    P('A forward pass uses only input values and fixed current weights. Training additionally compares its output with a known historical outcome and updates weights. The MLP has no recurrent state of its own in this design. The recurrent feedback within the LSTM and the training feedback through gradients are different mechanisms.',
      'Прямой проход использует только входы и текущие фиксированные веса. При обучении выход дополнительно сравнивается с известным историческим результатом, после чего веса обновляются. В данной схеме у MLP нет собственного рекуррентного состояния. Рекуррентная связь внутри LSTM и обратная связь обучения через градиенты — разные механизмы.')
    P('Let R_net,t be the complete series return after the applicable execution and borrowing costs, expressed consistently with the threshold τ. The label equals 1 only when that return exceeds τ. Class 0 includes losses, breakeven and returns below the chosen minimum. A probability cutoff for admitting an order is a separate setting.',
      'Пусть R_net,t — доходность полной серии после применимых издержек исполнения и заимствования, в тех же единицах, что и порог τ. Метка равна 1 только при превышении τ. Класс 0 включает убытки, безубыточность и доходность ниже выбранного минимума. Порог вероятности для допуска заявки задаётся отдельно.')
    E('target')
    P('The existing field target_gross_return_pct is a gross series result, not this net binary label. Its legacy definition sums per-trade percentage returns. A future label builder must define costs, sizing and the return denominator explicitly. Unfilled initial orders and series with unavailable final targets must not silently become losses.',
      'Существующее поле target_gross_return_pct содержит валовой результат серии, а не такую чистую бинарную метку. В исходной логике оно суммирует процентные результаты отдельных сделок. При создании меток нужно явно определить издержки, размеры позиций и базу расчёта доходности. Неисполненные первоначальные заявки и серии без доступного итогового результата нельзя автоматически считать убытками.')
    P('Use a mask v_t equal to 1 for main-anchor opportunities with valid targets and 0 elsewhere. Compute the following binary cross entropy only when the sum of the mask is positive. Non-anchor minutes supply context without receiving a negative label. In code, a logits-based loss is numerically preferable to explicitly taking logarithms of sigmoid output. [4]',
      'Маска v_t равна 1 для возможностей по основным якорям с корректными целями и 0 в остальных случаях. Следующая бинарная перекрёстная энтропия вычисляется только при положительной сумме маски. Минуты без якоря дают контекст и не получают отрицательную метку. В коде численно устойчивее функция потерь по логитам, чем явное логарифмирование выхода сигмоиды. [4]')
    E('loss');E('grad');E('update')
    P('θ denotes trainable weights and η the learning rate; the update is a simple gradient-descent illustration. The loss gradient reaches the memory coordinates through the MLP and then reaches LSTM weights through the recurrent computation. Both branches can therefore learn from the same binary target. An intentional detach or a precomputed frozen summary blocks that gradient at the boundary.',
      'θ обозначает обучаемые веса, η — скорость обучения; обновление показано на примере простого градиентного спуска. Градиент потерь проходит через MLP к координатам памяти, затем через рекуррентные вычисления к весам LSTM. Поэтому обе ветви могут обучаться на одной бинарной цели. Явный detach или заранее вычисленная замороженная сводка останавливают этот градиент на границе.')

    w.page('8 Batches and the learning horizon','8 Пакеты и горизонт обучения')
    P('The three-dimensional input is a storage convention for batches of sequences. It remains useful for this hybrid design. Processing in chunks avoids rebuilding a separate full historical prefix for every anchor.',
      'Трёхмерный вход — способ хранения пакетов последовательностей. Он остаётся полезным и в этой гибридной схеме. Обработка блоками позволяет не собирать заново всю старую историю для каждого якоря.')
    T(w.pick(['Object','General shape','Example with B 32'],['Объект','Общий размер','Пример при B 32']),[
      [w.pick('Older input chunk','Блок старых входов'),'[B,K,F]','[32,256,36]'],
      [w.pick('Hidden and cell states each','Скрытое состояние и память каждое'),'[1,B,H]','[1,32,64]'],
      [w.pick('Summary selected at boundary','Сводка на нужной границе'),'[B,H]','[32,64]'],
      [w.pick('Recent windows','Недавние окна'),'[B,L,F]','[32,60,36]'],
      [w.pick('Flattened recent values','Развёрнутые недавние значения'),'[B,LF]','[32,2160]'],
      [w.pick('Current anchors','Текущие якоря'),'[B,A]','[32,18]'],
      [w.pick('Combined MLP input','Общий вход MLP'),'[B,LF+H+A]','[32,2242]'],
      [w.pick('Logits probabilities targets','Логиты вероятности цели'),'[B,1]','[32,1]'],
    ],[7.2,4.6,5.4])
    P('The state shape [1,B,H] follows the PyTorch convention for one layer and one direction; batch_first changes the input convention, not the state convention. A summary used in the MLP is selected from that layer as [B,H]. Padded sequences require true lengths or a valid mask. Padding zeros must not overwrite the state after the final real step.',
      'Размер состояния [1,B,H] соответствует соглашению PyTorch для одного слоя и одного направления; batch_first меняет представление входа, но не состояния. Сводка для MLP выбирается из этого слоя в виде [B,H]. При дополнении последовательностей нужны реальные длины или корректная маска. Нули дополнения не должны обновлять состояние после последнего настоящего шага.')
    P('K is the processed chunk length, not the detailed lookback length L. With truncated backpropagation, h and c may continue numerically across chunks while their earlier computation graph is detached. Thus retained history and the range over which an outcome can teach the LSTM are different. Detaching the entire older branch before every eligible loss prevents its end-to-end learning. [5]',
      'K — длина обрабатываемого блока, а не длина подробного окна L. При усечённом обратном распространении h и c могут численно переходить между блоками, а граф предыдущих вычислений отсоединяется. Поэтому сохранённая история и расстояние, на котором результат обучает LSTM, различаются. Отсоединение всей старой ветви перед каждой целевой ошибкой не даст обучать её сквозным образом. [5]')
    P('Each persistent batch slot must continue the same chronological stream. Independent examples may be batched if each has its own correct prefix and state. Chronological evaluation must separate training outcomes from validation and test periods, including overlapping target intervals. Replaying allowed past inputs with fixed trained weights is compatible with causal evaluation.',
      'Каждая постоянная позиция пакета должна продолжать тот же хронологический поток. Независимые примеры можно объединять, если у каждого свой корректный префикс и состояние. При хронологической оценке обучающие исходы отделяются от периодов валидации и теста, включая пересекающиеся интервалы целей. Проигрывание разрешённого прошлого с фиксированными обученными весами совместимо с причинной оценкой.')

    w.page('9 Reference forward algorithm','9 Алгоритм прямого прохода')
    P('The pseudocode below describes one stream and one decision. The feature timeline is already causal, finite and consistently ordered. h and c were initialized at the defined history start; older_cursor identifies the last minute already absorbed by the older branch. The loop also handles several new minutes between anchor decisions.',
      'Псевдокод ниже описывает один поток и одно решение. История признаков уже причинна, содержит конечные значения и имеет постоянный порядок колонок. h и c инициализированы в определённом начале истории; older_cursor указывает последнюю минуту, включённую в старую ветвь. Цикл также обрабатывает несколько новых минут между решениями по якорям.')
    code='''# t: last completed M1 index; x[j] has F values
s = t - L
while older_cursor < s:
    older_cursor += 1
    h, c = lstm_cell(x[older_cursor], (h, c))

R = x[s + 1 : t + 1]           # [L,F], oldest first
r = R.reshape(L * F)           # [LF]
m = h                          # [H], state at s
a = anchor_flags_at_decision    # [A]
z = concatenate([r, m, a])     # [LF+H+A]

u1 = relu(W1 @ z + b1)         # [128]
u2 = relu(W2 @ u1 + b2)        # [64]
logit = w3 @ u2 + b3           # [1]
p = sigmoid(logit)             # [1]'''
    code_ru='''# t: индекс последней закрытой M1; x[j] содержит F значений
s = t - L
while older_cursor < s:
    older_cursor += 1
    h, c = lstm_cell(x[older_cursor], (h, c))

R = x[s + 1 : t + 1]           # [L,F], от старых к новым
r = R.reshape(L * F)           # [LF]
m = h                          # [H], состояние на s
a = anchor_flags_at_decision    # [A]
z = concatenate([r, m, a])     # [LF+H+A]

u1 = relu(W1 @ z + b1)         # [128]
u2 = relu(W2 @ u1 + b2)        # [64]
logit = w3 @ u2 + b3           # [1]
p = sigmoid(logit)             # [1]'''
    w.code(code,code_ru)
    P('Keep the current anchor flags tied to order placement time even if the initial order fills later. At 13:05, the latest closed M5 candle may create an anchor, while the 13:05 M1 candle has just opened and supplies no completed-candle features.',
      'Текущие флаги якорей привязываются ко времени выставления заявки, даже если первоначальное исполнение происходит позже. В 13:05 последняя закрытая свеча M5 может дать якорь, а свеча M1 с открытием в 13:05 только началась и ещё не даёт признаков закрытой свечи.')
    H('Dimension arithmetic','Расчёт размерностей',2)
    P('The first dense layer has 128·2242 + 128 = 287232 parameters; the second has 64·128 + 64 = 8256; the output has 64 + 1 = 65. Total MLP parameters: 295553. The canonical single-bias LSTM has 4H(F + H + 1) = 25856 parameters; total: 321409.',
      'Первый полносвязный слой содержит 128·2242 + 128 = 287232 параметра; второй — 64·128 + 64 = 8256; выход — 64 + 1 = 65. Всего в MLP 295553 параметра. Каноническая LSTM с одним смещением на вентиль содержит 4H(F + H + 1) = 25856 параметров; всего — 321409.')
    P('PyTorch stores two bias vectors per gate, so its LSTM count is 26112 and the combined count 321665. Summing its input and recurrent biases gives the effective bias used in the equations. These counts describe the example architecture, not evidence that it is appropriately sized for the available training sample.',
      'PyTorch хранит два вектора смещений на вентиль, поэтому его LSTM содержит 26112 параметров, а вся сеть — 321665. Сумма входного и рекуррентного смещений даёт эффективное смещение в формулах. Эти числа описывают пример архитектуры и не доказывают, что такой размер подходит доступной обучающей выборке.')

    w.page('10 Outliers and implementation boundaries','10 Выбросы и границы реализации')
    T(w.pick(['Where the value exists','What reaches the MLP'],['Где находится значение','Что получает MLP']),[
      [w.pick('Selected feature inside the recent window','Выбранный признак внутри недавнего окна'),w.pick('Its separate transformed numeric value and exact lag position.','Его отдельное преобразованное числовое значение и точную позицию по времени.')],
      [w.pick('Selected feature before the recent window','Выбранный признак до недавнего окна'),w.pick('Compressed influence via the LSTM; recent derived indicators may also retain indirect influence. Exact retrieval is not guaranteed.','Сжатое влияние через LSTM; недавние индикаторы также могут сохранить косвенное влияние. Точное восстановление не гарантируется.')],
      [w.pick('Raw candle field absent from selected features','Сырое поле свечи отсутствует в выбранных признаках'),w.pick('No direct value. Presence in an inspection spreadsheet does not add it to the network input.','Прямого значения нет. Наличие в проверочной таблице не добавляет поле во вход сети.')],
    ],[7,10.2])
    P('If a particular older single event must remain explicitly accessible, add a defined event vector e_t. Examples are an event flag, its magnitude and elapsed minutes since it occurred, computed from past data only. This changes the model contract and dimensions; it cannot be assumed to emerge reliably from compression.',
      'Если конкретное старое единичное событие должно оставаться явно доступным, добавляется определённый вектор событий e_t. Например, флаг события, его величина и число прошедших минут, вычисленные только по прошлым данным. Это меняет состав входов и размерности; нельзя считать, что сжатие обязательно сохранит событие само.')
    E('extra')
    P('For E additional explicit values, the MLP input grows by E. Supplying both h_s and c_s instead of h_s alone would similarly change H to 2H in the join, giving 2306 inputs in this example. Such variants require their own training and evaluation.',
      'При E дополнительных явных значениях вход MLP увеличивается на E. Аналогично, передача h_s и c_s вместо одного h_s заменяет H на 2H при объединении и даёт 2306 входов в этом примере. Для таких вариантов нужны отдельное обучение и оценка.')
    H('Implementation review points','Пункты проверки реализации',2)
    B('Verify the exact s = t − L boundary, chronological flatten order, feature manifest and state-to-stream mapping. Check the emitted shapes against Section 8.',
      'Проверить точную границу s = t − L, хронологический порядок развёртки, список признаков и соответствие состояний потокам. Сверить выходные размеры с разделом 8.')
    B('Exclude future fill prices, stops, exits, target values and future timestamps from inputs. Preserve all eligible hypothetical anchor samples; apply the chronological admission rule separately during execution evaluation.',
      'Исключить будущие цены исполнения, стопы, выходы, цели и будущие времена из входов. Сохранять все допустимые гипотетические примеры по якорям; хронологическое ограничение допуска применять отдельно при оценке исполнения.')
    B('Establish whether LSTM gradients actually reach older inputs during training. Compare the full model with the same recent-window MLP alone before attributing an improvement to older context.',
      'Установить, доходят ли градиенты до старых входов LSTM при обучении. Сравнить полную модель с той же MLP на одном недавнем окне, прежде чем связывать улучшение со старым контекстом.')
    H('Technical references','Технические источники',2)
    refs_en=[
      '[1] PyTorch LSTM — recurrence and state shapes\nhttps://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html',
      '[2] Keras Concatenate — joining input vectors\nhttps://keras.io/api/layers/merging_layers/concatenate/',
      '[3] PyTorch Linear — affine transform and weight layout\nhttps://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html',
      '[4] PyTorch BCEWithLogitsLoss — binary loss from logits\nhttps://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html',
      '[5] Official PyTorch recurrent example — state detachment\nhttps://github.com/pytorch/examples/blob/main/word_language_model/main.py',
    ]
    refs_ru=[
      '[1] PyTorch LSTM — рекуррентные формулы и размеры состояний\nhttps://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html',
      '[2] Keras Concatenate — объединение входных векторов\nhttps://keras.io/api/layers/merging_layers/concatenate/',
      '[3] PyTorch Linear — аффинное преобразование и расположение весов\nhttps://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html',
      '[4] PyTorch BCEWithLogitsLoss — бинарная ошибка по логитам\nhttps://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html',
      '[5] Официальный пример PyTorch — отсоединение графа состояний\nhttps://github.com/pytorch/examples/blob/main/word_language_model/main.py',
    ]
    refs=refs_en if lang=='en' else refs_ru
    for ref in refs:
        p=w.d.add_paragraph(ref);p.paragraph_format.space_after=Pt(4)
        for run in p.runs:run.font.size=Pt(9)
    P('Architecture equations are a specification and worked derivation. Feature counts were checked against the local model_features schema on 26 October 2026. No model-performance claim is made.',
      'Формулы задают архитектуру и показывают её расчёт. Число признаков сверено с локальной схемой model_features 26 октября 2026 года. Утверждений о качестве модели здесь нет.')
    path=ROOT/('LSTM_lookback_MLP_forward_pass_EN.docx' if lang=='en' else 'LSTM_lookback_MLP_forward_pass_RU.docx')
    w.d.save(path)
    return {'language':lang,'path':str(path),'equations':w.used,'paragraphs':len(w.d.paragraphs),'tables':len(w.d.tables)}

if __name__=='__main__':
    result=[build('en'),build('ru')]
    (ROOT/'build_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
