"""Build only the two requested display columns, aligned by live Entry/Search keys."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
records=json.loads((P/'cluster-times.json').read_text(encoding='utf8'))
rows=[]
for r in records:
    lower=r['lower_below'] if r['type']=='type2' else r['lower_saved']
    rows.append({'values':[{'userEnteredValue':{'stringValue':lower}},{'userEnteredValue':{'stringValue':r['upper']}}]})
size=600
for start in range(0,len(rows),size):
    (P/f'cells-{start//size:02d}.json').write_text(json.dumps({'start':start+1,'end':min(start+size,len(rows))+1,'rows':rows[start:start+size]},ensure_ascii=False),encoding='utf8')
with (P/'support-cluster-times.csv').open('w',encoding='utf-8-sig',newline='') as f:
    import csv
    w=csv.writer(f);w.writerow(['entry_id','support_search_no','lower_cluster_candle_times_utc3','upper_cluster_candle_times_utc3'])
    for r in records:w.writerow([r['entry_id'],r['search_no'],r['lower_below'] if r['type']=='type2' else r['lower_saved'],r['upper']])
print(json.dumps({'rows':len(rows),'batches':(len(rows)+size-1)//size,'max_cell_chars':max(len(c['userEnteredValue']['stringValue']) for row in rows for c in row['values'])}))
