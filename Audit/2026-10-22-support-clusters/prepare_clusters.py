"""One-time support-search display enrichment; no strategy or target changes."""
import bisect, csv, json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TZ=timezone(timedelta(hours=3))
source=ROOT/'Handoff/2026-10-17-entry-rule/datasets/support_searches.csv'
rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
by_key={(int(r['entry_id']),int(r['support_search_no'])):r for r in rows}
keys=[tuple(x) for x in json.loads((OUT/'live-key-order.json').read_text())]
if len(keys)!=len(rows) or len(set(keys))!=len(keys) or set(keys)!=set(by_key):
    raise ValueError('Live and saved support-search keys differ')
candles=json.loads((ROOT/'Handoff/2026-10-16/candles/1m.json').read_text(encoding='utf8'))
candles.sort(key=lambda c:c[0])
mts=[c[0] for c in candles]
def parse_times(s):
    return [int(datetime.strptime(t.strip(),'%d.%m.%Y %H:%M').replace(tzinfo=TZ).timestamp()*1000) for t in s.split(';') if t.strip()]
def fmt(ts):
    return '; '.join(datetime.fromtimestamp(t/1000,TZ).strftime('%d %b %H:%M') for t in sorted(ts))
payload=[]; above_cases=[]; counts={'type2':0,'type2_with_above_outliers':0}; examples=[]
for key in keys:
    r=by_key[key]
    lower=parse_times(r['support_lower_cluster_times'])
    upper=parse_times(r['support_base_cluster_times'])
    below=lower;outside=lower
    if r['support_type']=='type2':
        start=int(float(r['source_wave_start_ms_utc']));end=int(float(r['source_wave_end_ms_utc']))
        wave=candles[bisect.bisect_left(mts,start):bisect.bisect_left(mts,end)]
        if len(wave)!=int(r['support_low_count']):raise ValueError(('Source candle count mismatch',key))
        upper_set=set(upper)
        if not upper_set or not upper_set.issubset({c[0] for c in wave}):raise ValueError(('Invalid upper times',key))
        base=min(c[4] for c in wave if c[0] in upper_set)
        outside=[c[0] for c in wave if c[0] not in upper_set]
        tolerance=float(r['support_threshold'])
        below=[c[0] for c in wave if c[0] not in upper_set and c[4]<base and (base-c[4])/abs(base)>tolerance]
        above=[c[0] for c in wave if c[0] not in upper_set and c[4]>=base]
        counts['type2']+=1
        if above:
            counts['type2_with_above_outliers']+=1
            if len(above_cases)<3:above_cases.append({'key':key,'above_count':len(above)})
    item={'entry_id':key[0],'search_no':key[1],'type':r['support_type'],'lower_saved':fmt(lower),'lower_below':fmt(below),'lower_all_outside':fmt(outside),'upper':fmt(upper)}
    payload.append(item)
    if key==(355,3):examples.append(item)
(OUT/'cluster-times.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf8')
(OUT/'summary.json').write_text(json.dumps({'rows':len(payload),'counts':counts,'above_examples':above_cases,'case355':examples},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'rows':len(payload),'counts':counts,'above_examples':above_cases,'case355':examples},ensure_ascii=False,indent=2))
