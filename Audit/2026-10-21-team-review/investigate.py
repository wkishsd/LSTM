"""Read-only reconstruction for the team's 21 October comments. No strategy edits."""
import csv, hashlib, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.python-deps'))
import pandas as pd
spec = importlib.util.spec_from_file_location('review_production', ROOT / 'Pipelines/Bitfinex/bitfinex_indicators_v2.1.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
P = ROOT / 'Handoff/2026-10-17-entry-rule'
OUT = Path(__file__).resolve().parent
native, cutoff = m.load_training_candles(ROOT / 'Handoff/2026-10-16/candles')
h = m.calculate_native(native['5m']).copy()
h['rsi_min10'] = h.rsi_14.shift(1).rolling(10).min()
h['atr_q75'] = h.atr_14.shift(1).rolling(80).quantile(.75)
h['volume_mean20'] = h.volume.shift(1).rolling(20).mean()
h['di_spread'] = h.di_minus_14 - h.di_plus_14
h['di_minus_change'] = h.di_minus_14.diff()
entry = pd.read_csv(P/'datasets/entries.csv').query('entry_id==355').iloc[0]
signal = pd.Timestamp(entry.entry_at) - pd.Timedelta(minutes=5)
j = h.index[h.timestamp.eq(signal)][0]
loc = h.index.get_loc(j)
cols = ['timestamp','rsi_14','rsi_min10','atr_14','atr_q75','volume','volume_mean20','di_plus_14','di_minus_14','di_spread','di_minus_change']
flags = m.detect_anchor_events(h)
saved = [int(k.split('_')[1]) for k in entry.index if k.startswith('anchor_') and int(entry[k])]
recomputed = [int(k.split('_')[1]) for k,v in flags.loc[pd.Timestamp(entry.entry_at)].items() if v]
searches = pd.read_csv(P/'datasets/support_searches.csv')
search = searches.loc[(searches.entry_id==355)&(searches.support_search_no==3)].iloc[0]
mins = native['1m'].copy()
source = mins.loc[(mins.timestamp>=pd.Timestamp(int(search.source_wave_start_ms_utc),unit='ms',tz='UTC'))&(mins.timestamp<pd.Timestamp(int(search.source_wave_end_ms_utc),unit='ms',tz='UTC'))].copy()
source['time_utc3'] = source.timestamp.dt.tz_convert('Etc/GMT-3')
source['time_utc3'] = source['time_utc3'].map(lambda t:t.isoformat())
source['above_min_pct'] = (source.low/source.low.min()-1)*100
base = 2449.3
source['distance_from_upper_base_pct'] = abs(source.low-base)/base*100
def rows(df):
    return json.loads(df.to_json(orient='records',date_format='iso'))
result = {
 'production_sha256':hashlib.sha256((ROOT/'Pipelines/Bitfinex/bitfinex_indicators_v2.1.py').read_bytes()).hexdigest(),
 'entry_id':355,'entry_at':entry.entry_at,'signal':str(signal),
 'saved_main_anchors':entry.main_anchor_numbers,'all_saved_flags':saved,'recomputed_flags':recomputed,
 'anchor_context':rows(h.iloc[loc-11:loc+1][cols]),
 'prior80_atr':rows(h.iloc[loc-80:loc][['timestamp','atr_14']]),
 'prior20_volume':rows(h.iloc[loc-20:loc][['timestamp','volume']]),
 'search3':search.where(pd.notna(search),None).to_dict(),
 'source_lows':rows(source[['time_utc3','open','high','low','close','distance_from_upper_base_pct']]),
 'upper_cluster_base':base,'upper_cluster_band_usd':base*.0003,
 'upper_cluster_last_low_relative_distance':(2449.7-base)/base,
 'groups':m.anchor_group_definitions(),
}
(OUT/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf8')
for name, frame in [('support-source-lows',source[['time_utc3','low','distance_from_upper_base_pct']]),('previous-80-atr',h.iloc[loc-80:loc][['timestamp','atr_14']]),('previous-20-volume',h.iloc[loc-20:loc][['timestamp','volume']]),('anchor-comparison',h.iloc[loc-1:loc+1][cols])]:
    frame=frame.copy()
    if 'timestamp' in frame:
        frame['timestamp']=frame.timestamp.dt.tz_convert('Etc/GMT-3').map(lambda t:t.isoformat())
    frame.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
print(json.dumps({k:result[k] for k in ['entry_id','entry_at','signal','saved_main_anchors','recomputed_flags','source_lows','upper_cluster_band_usd']},ensure_ascii=False,indent=2))
print(h.iloc[loc-1:loc+1][cols].to_string(index=False))
