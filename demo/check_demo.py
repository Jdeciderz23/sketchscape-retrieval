"""Check a running local demo against the official gallery and saved experiment ranks."""
import argparse
import base64
import csv
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args()
    origin=f'http://127.0.0.1:{args.port}'
    def request(path,body=None):
        headers={'Content-Type':'image/png'} if body else {}
        with urlopen(Request(origin+path,data=body,headers=headers),timeout=30) as response:
            return json.load(response)
    session=request('/api/session')
    assert session['gallery']==3000
    ids=set((ROOT/'data/fscoco/val_normal.txt').read_text().splitlines())
    catalog=request('/api/gallery')
    assert len(catalog)==len(ids) and {row['id'] for row in catalog}==ids
    case_ids={case['id'] for case in session['cases']}
    assert len(case_ids)==len(session['cases'])==12
    cache=ROOT/'outputs/adapter/cache/normal/manifest.json'
    folders={k:u for u,k in json.loads(cache.read_text())['ids']}
    with (ROOT/'outputs/adapter/normal/test_ranks.csv').open() as f:
        recorded={r['id']:r for r in csv.DictReader(f)}
    checked=0
    for case in session['cases']:
        key=case['id']
        raw=json.loads((ROOT/'data/fscoco/raw_data'/folders[key]/(key+'.json')).read_bytes())
        seq=request('/api/case?id='+key)
        clock=raw[0]['timestamp']
        assert len(seq['points'])==len(raw)
        for point, frame in zip(raw, seq['points']):
            clock=max(clock,point['timestamp'])
            x,y=point['coordinates']
            assert frame==[int(float(x)*256), int(float(y)*256), clock-raw[0]['timestamp'],
                           int(point['pen_state']==[1,0,0])]
        assert seq['duration']==seq['points'][-1][2]>0
        for pct in (25,50,75,100):
            result=request(f'/api/replay?id={key}&tick={100*pct}')
            imgdir=ROOT/'data/time_prefix_v1'/str(pct) if pct<100 else ROOT/'data/fscoco/raster_sketches'
            original=imgdir/folders[key]/(key+'.png')
            with Image.open(original) as saved, Image.open(io.BytesIO(base64.b64decode(result['raster']))) as rendered:
                assert ImageChops.difference(saved,rendered).getbbox() is None
            for side in ('base','adapter'):
                top=result[side]['top']
                assert len({x['id'] for x in top})==5
                assert all(x['id'] in ids for x in top)
                assert [x['score'] for x in top]==sorted((x['score'] for x in top),reverse=True)
                assert result[side]['rank']==int(recorded[key][f'{side}_rank_{pct}'])
                assert top[0]['id']==recorded[key][f'{side}_first_{pct}']
            checked+=1
            if pct==25:
                assert result['adapter']['rank']==1 and result['base']['rank']>1
    other=next(row['id'] for row in catalog if row['id'] not in case_ids)
    with urlopen(origin+'/thumb/'+other) as response, Image.open(io.BytesIO(response.read())) as thumb:
        assert thumb.format=='JPEG' and thumb.width<=320 and thumb.height<=240
    query=ROOT/'data/time_prefix_v1/25'/folders[other]/(other+'.png')
    independent=request('/api/draw?target='+other,query.read_bytes())
    for side in ('base','adapter'):
        assert independent[side]['rank']==int(recorded[other][side+'_rank_25'])
    key=session['cases'][0]['id']
    path=ROOT/'data/time_prefix_v1/25'/folders[key]/(key+'.png')
    live=request('/api/draw?target='+key,path.read_bytes())
    assert [live[s]['rank'] for s in ('base','adapter')]==[int(recorded[key][s+'_rank_25']) for s in ('base','adapter')]
    full=ROOT/'data/fscoco/raster_sketches'/folders[key]/(key+'.png')
    replay_all=request('/api/replay?id='+key+'&tick=10000&always=1')
    drawn_all=request('/api/draw?complete=1&always=1&target='+key,full.read_bytes())
    assert replay_all['always_on'] and drawn_all['always_on']
    assert [a['id'] for a in drawn_all['adapter']['top']]==[a['id'] for a in replay_all['adapter']['top']]
    assert drawn_all['adapter']['top']!=drawn_all['base']['top']
    arbitrary=request('/api/replay?id='+key+'&tick=3765')
    assert arbitrary['ink'] and not arbitrary['anchor']
    same_pixels=request('/api/draw?target='+key,base64.b64decode(arbitrary['raster']))
    for side in ('base','adapter'):
        assert same_pixels[side]==arbitrary[side]
    free=request('/api/draw',path.read_bytes())
    assert free['base']['rank'] is None and len(free['adapter']['top'])==5
    done=request('/api/draw?complete=1',path.read_bytes())
    assert done['base']['top']==done['adapter']['top']
    buf=io.BytesIO()
    Image.new('L',(256,256),255).save(buf,format='PNG')
    assert not request('/api/draw',buf.getvalue())['ink']
    assert not request('/api/replay?id='+key+'&tick=0')['ink']
    print(f"PASS {session['device']}: {len(case_ids)} official stroke sequences, {checked} replay checkpoints with matching pixels, ranks and top-1 IDs, top-5 sizes and ordering; all {len(ids)} gallery IDs, thumbnail and non-curated reference ranks; live inference, full, free and empty canvas.")

if __name__=='__main__':
    main()
