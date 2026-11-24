"""Construct time-prefix sketches with the official full-sketch rasterizer."""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'SketchScape'))
from data import _render_sketch

DATA = ROOT / 'data/fscoco'
OUT = ROOT / 'data/time_prefix_v1'
FRACTIONS = (25, 50, 75)

def render(path):
    points = json.loads(path.read_bytes())
    times = np.array([p['timestamp'] for p in points], dtype=np.int64)
    corrected = np.maximum.accumulate(times)
    max_regression = int((corrected-times).max())
    times = corrected
    if times[-1] <= times[0]:
        raise ValueError(f'Invalid timestamp sequence: {path}')
    elapsed = times-times[0]
    user, sample = path.parent.name, path.stem
    full = np.asarray(Image.open(DATA/'raster_sketches'/user/(sample+'.png')).convert('L'))
    rebuilt = _render_sketch(points)
    if not np.array_equal(rebuilt,full):
        raise ValueError(f'Full rendering differs from baseline: {path}')
    previous = np.full_like(full,255)
    result = [user,sample,len(times),int(elapsed[-1]),max_regression]
    for fraction in FRACTIONS:
        n = np.searchsorted(100*elapsed,fraction*elapsed[-1],side='right')
        canvas = _render_sketch(points[:n])
        if (canvas>previous).any() or (full>canvas).any():
            raise ValueError(f'Prefix pixels inconsistent: {path}')
        dest = OUT/str(fraction)/user/(sample+'.png')
        Image.fromarray(canvas).save(dest)
        result.extend([int(n),int((canvas<255).sum())])
        previous = canvas
    return result

def main():
    paths = sorted((DATA/'raw_data').glob('*/*.json'),key=lambda p:(int(p.parent.name),p.stem))
    assert len(paths)==10000
    for user in range(1,101):
        for f in FRACTIONS:
            (OUT/str(f)/str(user)).mkdir(parents=True,exist_ok=True)
    manifest = OUT/'manifest.csv'
    tmp = manifest.with_suffix('.tmp')
    start = perf_counter()
    with ProcessPoolExecutor(4) as pool,tmp.open('w') as out:
        writer=csv.writer(out)
        writer.writerow(['folder','id','points','elapsed_timestamp_units','max_clock_correction','points25','ink25','points50','ink50','points75','ink75'])
        for i,row in enumerate(pool.map(render,paths,chunksize=16),1):
            writer.writerow(row)
            if i%1000==0: print('rendered',i,round(perf_counter()-start,1),flush=True)
    os.replace(tmp,manifest)

if __name__=='__main__':
    main()
