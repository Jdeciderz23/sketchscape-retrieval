"""Safely extract FS-COCO, validate all samples and render missing complete sketches."""
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'SketchScape'))
from data import _render_sketch, load_dataset

def extract():
    archive=ROOT/'data/fscoco.tar.gz'
    dest=ROOT/'data/fscoco'
    report=ROOT/'outputs/metrics/archive.json'
    if dest.exists():
        if not report.is_file():
            raise RuntimeError('Existing extracted dataset lacks archive evidence')
        return
    stage=ROOT/'data/.extracting'
    stage.mkdir(exist_ok=True)
    with archive.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    seen=set()
    raster_files=Counter()
    expanded=0
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            name=Path(member.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0]!='fscoco':
                raise ValueError(member.name)
            if member.name in seen or not (member.isfile() or member.isdir()):
                raise ValueError(f'Duplicate or unsupported member: {member.name}')
            seen.add(member.name)
            if member.isfile() and name.parts[:2]==('fscoco','raster_sketches'):
                raster_files[name.suffix.lower()]+=1
            expanded+=member.size
            tf.extract(member,path=stage,filter='data')
    record=dict(archive=archive.name,sha256=digest,compressed=archive.stat().st_size,
                members=len(seen),expanded_file_bytes=expanded,raster_files=dict(raster_files))
    report.write_text(json.dumps(record,indent=2)+'\n')
    (stage/'fscoco').rename(dest)
    stage.rmdir()
    print(record,flush=True)

def main():
    extract()
    root=ROOT/'data/fscoco'
    def keys(d,suffix):
        return {(p.parent.name,p.stem) for p in (root/d).glob('*/*') if p.suffix.lower()==suffix}
    photos=keys('images','.jpg')
    raws=keys('raw_data','.json')
    rasters=keys('raster_sketches','.png')
    jpeg_sketches=keys('raster_sketches','.jpg')
    assert jpeg_sketches==photos
    assert photos==raws and len(photos)>0 and rasters<=raws
    assert {int(u) for u,_ in photos}==set(range(1,101))
    duplicated=[k for k,n in Counter(int(k) for _,k in photos).items() if n>1]
    for u,k in sorted(photos):
        strokes=json.loads((root/'raw_data'/u/f'{k}.json').read_text())
        assert strokes and all(len(p['coordinates'])==2 and len(p['pen_state'])==3 and 'timestamp' in p for p in strokes)
        png=root/'raster_sketches'/u/f'{k}.png'
        if (u,k) not in rasters:
            png.parent.mkdir(exist_ok=True,parents=True)
            Image.fromarray(_render_sketch(strokes)).save(png)
        for path in (root/'images'/u/f'{k}.jpg',root/'raster_sketches'/u/f'{k}.jpg',png):
            with Image.open(path) as im:
                im.load()
    archive=json.loads((ROOT/'outputs/metrics/archive.json').read_text())
    results=dict(pairs=len(photos),numbered_folders=100,unique_numeric_ids=len({int(k) for _,k in photos}),
                 duplicate_ids=duplicated,decoded_photos=len(photos),decoded_rasters=len(photos),decoded_source_jpg_sketches=len(jpeg_sketches),
                 validated_json=len(raws),raster_files_in_archive=archive['raster_files'],official_png_count=len(photos),
                 png_generated_this_validation=len(photos-rasters),splits={})
    for protocol,file in [('normal','val_normal.txt'),('unseen','val_unseen_user.txt')]:
        ids=[int(v) for v in (root/file).read_text().splitlines()]
        collection=load_dataset(str(root),mode=protocol)
        train,test,_=collection.split()
        train_ids={int(k) for _,k in train.file_names()}
        test_keys=test.file_names()
        assert len(ids)==len(set(ids)) and sorted(ids)==sorted(int(k) for _,k in test_keys)
        assert not train_ids.intersection(ids) and len(train)+len(test)==len(photos)
        results['splits'][protocol]=dict(file=file,train=len(train),test=len(test),test_folders=len({u for u,_ in test_keys}),
                                         folder_overlap=len({u for u,_ in train.file_names()} & {u for u,_ in test_keys}))
        sizes=Counter(u for u,_ in test_keys)
        results['splits'][protocol].update(min_test_per_folder=min(sizes.values()),max_test_per_folder=max(sizes.values()))
    (ROOT/'outputs/metrics/dataset.json').write_text(json.dumps(results,indent=2)+'\n')
    print(results,flush=True)

if __name__=='__main__':
    main()
