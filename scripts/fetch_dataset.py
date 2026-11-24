"""Download the official FS-COCO archive using resumable, verified byte ranges."""
import concurrent.futures
import gzip
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1] / 'data'
URL = 'https://cvssp.org/data/fscoco/fscoco.tar.gz'
SIZE = 2378884718
BLOCK = 16 * 1024**2
PARTS = ROOT / '.ranges'
PREFIX = ROOT / 'fscoco.tar.gz.part'
FINAL = ROOT / 'fscoco.tar.gz'
ETAG = '"8dcae66e-5e4b2b8982ff4"'

def fetch(bounds):
    start, end = bounds
    dest = PARTS / f'{start}-{end}'
    if dest.exists() and dest.stat().st_size == end-start+1:
        return dest
    partial = dest.with_suffix('.progress')
    written = partial.stat().st_size if partial.exists() else 0
    failures = 0
    while start+written<=end:
        first = start+written
        last = min(end, first + 4*1024**2 - 1)
        with tempfile.NamedTemporaryFile(dir=PARTS) as hdr, tempfile.NamedTemporaryFile(dir=PARTS) as piece:
            result = subprocess.run(['curl', '-fsSL', '--connect-timeout', '10', '--max-time', '40',
                                     '-H', f'If-Range: {ETAG}', '--range', f'{first}-{last}',
                                     '--dump-header', hdr.name, '-o', piece.name, URL])
            headers = pathlib.Path(hdr.name).read_text().lower()
            size = pathlib.Path(piece.name).stat().st_size
            if f'content-range: bytes {first}-{last}/{SIZE}' not in headers or not 0 < size <= last-first+1:
                failures += 1
                if failures >= 20:
                    raise IOError(f'Range repeatedly failed: {first}-{last}; curl {result.returncode}')
                continue
            with partial.open('ab') as out:
                shutil.copyfileobj(piece,out)
            written += size
            failures = 0
    partial.replace(dest)
    print(f'validated {start}-{end}', flush=True)
    return dest

def main():
    if FINAL.exists():
        if FINAL.stat().st_size != SIZE:
            raise ValueError('Existing file has an unexpected size')
        return
    PARTS.mkdir(exist_ok=True, parents=True)
    # Freeze the prefix length once so a retry reuses the identical range boundaries.
    offset = PARTS / 'prefix_size'
    if not offset.exists():
        offset.write_text(str(PREFIX.stat().st_size if PREFIX.exists() else 0))
    n = int(offset.read_text())
    bounds = [(k, min(k + BLOCK, SIZE)-1) for k in range(n, SIZE, BLOCK)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        parts = list(pool.map(fetch, bounds))
    assembled = ROOT / 'fscoco.assembling'
    with assembled.open('wb') as out:
        if n:
            with PREFIX.open('rb') as src:
                left = n
                while left:
                    block = src.read(min(left, 1024**2))
                    if not block:
                        raise EOFError('Partial prefix missing')
                    out.write(block)
                    left -= len(block)
        for path in parts:
            with path.open('rb') as src:
                shutil.copyfileobj(src, out)
    if assembled.stat().st_size != SIZE:
        raise ValueError('Size mismatch')
    with gzip.open(assembled) as stream:
        while stream.read(8*1024**2):
            pass
    assembled.rename(FINAL)
    for path in parts:
        path.unlink()
    offset.unlink()
    for leftover in PARTS.glob('*.tmp'):
        leftover.unlink()
    if not any(PARTS.iterdir()):
        PARTS.rmdir()
    if PREFIX.exists():
        PREFIX.unlink()
    print(f'Validated gzip footer and {SIZE} bytes', flush=True)

if __name__ == '__main__':
    main()
