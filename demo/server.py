"""Offline SketchScape retrieval workbench for the FS-COCO Normal protocol."""
import argparse
import base64
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import random
import sys
from threading import Lock
from time import perf_counter
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen
import webbrowser

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cache_prefix_features import CACHE, DATA, WEIGHTS
from train_prefix_adapter import SketchAdapter
from data import _render_sketch
from model import SBIRModel
from utils import compute_distances

STATIC = Path(__file__).parent / 'static'
CASES = {
    '000000197222': '草地上的斑马',
    '000000542257': '弯道上的列车',
    '000000458175': '跃过障碍的马',
    '000000169845': '城市钟楼',
    '000000253125': '公园里放风筝',
    '000000343951': '岸边的小船',
    '000000006819': '跑道上的飞机',
    '000000105200': '草坪上的球赛',
    '000000038196': '小路上的羊群',
    '000000052322': '树枝上的鸟',
    '000000243472': '路口的骑行者',
    '000000419560': '相对而立的长颈鹿',
}
ANCHORS = (25, 50, 75, 100)
APP_ID = 'sketchscape-normal-workbench-v2'
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/picker.js': ('picker.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8'), '/favicon.svg': ('favicon.svg','image/svg+xml')}


class Workbench:
    def __init__(self, device):
        self.device = device
        torch.set_num_threads(3)
        cache = CACHE / 'normal'
        meta = json.loads((cache / 'manifest.json').read_text())
        if meta['weight_sha256'] != WEIGHTS['normal'] or meta['protocol'] != 'normal':
            raise ValueError('照片缓存与 Normal 编码器不匹配')
        test = set((DATA / 'val_normal.txt').read_text().splitlines())
        selected = [(i, folder, key) for i, (folder, key) in enumerate(meta['ids']) if key in test]
        self.ids = [key for _, _, key in selected]
        self.folders = {key: folder for _, folder, key in selected}
        if len(self.ids) != 3000 or set(self.ids) != test:
            raise ValueError('Normal 测试图库不完整')
        if not CASES.keys() <= test:
            raise ValueError('回放案例缺少配对照片')
        self.catalog = [
            {'id': key, 'caption': (DATA/'text'/folder/(key+'.txt')).read_text().strip()}
            for key, folder in self.folders.items()
        ]
        random.Random(23).shuffle(self.catalog)
        photo_blocks = [np.load(cache / f'photo_{i:05d}.npy', mmap_mode='r') for i in range(0,10000,1000)]
        self.gallery = np.stack([photo_blocks[i//1000][i%1000] for i, _, _ in selected])
        self.anchors = {}
        for i, _, key in selected:
            if key in CASES:
                for pct in ANCHORS:
                    block = np.load(cache / f'{pct}_{i//1000*1000:05d}.npy', mmap_mode='r')
                    self.anchors[key, pct] = torch.from_numpy(block[i%1000].copy())[None]
        self.encoder = SBIRModel.load_module(str(ROOT/'models/fscoco_normal.pth'), strict=True, cuda=False)
        self.encoder.eval().requires_grad_(False).to(device)
        self.transform = self.encoder.create_sketch_transforms()
        ckpt = torch.load(ROOT/'models/adapter_normal.pth', weights_only=True, map_location='cpu')
        if ckpt['base_sha256'] != WEIGHTS['normal'] or ckpt['protocol'] != 'normal':
            raise ValueError('适配器与官方编码器不匹配')
        self.adapter = SketchAdapter().eval().requires_grad_(False)
        self.adapter.load_state_dict(ckpt['state_dict'], strict=True)
        self.lock = Lock()

    @lru_cache(maxsize=384)
    def thumbnail(self, key):
        with Image.open(DATA/'images'/self.folders[key]/(key+'.jpg')) as img:
            img.thumbnail((320, 240))
            buf = io.BytesIO()
            img.convert('RGB').save(buf, format='JPEG', quality=82)
            return buf.getvalue()

    @lru_cache(maxsize=len(CASES))
    def sample(self,key):
        if key not in CASES:
            raise ValueError('未知回放案例')
        pts = json.loads((DATA/'raw_data'/self.folders[key]/(key+'.json')).read_bytes())
        times = np.maximum.accumulate([p['timestamp'] for p in pts])
        return pts, times-times[0]

    def sequence(self,key):
        pts,times = self.sample(key)
        return {'id':key,'duration':int(times[-1]),'points':[
            [int(float(p['coordinates'][0])*256),int(float(p['coordinates'][1])*256),int(t),
             int(p['pen_state']==[1,0,0])] for p,t in zip(pts,times)]}

    @torch.inference_mode()
    def encode(self,image):
        return self.encoder(self.transform(image).unsqueeze(0).to(self.device)).cpu()

    @torch.inference_mode()
    def retrieve(self,query,target,bypass):
        result = {}
        for name,q in [('base',query), ('adapter',query if bypass else self.adapter(query))]:
            dist = compute_distances(q.numpy(),self.gallery)[0]
            order = dist.argsort()
            result[name] = {
                'rank': next((n for n,j in enumerate(order,1) if self.ids[j]==target),None) if target else None,
                'top': [{'id':self.ids[j], 'score':float(1-dist[j]/2)} for j in order[:5]]}
        return result

    @lru_cache(maxsize=150)
    def replay(self,key,tick,always_on=False):
        pts,times=self.sample(key)
        n=int(np.searchsorted(times*10000, tick*times[-1],side='right'))
        img=Image.fromarray(_render_sketch(pts[:n]))
        buf=io.BytesIO()
        img.save(buf,format='PNG')
        mask_image=base64.b64encode(buf.getvalue()).decode('ascii')
        start=perf_counter()
        result={}
        if tick and img.getextrema()[0]<255:
            with self.lock:
                q=self.anchors.get((key,tick//100)) if tick%100==0 else None
                if q is None:
                    q=self.encode(img)
                result=self.retrieve(q,key,tick==10000 and not always_on)
        return {'tick':tick,'ink':bool(result),'complete':tick==10000,'raster':mask_image,
                'always_on':always_on,'anchor':tick in (2500,5000,7500,10000),
                'ms':round(1000*(perf_counter()-start)),**result}

    def draw(self,raw,target,complete,always_on=False):
        if target and target not in self.folders:
            raise ValueError('未知参考图')
        try:
            with Image.open(io.BytesIO(raw),formats=['PNG']) as im:
                if im.size!=(256,256):
                    raise ValueError('画布大小应为 256 × 256')
                rgba=im.convert('RGBA')
                image=Image.alpha_composite(Image.new('RGBA',im.size,'white'),rgba).convert('L')
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError('无法读取画布 PNG') from exc
        if image.getextrema()[0]==255:
            return {'ink':False}
        start=perf_counter()
        with self.lock:
            result=self.retrieve(self.encode(image),target,complete and not always_on)
        return {'ink':True,'complete':complete,'always_on':always_on,'ms':round((perf_counter()-start)*1000),**result}


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self,body,kind,cache='no-store',status=200):
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control',cache)
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self,obj,status=200):
        self.send_bytes(json.dumps(obj,ensure_ascii=False).encode(),'application/json; charset=utf-8',status=status)

    def handle_request(self,post):
        url=urlsplit(self.path)
        params=parse_qs(url.query)
        def param(key,default=''):
            return params.get(key,[default])[0]
        app=self.server.app
        if post:
            if url.path!='/api/draw':
                return self.send_error(HTTPStatus.NOT_FOUND)
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=400000 or self.headers.get_content_type()!='image/png':
                raise ValueError('请求需要有效的画布 PNG')
            return self.send_json(app.draw(self.rfile.read(size),param('target'),param('complete')=='1',param('always')=='1'))
        if url.path in ASSETS:
            name,kind=ASSETS[url.path]
            return self.send_bytes((STATIC/name).read_bytes(),kind)
        if url.path=='/api/session':
            return self.send_json({'app':APP_ID,'device':app.device.upper(),'gallery':len(app.ids),
                                   'cases':[{'id':key,'name':v} for key,v in CASES.items()]})
        if url.path=='/api/gallery':
            return self.send_json(app.catalog)
        if url.path=='/api/case':
            return self.send_json(app.sequence(param('id')))
        if url.path=='/api/replay':
            tick=int(param('tick'))
            if not 0<=tick<=10000:
                raise ValueError('回放位置超出范围')
            return self.send_json(app.replay(param('id'),tick,param('always')=='1'))
        if url.path.startswith('/thumb/'):
            key=url.path.removeprefix('/thumb/')
            if key in app.folders:
                return self.send_bytes(app.thumbnail(key),'image/jpeg','private, max-age=86400')
        if url.path.startswith('/photo/'):
            key=url.path.removeprefix('/photo/')
            if key in app.folders:
                return self.send_bytes((DATA/'images'/app.folders[key]/(key+'.jpg')).read_bytes(),
                                       'image/jpeg','private, max-age=86400')
        self.send_error(HTTPStatus.NOT_FOUND)

    def dispatch(self,post):
        try:
            self.handle_request(post)
        except (ConnectionResetError,BrokenPipeError):
            pass
        except ValueError as exc:
            self.send_json({'error':str(exc)},400)
        except Exception:
            import traceback
            traceback.print_exc()
            self.send_json({'error':'检索失败，请检查终端日志后重试。'},500)

    def do_GET(self):
        self.dispatch(False)

    def do_POST(self):
        self.dispatch(True)

    def log_message(self,fmt,*args):
        if len(args)>1 and str(args[1]).startswith(('4','5')):
            super().log_message(fmt,*args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',default=8765,type=int)
    parser.add_argument('--device',choices=['cpu','mps'],default=None)
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    url=f'http://127.0.0.1:{args.port}'
    try:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    except OSError:
        if args.open:
            try:
                with urlopen(url+'/api/session',timeout=3) as active:
                    if json.load(active).get('app')==APP_ID:
                        webbrowser.open(url)
                        return
            except (OSError,ValueError):
                pass
        raise
    with server:
        device=args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        server.app=Workbench(device)
        print(f'SketchScape 就绪  {url}  ·  {device.upper()}  ·  3,000 张照片',flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__=='__main__':
    main()
