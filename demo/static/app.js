import { loadLibrary, openPicker } from './picker.js';
const $ = id => document.getElementById(id);
const paths = {
    film: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4m-4 6h4m10-6h4m-4 6h4"/>',
    pen: '<path d="m4 20 1-5L16 4a2 2 0 0 1 4 4L9 19l-5 1zM14 6l4 4"/>',
    collection: '<rect x="6" y="5" width="14" height="13" rx="2"/><path d="M3 9v11h12M8 15l4-4 3 2 3-4"/>',
    dice: '<rect x="3" y="3" width="18" height="18" rx="4"/><g fill="currentColor" stroke="none"><circle cx="8" cy="8" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="16" cy="16" r="1.4"/></g>',
    play: '<path fill="currentColor" stroke="none" d="m7 4 13 8-13 8z"/>',
    pause: '<path fill="currentColor" stroke="none" d="M6 4h4v16H6zm8 0h4v16h-4z"/>',
    reset: '<path d="M3 11a9 9 0 1 1 3 8M3 5v6h6"/>', undo: '<path d="m9 5-6 6 6 6M4 11h10c4 0 7 2 7 6"/>',
    trash: '<path d="M3 6h18M9 6V4h6v2M6 6l1 14h10l1-14M10 10v6m4-6v6"/>',
    check: '<path d="m4 12 5 5L20 6"/>',
    photo: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="1.5"/><path d="m4 18 6-6 3 3 3-5 4 6"/>'
};
function svg(name) { return '<svg aria-hidden="true" viewBox="0 0 24 24">' + paths[name] + '</svg>'; }
function setIcons(el) { el.querySelectorAll('i[data-icon]').forEach(i => i.innerHTML = svg(i.dataset.icon)); }
setIcons(document);
const ctx = $('sketch').getContext('2d');
let mode = 'replay', cases = [], key = '', seq = null, tick = 2500, playing = false, anim = 0, clock = 0, lastRequest = 0;
let strokes = [], pointer = null, finished = false, revision = 0, busy = false, pending = null, caseRequest = 0;
let liveTarget = null; // First visit draws a random reference. An empty string means free drawing.
const photoURL = id => '/photo/' + id;
const percent = t => (Number.isInteger(t / 100) ? t / 100 : (t / 100).toFixed(1)) + '%';
async function api(url, options) {
    const r = await fetch(url, options);
    if (!r.ok)
        throw new Error((await r.json().catch(() => ({}))).error || '无法连接本地检索服务');
    return r.json();
}
function message(text) { $('alert').textContent = text; $('alert').hidden = !text; }
function invalidate() { revision++; pending = null; }
function clearPaper() { ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 256, 256); }
function drawReplay() {
    clearPaper();
    if (!seq || !tick)
        return;
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    let prev = null;
    for (const pt of seq.points) {
        if (pt[2] * 10000 > tick * seq.duration)
            break;
        if (prev && prev[0] >= 0 && prev[0] < 256 && prev[1] >= 0 && prev[1] < 256 && pt[0] >= 0 && pt[0] < 256 && pt[1] >= 0 && pt[1] < 256) {
            ctx.moveTo(prev[0], prev[1]);
            ctx.lineTo(pt[0], pt[1]);
        }
        prev = prev && pt[3] ? null : pt;
    }
    ctx.stroke();
}
function drawLive() {
    updateCompleteOption();
    clearPaper();
    ctx.strokeStyle = '#000';
    ctx.fillStyle = '#000';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const stroke of strokes) {
        ctx.beginPath();
        ctx.moveTo(...stroke[0]);
        for (const p of stroke.slice(1))
            ctx.lineTo(...p);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(...stroke[0], 1.5, 0, Math.PI * 2);
        ctx.fill();
    }
    $('paper-empty').hidden = strokes.length > 0;
    $('undo').disabled = $('clear').disabled = $('finish').disabled = !strokes.length;
    $('finish').querySelector('span').textContent = finished ? '继续绘制' : '画完了';
}
function updateCompleteOption() {
    $('complete-option').hidden = !(mode === 'replay' ? tick === 10000 : finished);
}
function timeUI() {
    updateCompleteOption();
    $('pct').textContent = percent(tick);
    $('progress').value = tick;
    $('progress').style.setProperty('--fill', tick / 100 + '%');
    document.querySelectorAll('[data-tick]').forEach(b => b.classList.toggle('current', +b.dataset.tick === tick));
}
function pause() { playing = false; cancelAnimationFrame(anim); $('play').querySelector('i').innerHTML = svg('play'); $('play').querySelector('span').textContent = '播放'; }
function viewPhoto(id, title) {
    if (playing) {
        pause();
        enqueueReplay();
    }
    $('viewer-title').textContent = title;
    $('large-photo').src = photoURL(id);
    $('viewer').showModal();
}
$('close-viewer').onclick = () => $('viewer').close();
$('viewer').onclick = e => { if (e.target === $('viewer'))
    $('viewer').close(); };
function currentTarget() {
    return mode === 'replay' ? key : liveTarget || '';
}
function showTarget() {
    const live = mode === 'live', target = currentTarget();
    $('sample').hidden = live;
    $('target-actions').hidden = !live;
    $('free-draw').hidden = !live || !target;
    $('target-image').hidden = !target;
    $('target-label').textContent = live ? (liveTarget === '' ? '自由手绘' : '参考照片') : '目标照片';
    $('target-name').hidden = live;
    if (!live) $('target-name').textContent = cases.find(c => c.id === key)?.name || '';
    if (target) $('target-photo').src = photoURL(target);
}
$('target-image').onclick = () => viewPhoto(currentTarget(), mode === 'replay' ? '目标照片' : '参考照片');

function setReference(id) {
    liveTarget = id;
    invalidate();
    message('');
    showTarget();
    renderResults(null, id);
    if (pointer === null) enqueueLive();
}
async function randomReference(initial = false) {
    try {
        const photos = await loadLibrary();
        if (mode !== 'live' || (initial && liveTarget !== null)) return;
        const previous = photos.findIndex(photo => photo.id === liveTarget);
        const count = photos.length - (previous >= 0 ? 1 : 0);
        let index = Math.floor(Math.random() * count);
        if (previous >= 0 && index >= previous) index++;
        setReference(photos[index].id);
    } catch (error) {
        if (mode === 'live') message(error.message);
    }
}
$('choose-photo').onclick = async () => {
    try {
        const photos = await loadLibrary();
        if (mode === 'live' && !$('picker').open) openPicker(photos, liveTarget, true, setReference);
    } catch (error) {
        message(error.message);
    }
};
$('roll-photo').onclick = () => randomReference();
$('free-draw').onclick = () => setReference('');
$('sample').onclick = () => {
    if (playing) { pause(); invalidate(); enqueueReplay(); }
    openPicker(cases, key, false, selectCase);
};

const columns = {base: '原始检索', adapter: '优化检索'};
for (const [side, title] of Object.entries(columns)) {
    $('model-' + side).innerHTML = '<h3>' + title + '</h3><div class="rankline"><span>目标排名</span><strong>—</strong></div><div class="list"></div>';
}
function renderResults(result, target) {
    for (const [side, title] of Object.entries(columns)) {
        const card = $('model-' + side), list = card.querySelector('.list'), data = result?.[side];
        card.querySelector('.rankline').hidden = !target;
        card.querySelector('.rankline strong').textContent = data?.rank ? '第 ' + data.rank + ' 名' : '—';
        list.replaceChildren();
        if (!data) {
            list.className = 'list empty-result';
            list.innerHTML = '<i data-icon="photo"></i><span>' + (mode === 'live' ? '画下一笔，查看结果' : '等待草图笔画') + '</span>';
            setIcons(list);
            continue;
        }
        list.className = 'list photos';
        data.top.forEach((hit, i) => {
            const btn = document.createElement('button'), match = hit.id === target;
            const label = title + '第 ' + (i + 1) + ' 名' + (match ? '，目标照片' : '');
            btn.type = 'button';
            btn.className = 'photo' + (i === 0 ? ' hero' : '') + (match ? ' match' : '');
            btn.setAttribute('aria-label', label);
            btn.innerHTML = '<span class="frame"><img alt="" src="' + photoURL(hit.id) + '"><span class="position">' + (i + 1) + '</span>' + (match ? '<span class="matchmark">目标</span>' : '') + '</span>';
            btn.onclick = () => viewPhoto(hit.id, label);
            list.append(btn);
        });
    }
}
function showQueryResult(out, job) {
    renderResults(out, job.target);
    $('result-status').textContent = job.mode === 'replay' && playing && out.ink ? '显示 ' + percent(out.tick) + ' 时的结果' : '';
    if (job.mode === 'replay' && !playing && out.tick === tick && out.raster) {
        const img = new Image();
        img.onload = () => { if (job.rev === revision && mode === 'replay' && !playing && tick === out.tick)
            ctx.drawImage(img, 0, 0, 256, 256); };
        img.src = 'data:image/png;base64,' + out.raster;
    }
}
async function drain() {
    if (busy || !pending)
        return;
    busy = true;
    const job = pending;
    pending = null;
    try {
        let out;
        if (job.mode === 'replay') {
            out = await api('/api/replay?id=' + job.target + '&tick=' + job.tick + '&always=' + Number(job.always));
        }
        else {
            const blob = await new Promise(resolve => $('sketch').toBlob(resolve, 'image/png'));
            if (job.rev !== revision)
                return;
            out = await api('/api/draw?target=' + job.target + '&complete=' + Number(job.complete) + '&always=' + Number(job.always), { method: 'POST', body: blob, headers: { 'Content-Type': 'image/png' } });
        }
        if (job.rev === revision) {
            message('');
            showQueryResult(out, job);
        }
    }
    catch (e) {
        if (job.rev === revision) {
            message(e.message);
            $('result-status').textContent = '检索失败';
        }
    }
    finally {
        busy = false;
        drain();
    }
}
function enqueueReplay() {
    if (mode !== 'replay' || !seq)
        return;
    pending = { mode, rev: revision, tick, target: key, always: tick === 10000 && $('always').checked };
    drain();
}
function enqueueLive() {
    if (mode !== 'live')
        return;
    if (!strokes.length) {
        renderResults(null, currentTarget());
        $('result-status').textContent = '';
        return;
    }
    $('result-status').textContent = '正在检索…';
    pending = { mode, rev: revision, target: currentTarget(), complete: finished, always: finished && $('always').checked };
    drain();
}
function seek(t) { pause(); invalidate(); tick = t; timeUI(); drawReplay(); $('result-status').textContent = '正在检索…'; enqueueReplay(); }
$('progress').oninput = e => seek(+e.target.value);
document.querySelectorAll('[data-tick]').forEach(b => b.onclick = () => seek(+b.dataset.tick));
function playback(now) {
    if (!playing || mode !== 'replay')
        return;
    const delta = (now - clock) * Number($('speed').value);
    clock = now;
    floatTick = Math.min(10000, floatTick + delta * 10000 / Math.max(1, seq.duration));
    tick = Math.round(floatTick);
    timeUI();
    drawReplay();
    if (tick === 10000) {
        pause();
        invalidate();
        enqueueReplay();
        return;
    }
    if (now - lastRequest > 450) {
        lastRequest = now;
        enqueueReplay();
    }
    anim = requestAnimationFrame(playback);
}
let floatTick = 0;
function start() { if (!seq)
    return; if (tick === 10000)
    tick = 0; invalidate(); floatTick = tick; clock = performance.now(); lastRequest = 0; playing = true; $('play').querySelector('i').innerHTML = svg('pause'); $('play').querySelector('span').textContent = '暂停'; anim = requestAnimationFrame(playback); }
$('play').onclick = () => { if (playing) {
    pause();
    invalidate();
    enqueueReplay();
}
else
    start(); };
$('reset').onclick = () => { seek(0); start(); };
async function selectCase(id) {
    const task = ++caseRequest;
    pause();
    invalidate();
    key = id;
    seq = null;
    showTarget();
    message('');
    if (mode === 'replay') {
        renderResults(null, key);
        clearPaper();
        $('result-status').textContent = '正在载入笔画…';
    }
    try {
        const data = await api('/api/case?id=' + id);
        if (task !== caseRequest)
            return;
        seq = data;
        if (mode === 'replay')
            seek(2500);
    }
    catch (e) {
        if (task === caseRequest)
            message(e.message);
    }
}
$('always').onchange = () => { invalidate(); if (mode === 'replay')
    enqueueReplay();
else
    enqueueLive(); };
function setMode(next) {
    if (mode === next)
        return;
    pause();
    invalidate();
    mode = next;
    const live = mode === 'live';
    $('mode-replay').setAttribute('aria-pressed', String(!live));
    $('mode-live').setAttribute('aria-pressed', String(live));
    $('pad').classList.toggle('drawing', live);
    $('live-controls').hidden = !live;
    $('replay-controls').hidden = live;
    document.querySelector('.input-panel').classList.toggle('live', live);
    $('input-title').textContent = live ? '手绘画布' : '回放画布';
    $('paper-empty').hidden = true;
    message('');
    showTarget();
    renderResults(null, currentTarget());
    if (live) {
        drawLive();
        enqueueLive();
        if (liveTarget === null) randomReference(true);
    }
    else {
        timeUI();
        drawReplay();
        enqueueReplay();
    }
}
$('mode-replay').onclick = () => setMode('replay');
$('mode-live').onclick = () => setMode('live');
function coord(e) {
    const b = $('sketch').getBoundingClientRect();
    return [Math.max(0, Math.min(255, 256 * (e.clientX - b.left) / b.width)), Math.max(0, Math.min(255, 256 * (e.clientY - b.top) / b.height))];
}
const canvas = $('sketch');
canvas.onpointerdown = e => {
    if (mode !== 'live' || pointer !== null || e.button !== 0)
        return;
    e.preventDefault();
    invalidate();
    pointer = e.pointerId;
    canvas.setPointerCapture(pointer);
    finished = false;
    strokes.push([coord(e)]);
    drawLive();
    $('result-status').textContent = '绘制中…';
};
canvas.onpointermove = e => {
    if (pointer !== e.pointerId || mode !== 'live')
        return;
    const batch = e.getCoalescedEvents?.() || [];
    for (const point of batch.length ? batch : [e])
        strokes.at(-1).push(coord(point));
    drawLive();
};
function endStroke(e) {
    if (pointer !== e.pointerId)
        return;
    pointer = null;
    if (canvas.hasPointerCapture(e.pointerId))
        canvas.releasePointerCapture(e.pointerId);
    drawLive();
    enqueueLive();
}
canvas.onpointerup = endStroke;
canvas.onpointercancel = endStroke;
canvas.onlostpointercapture = endStroke;
$('undo').onclick = () => { invalidate(); finished = false; strokes.pop(); drawLive(); enqueueLive(); };
$('clear').onclick = () => { invalidate(); finished = false; strokes = []; drawLive(); enqueueLive(); };
$('finish').onclick = () => { invalidate(); finished = !finished; drawLive(); enqueueLive(); };
async function boot() {
    try {
        const s = await api('/api/session');
        cases = s.cases;
        $('sample').disabled = false;
        $('mode-live').disabled = false;
        await selectCase(cases[0].id);
    }
    catch (e) {
        message('连接失败，刷新页面后重试。' + e.message);
        $('result-status').textContent = '服务不可用';
    }
}
boot();
