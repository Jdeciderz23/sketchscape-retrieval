// FS-COCO captions are in English. These aliases cover common subjects for Chinese searches.
const topics = [
    [/zebra/, '斑马 动物'], [/giraffe/, '长颈鹿 动物'], [/elephant/, '大象 动物'],
    [/sheep|lamb/, '绵羊 羊群 动物'], [/cow|cattle/, '奶牛 黄牛 牛群 动物'],
    [/horse|pony/, '马 马术 骑马 动物'], [/bear/, '熊 动物'], [/cat\b|cats\b|kitten/, '小猫 猫咪 动物'],
    [/dog|puppy/, '狗 小狗 动物'], [/bird|duck|swan/, '鸟 禽 动物'], [/duck/, '鸭子'],
    [/train|railway|railroad|locomotive/, '火车 列车 铁路 交通'], [/bus\b|buses/, '巴士 公交车 客车 交通'],
    [/plane|aircraft|flight|flying jet/, '飞机 航空 交通'], [/boat|ship|ferry/, '船 轮船 小舟 交通'],
    [/bicycl|bike|cyclist/, '自行车 单车 骑行 交通'], [/motorcycle|motorbike/, '摩托车 交通'],
    [/truck/, '卡车 货车 交通'], [/\bcar\b|\bcars\b|taxi/, '汽车 小轿车 交通'],
    [/traffic|signal|crossing/, '路口 红绿灯 交通'], [/sign|street name/, '路牌 标志'],
    [/surfer|surfing|surfboard/, '冲浪 运动'], [/skateboard/, '滑板 运动'], [/skier|skiing|\bskis\b/, '滑雪 运动'],
    [/football|soccer/, '足球 球赛 运动'], [/baseball|batter/, '棒球 运动'], [/tennis/, '网球 运动'],
    [/kite/, '风筝 运动'], [/frisbee|disc/, '飞盘 运动'], [/clock|tower/, '钟楼 塔 建筑'],
    [/church|building|palace/, '建筑'], [/tree|forest/, '树 树林'], [/beach|seashore/, '海滩 海边'],
    [/river|lake|pond/, '水边 湖 河'], [/bench/, '长椅'], [/umbrella/, '雨伞'], [/park\b/, '公园']
];
const $ = id => document.getElementById(id);
const dialog = $('picker');
const pageSize = 20;
let libraryPromise, entries = [], matches = [], page = 0, selected, isLibrary, choose;

export function loadLibrary() {
    if (!libraryPromise) {
        libraryPromise = fetch('/api/gallery').then(async response => {
            if (!response.ok) throw new Error('图库载入失败，请重试。');
            const photos = await response.json();
            return photos.map(photo => {
                const caption = photo.caption.toLowerCase();
                const tags = topics.filter(([pattern]) => pattern.test(caption)).map(([, words]) => words);
                return { ...photo, search: [caption, photo.id, ...tags].join(' ') };
            });
        }).catch(error => {
            libraryPromise = null;
            throw error;
        });
    }
    return libraryPromise;
}

function renderPage() {
    const choices = $('choices');
    const pages = Math.ceil(matches.length / pageSize);
    choices.replaceChildren();
    for (const item of matches.slice(page * pageSize, (page + 1) * pageSize)) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'choice';
        button.dataset.id = item.id;
        button.setAttribute('aria-pressed', String(item.id === selected));
        button.setAttribute('aria-label', item.name || item.caption || '选择这张照片');
        const img = document.createElement('img');
        img.alt = '';
        img.src = '/thumb/' + item.id;
        button.append(img);
        if (!isLibrary) {
            const label = document.createElement('span');
            label.textContent = item.name;
            button.append(label);
        }
        button.onclick = () => {
            dialog.close();
            choose(item.id);
        };
        choices.append(button);
    }
    $('no-choices').hidden = matches.length > 0;
    $('pagination').hidden = !isLibrary || pages <= 1;
    $('page-label').textContent = (page + 1) + ' / ' + pages;
    $('prev-page').disabled = page === 0;
    $('next-page').disabled = page + 1 >= pages;
    $('picker-scroll').scrollTop = 0;
}

export function openPicker(items, current, library, onChoose) {
    entries = matches = items;
    selected = current;
    isLibrary = library;
    choose = onChoose;
    page = 0;
    $('picker-title').textContent = library ? '选择参考照片' : '选择回放案例';
    $('gallery-tools').hidden = !library;
    $('gallery-query').value = '';
    $('choices').classList.toggle('library', library);
    renderPage();
    dialog.showModal();
    if (library) $('gallery-query').focus();
}
$('gallery-query').oninput = event => {
    const words = event.target.value.trim().toLowerCase().split(/\s+/);
    matches = entries.filter(item => words.every(word => item.search.includes(word)));
    page = 0;
    renderPage();
};
$('prev-page').onclick = () => { page--; renderPage(); };
$('next-page').onclick = () => { page++; renderPage(); };
$('close-picker').onclick = () => dialog.close();
dialog.onclick = event => { if (event.target === dialog) dialog.close(); };
