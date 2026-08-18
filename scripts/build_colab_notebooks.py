import json
from pathlib import Path


ROOT = Path("/Users/ann.lebedeva/Documents/lowres")
OUT = ROOT / "colab_notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = """
import os, re, json, textwrap, math, statistics, random, io
from pathlib import Path
import pandas as pd
import numpy as np
import requests

DATA_DIR = Path('/content/lowres_lab')
DATA_DIR.mkdir(exist_ok=True)

def show_df(df, n=10):
    display(df.head(n))

def save_artifact(name, obj):
    path = DATA_DIR / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(str(obj), encoding='utf-8')
    print('saved:', path)
"""


def lesson01_real_agent_cells():
    return [
        md("""
# Занятие 1. Реальный мини-агент языковой лаборатории

**Цель практики:** собрать не теоретический пример, а маленького агента, который делает полезную работу для проекта сохранения малоресурсного языка.

Сценарий: у нас есть задача первичной разведки по удмуртскому языку. Агент должен:

1. найти реальные открытые материалы в Wikimedia Commons;
2. выбрать изображение, которое можно скачать;
3. прогнать OCR baseline в бесплатном Colab;
4. взять небольшой контрольный корпус из Удмуртской Википедии;
5. собрать отчет: что найдено, насколько читаемый OCR, что нужно проверить человеку.

Сначала соберем агента без фреймворка: обычные функции, словарь `state`, явные вызовы инструментов и решения. Потом завернем те же шаги в `LangGraph`, чтобы увидеть, зачем вообще нужен фреймворк для агентских workflow.

LLM/API-ключи не нужны: на первом занятии важнее понять архитектуру агента, чем подключать внешнюю модель.
"""),
        code("""
!apt-get -qq update
!apt-get -qq install -y tesseract-ocr tesseract-ocr-rus
!pip -q install langgraph pytesseract pillow pandas matplotlib requests
"""),
        code(COMMON_SETUP + """
from typing import Any, Dict, List, TypedDict
from urllib.parse import quote
from PIL import Image, UnidentifiedImageError
from IPython.display import display
import matplotlib.pyplot as plt
import pytesseract
from langgraph.graph import StateGraph, END
"""),
        md("""
## 1. Настройка реальных источников

Берем два открытых источника:

- Wikimedia Commons: категория `Udmurt Dunne`, где лежат реальные файлы, связанные с удмуртской газетой.
- Удмуртская Википедия: небольшой корпус страниц для сравнения с OCR-выводом.

Это не учебная таблица: агент будет обращаться к API и сохранять полученные данные.
"""),
        code("""
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
UDM_WIKI_API = 'https://udm.wikipedia.org/w/api.php'

COMMONS_CATEGORY = 'Category:Udmurt Dunne'
LANGUAGE = 'удмуртский'
FALLBACK_FILE_TITLE = 'File:Удномер.jpg'
RASTER_MIME_TYPES = {'image/jpeg', 'image/png', 'image/tiff', 'image/webp'}

def api_get(url, params, timeout=30):
    r = requests.get(url, params=params, timeout=timeout, headers={'User-Agent': 'lowres-course-colab/1.0'})
    r.raise_for_status()
    return r.json()

def commons_category_files(category, limit=20):
    data = api_get(COMMONS_API, {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': category,
        'cmtype': 'file',
        'cmlimit': limit,
        'format': 'json',
    })
    return data.get('query', {}).get('categorymembers', [])

def commons_imageinfo(title):
    data = api_get(COMMONS_API, {
        'action': 'query',
        'titles': title,
        'prop': 'imageinfo',
        'iiprop': 'url|mime|size|extmetadata',
        'format': 'json',
    })
    pages = data.get('query', {}).get('pages', {})
    page = next(iter(pages.values()))
    info = page.get('imageinfo', [{}])[0]
    meta = info.get('extmetadata', {})
    return {
        'title': title,
        'url': info.get('url'),
        'mime': info.get('mime'),
        'width': info.get('width'),
        'height': info.get('height'),
        'license': meta.get('LicenseShortName', {}).get('value'),
        'artist': meta.get('Artist', {}).get('value'),
        'description': meta.get('ImageDescription', {}).get('value'),
    }

def is_raster_image(info):
    return bool(info.get('url')) and info.get('mime') in RASTER_MIME_TYPES

def fetch_udm_wiki_pages(limit=5):
    random_pages = api_get(UDM_WIKI_API, {
        'action': 'query',
        'generator': 'random',
        'grnnamespace': 0,
        'grnlimit': limit,
        'prop': 'extracts',
        'explaintext': 1,
        'exintro': 1,
        'format': 'json',
    })
    pages = []
    for page in random_pages.get('query', {}).get('pages', {}).values():
        title = page.get('title', '')
        extract = page.get('extract', '') or ''
        pages.append({
            'title': title,
            'chars': len(extract),
            'tokens': len(re.findall(r'\\w+', extract.lower())),
            'cyrillic_share': cyrillic_share(extract),
            'url': 'https://udm.wikipedia.org/wiki/' + quote(title.replace(' ', '_')),
            'extract_preview': extract[:400],
        })
    return pages

def cyrillic_share(text):
    letters = re.findall(r'[A-Za-zА-Яа-яЁёӜӝӞӟӤӥӦӧӴӵӸӹІіЇїЄєЎўҐґ]', text)
    if not letters:
        return 0.0
    cyr = [x for x in letters if re.match(r'[А-Яа-яЁёӜӝӞӟӤӥӦӧӴӵӸӹІіЇїЄєЎўҐґ]', x)]
    return round(len(cyr) / len(letters), 3)

def text_diagnostics(text):
    tokens = re.findall(r'\\w+', text.lower())
    short_tokens = [t for t in tokens if len(t) <= 2]
    weird = re.findall(r'[^\\w\\s.,:;!?()\\-«»\"\\'/А-Яа-яЁёӜӝӞӟӤӥӦӧӴӵӸӹІіЇїЄєЎўҐґ]', text)
    return {
        'chars': len(text),
        'tokens': len(tokens),
        'cyrillic_share': cyrillic_share(text),
        'short_token_share': round(len(short_tokens) / max(1, len(tokens)), 3),
        'weird_char_count': len(weird),
        'sample': text[:700],
    }
"""),
        md("""
## 2. Агент без фреймворка: что под капотом

Минимальный агент состоит из четырех вещей:

- `state`: явная модель текущего контекста задачи;
- tools: функции, которые ходят во внешний мир или обрабатывают данные;
- observations: результаты вызова tools;
- policy: простые правила, которые решают, что делать дальше.

Ниже это обычный Python. Никакого LangGraph пока нет.
"""),
        code("""
def scout_sources(state):
    files = commons_category_files(state['commons_category'])
    if not files:
        files = [{'title': FALLBACK_FILE_TITLE, 'pageid': None}]
    state['commons_files'] = files
    save_artifact('lesson01_commons_candidates.csv', pd.DataFrame(files))
    return state

def choose_downloadable_image(state):
    inspected = []
    selected = None
    for item in state['commons_files']:
        info = commons_imageinfo(item['title'])
        inspected.append(info)
        if is_raster_image(info) and selected is None:
            selected = info
    if selected is None:
        selected = commons_imageinfo(FALLBACK_FILE_TITLE)
    state['selected_file'] = selected
    save_artifact('lesson01_commons_sources.csv', pd.DataFrame(inspected))
    return state

def download_openable_image(info, candidate_index=0):
    response = requests.get(
        info['url'],
        timeout=60,
        headers={'User-Agent': 'lowres-course-colab/1.0 (teaching notebook)'},
    )
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
    if content_type and content_type not in RASTER_MIME_TYPES and content_type != 'application/octet-stream':
        raise ValueError(f'expected raster image, got Content-Type={content_type}')

    raw = response.content
    try:
        img = Image.open(io.BytesIO(raw)).convert('RGB')
    except UnidentifiedImageError as exc:
        raise ValueError('downloaded bytes are not an openable raster image') from exc

    if max(img.size) > 2200:
        img.thumbnail((2200, 2200))

    image_path = DATA_DIR / f'lesson01_ocr_source_{candidate_index}.jpg'
    img.save(image_path, format='JPEG', quality=92)
    return image_path

def run_ocr_baseline(state):
    selected_title = state['selected_file'].get('title')
    candidates = [state['selected_file']]
    for item in state['commons_files']:
        if item.get('title') != selected_title:
            candidates.append(commons_imageinfo(item['title']))

    errors = []
    image_path = None
    selected = None
    for i, candidate in enumerate(candidates):
        if not is_raster_image(candidate):
            errors.append({'title': candidate.get('title'), 'error': f"unsupported mime: {candidate.get('mime')}"})
            continue
        try:
            image_path = download_openable_image(candidate, i)
            selected = candidate
            break
        except Exception as exc:
            errors.append({'title': candidate.get('title'), 'error': str(exc)})

    if selected is None or image_path is None:
        save_artifact('lesson01_download_errors.csv', pd.DataFrame(errors))
        raise RuntimeError('Не удалось скачать ни одного открываемого raster-изображения из Commons candidates.')

    if errors:
        save_artifact('lesson01_download_errors.csv', pd.DataFrame(errors))

    text = pytesseract.image_to_string(Image.open(image_path), lang='rus')
    save_artifact('lesson01_ocr_raw.txt', text)

    state['selected_file'] = selected
    state['image_path'] = str(image_path)
    state['ocr_text'] = text
    return state

def decide_human_tasks(state):
    diag = text_diagnostics(state.get('ocr_text', ''))
    tasks = []
    if diag['chars'] < 100:
        tasks.append('OCR почти ничего не извлек: выбрать другой скан или улучшить предобработку.')
    if diag['cyrillic_share'] < 0.7:
        tasks.append('В тексте много некириллического шума: проверить язык OCR и качество изображения.')
    if diag['short_token_share'] > 0.45:
        tasks.append('Много коротких фрагментов: нужна ручная проверка строк и сегментации.')
    if not tasks:
        tasks.append('Выбрать 20-30 строк и вручную оценить ошибки OCR.')
    state['ocr_diagnostics'] = diag
    state['human_tasks'] = tasks
    return state

def add_wiki_probe(state):
    pages = fetch_udm_wiki_pages(limit=5)
    save_artifact('lesson01_udm_wiki_probe.csv', pd.DataFrame(pages))
    all_text = ' '.join(p['extract_preview'] for p in pages)
    state['wiki_pages'] = pages
    state['wiki_stats'] = text_diagnostics(all_text)
    return state

def build_agent_report(state):
    report = {
        'language': state['language'],
        'agent_type': state.get('agent_type', 'plain Python tool-using agent'),
        'real_sources': {
            'commons_category': state['commons_category'],
            'selected_file_title': state['selected_file'].get('title'),
            'selected_file_url': state['selected_file'].get('url'),
            'license': state['selected_file'].get('license'),
            'wiki_pages': [p['url'] for p in state['wiki_pages']],
        },
        'ocr_diagnostics': state['ocr_diagnostics'],
        'wiki_probe_stats': state['wiki_stats'],
        'next_human_tasks': state['human_tasks'],
        'state_fields_used': sorted(state.keys()),
    }
    state['report'] = report
    save_artifact('lesson01_real_agent_report.json', json.dumps(report, ensure_ascii=False, indent=2))
    return state

def run_plain_agent(language, commons_category):
    state = {
        'language': language,
        'commons_category': commons_category,
        'goal': 'оценить, можно ли начать OCR-разведку по открытым удмуртским материалам',
        'plan': [
            'найти файлы',
            'выбрать изображение',
            'запустить OCR',
            'оценить текст',
            'сравнить с wiki-корпусом',
            'собрать отчет',
        ],
    }
    for step in [
        scout_sources,
        choose_downloadable_image,
        run_ocr_baseline,
        decide_human_tasks,
        add_wiki_probe,
        build_agent_report,
    ]:
        state = step(state)
        print('done:', step.__name__, '| state keys:', sorted(state.keys()))
    return state

plain_state = run_plain_agent(LANGUAGE, COMMONS_CATEGORY)
print(json.dumps(plain_state['report'], ensure_ascii=False, indent=2))
"""),
        md("""
## 3. Та же логика в LangGraph

Теперь берем те же функции и раскладываем их в граф. Смысл LangGraph не в том, что он “умнее”, а в том, что он делает архитектуру явной:

- у каждого узла есть входной и выходной `state`;
- переходы между шагами видны отдельно от кода инструментов;
- позже можно добавлять условия, повторы, LLM-узлы, human review и сохранение состояния между запусками.
"""),
        code("""
class LabAgentState(TypedDict, total=False):
    language: str
    commons_category: str
    commons_files: List[Dict[str, Any]]
    selected_file: Dict[str, Any]
    image_path: str
    ocr_text: str
    ocr_diagnostics: Dict[str, Any]
    wiki_pages: List[Dict[str, Any]]
    wiki_stats: Dict[str, Any]
    report: Dict[str, Any]
    human_tasks: List[str]
    agent_type: str

def source_scout_node(state: LabAgentState):
    next_state = scout_sources(dict(state))
    return {'commons_files': next_state['commons_files']}

def metadata_node(state: LabAgentState):
    next_state = choose_downloadable_image(dict(state))
    return {'selected_file': next_state['selected_file']}

def ocr_node(state: LabAgentState):
    next_state = run_ocr_baseline(dict(state))
    return {'image_path': next_state['image_path'], 'ocr_text': next_state['ocr_text']}

def ocr_diagnostics_node(state: LabAgentState):
    next_state = decide_human_tasks(dict(state))
    return {'ocr_diagnostics': next_state['ocr_diagnostics'], 'human_tasks': next_state['human_tasks']}

def wiki_probe_node(state: LabAgentState):
    next_state = add_wiki_probe(dict(state))
    return {'wiki_pages': next_state['wiki_pages'], 'wiki_stats': next_state['wiki_stats']}

def report_node(state: LabAgentState):
    next_state = build_agent_report(dict(state))
    return {'report': next_state['report']}

workflow = StateGraph(LabAgentState)
workflow.add_node('source_scout', source_scout_node)
workflow.add_node('metadata', metadata_node)
workflow.add_node('ocr', ocr_node)
workflow.add_node('ocr_diagnostics', ocr_diagnostics_node)
workflow.add_node('wiki_probe', wiki_probe_node)
workflow.add_node('report', report_node)

workflow.set_entry_point('source_scout')
workflow.add_edge('source_scout', 'metadata')
workflow.add_edge('metadata', 'ocr')
workflow.add_edge('ocr', 'ocr_diagnostics')
workflow.add_edge('ocr_diagnostics', 'wiki_probe')
workflow.add_edge('wiki_probe', 'report')
workflow.add_edge('report', END)

agent = workflow.compile()
"""),
        md("## 4. Запускаем LangGraph-версию"),
        code("""
state = agent.invoke({
    'language': LANGUAGE,
    'commons_category': COMMONS_CATEGORY,
    'agent_type': 'LangGraph tool-using workflow agent',
})

print(json.dumps(state['report'], ensure_ascii=False, indent=2))
"""),
        md("## 5. Смотрим реальные артефакты"),
        code("""
print('Выбранный файл:', state['selected_file']['title'])
print('Лицензия:', state['selected_file'].get('license'))
print('URL:', state['selected_file'].get('url'))

img = Image.open(state['image_path'])
print('Размер изображения для OCR:', img.size)
display(img)
"""),
        code("""
print(state['ocr_text'][:2000])
"""),
        code("""
display(pd.DataFrame(state['wiki_pages'])[['title', 'chars', 'tokens', 'cyrillic_share', 'url']])
display(pd.DataFrame([state['ocr_diagnostics'], state['wiki_stats']], index=['ocr', 'wiki_probe']))
"""),
        md("""
## 6. Что здесь агентского

В этой вводной тетрадке важны не “виды агентов”, а архитектура на конкретной задаче:

- `state` хранит текущую картину задачи: цель, найденные источники, выбранный файл, OCR-текст, диагностику, контрольный корпус и задачи для человека.
- каждый узел делает один проверяемый шаг;
- следующий шаг выбирается на основании уже собранного состояния;
- человек остается в контуре там, где нельзя автоматически решать про права, качество и публикацию.

В следующих тетрадках мы уже не будем каждый раз подробно разбирать архитектуру. Будем использовать этот принцип как рабочий шаблон: источник, tool, state, диагностика, human boundary, отчет.
"""),
        md("""
## Вопросы для отчета

1. Какой реальный материал нашел агент и можно ли понять его лицензию?
2. Какие части plain Python-агента соответствуют `state`, tools, observations и policy?
3. Что стало понятнее или надежнее после переноса той же логики в LangGraph?
4. Получился ли OCR-текст вменяемым? Покажите 3-5 характерных ошибок.
5. Что в этой задаче нельзя отдавать агенту полностью автоматически?
"""),
    ]


NOTEBOOKS = {
    "01_agents_for_language_preservation.ipynb": lesson01_real_agent_cells(),
    "02_web_scraping_sources.ipynb": [
        md("""
# Занятие 2. Веб-сбор, источники и правовая рамка

**Цель практики:** собрать маленькую таблицу источников через открытые API, извлечь текст, проверить шум и зафиксировать лицензионные/этические риски.

Работает в бесплатном Colab на CPU. Используем Wikipedia API вместо агрессивного скрейпинга сайтов.
"""),
        code(COMMON_SETUP),
        md("## 1. Получаем страницы из Удмуртской Википедии"),
        code("""
LANG = 'udm'
API = f'https://{LANG}.wikipedia.org/w/api.php'

def search_pages(query, limit=10):
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'srlimit': limit,
        'format': 'json',
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()['query']['search']

hits = search_pages('удмурт', 10)
sources = pd.DataFrame([{
    'title': h['title'],
    'pageid': h['pageid'],
    'url': f'https://{LANG}.wikipedia.org/wiki/' + requests.utils.quote(h['title'].replace(' ', '_')),
    'type': 'wiki_article',
    'license_or_access': 'Wikipedia text: CC BY-SA, check page history and terms',
    'notes': 'API search result',
} for h in hits])
show_df(sources, 10)
save_artifact('lesson02_sources.csv', sources)
"""),
        md("## 2. Извлекаем тексты и считаем базовый шум"),
        code("""
def get_extract(pageid):
    params = {
        'action': 'query',
        'pageids': pageid,
        'prop': 'extracts|info',
        'explaintext': 1,
        'inprop': 'url',
        'format': 'json',
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    page = next(iter(r.json()['query']['pages'].values()))
    return page.get('extract', ''), page.get('fullurl')

texts = []
for _, row in sources.iterrows():
    txt, url = get_extract(row.pageid)
    texts.append({'pageid': row.pageid, 'title': row.title, 'url': url, 'text': txt, 'chars': len(txt)})

corpus = pd.DataFrame(texts)
corpus['cyrillic_share'] = corpus['text'].apply(lambda x: len(re.findall(r'[А-Яа-яЁёӐ-ӿ]', x)) / max(len(x), 1))
corpus['line_count'] = corpus['text'].apply(lambda x: len([l for l in x.splitlines() if l.strip()]))
show_df(corpus[['title', 'chars', 'cyrillic_share', 'line_count', 'url']], 10)
save_artifact('lesson02_raw_corpus.csv', corpus)
"""),
        md("## 3. Простая фильтрация и отчет агента-ревьюера"),
        code("""
filtered = corpus[(corpus['chars'] >= 200) & (corpus['cyrillic_share'] > 0.45)].copy()
filtered['duplicate_text'] = filtered.duplicated('text')

review = {
    'n_sources': len(sources),
    'n_texts_kept': len(filtered),
    'too_short': int((corpus['chars'] < 200).sum()),
    'low_cyrillic_share': int((corpus['cyrillic_share'] <= 0.45).sum()),
    'duplicates': int(filtered['duplicate_text'].sum()),
    'human_checks': [
        'проверить лицензию и условия использования',
        'проверить, что текст действительно на нужном языке',
        'убрать навигацию, списки и служебные фрагменты',
    ],
}
print(json.dumps(review, ensure_ascii=False, indent=2))
save_artifact('lesson02_filtered_corpus.csv', filtered)
save_artifact('lesson02_review.json', json.dumps(review, ensure_ascii=False, indent=2))
"""),
        md("""
## Вопросы для отчёта

1. Какие страницы пришлось бы исключить и почему?
2. Достаточно ли API-источника для курса или нужны другие сайты/архивы?
3. Какие поля метаданных нужно добавить в `sources.csv`?
4. Что агент может сделать сам, а что должен подтвердить человек?
"""),
    ],
    "03_ocr_udmurt_commons.ipynb": [
        md("""
# Занятие 3. OCR для открытого скана на малоресурсном языке России

**Цель практики:** взять открытое изображение из Wikimedia Commons, прогнать лёгкий OCR baseline и понять, получается ли текст на вменяемом языке.

Пример использует материалы категории **Udmurt Dunne** на Wikimedia Commons и Tesseract с русской OCR-моделью как baseline для кириллического удмуртского текста. Это не “правильная удмуртская OCR-модель”, а проверка: насколько далеко можно зайти простым бесплатным инструментом.

Работает в бесплатном Colab на CPU.
"""),
        code("""
!apt-get -qq update
!apt-get -qq install -y tesseract-ocr tesseract-ocr-rus
!pip -q install pytesseract pillow opencv-python-headless pandas matplotlib
"""),
        code(COMMON_SETUP + "\nfrom PIL import Image\nimport pytesseract\nimport matplotlib.pyplot as plt\nimport cv2\n"),
        md("## 1. Находим файл в Wikimedia Commons через API"),
        code("""
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
FILE_TITLE = 'File:Удномер.jpg'  # из категории Wikimedia Commons: Udmurt Dunne

params = {
    'action': 'query',
    'titles': FILE_TITLE,
    'prop': 'imageinfo',
    'iiprop': 'url|mime|size|extmetadata',
    'format': 'json',
}
r = requests.get(COMMONS_API, params=params, timeout=30)
r.raise_for_status()
page = next(iter(r.json()['query']['pages'].values()))
info = page['imageinfo'][0]
image_url = info['url']
print(image_url)
print('license:', info.get('extmetadata', {}).get('LicenseShortName', {}).get('value'))
"""),
        md("## 2. Скачиваем изображение и смотрим на него"),
        code("""
img_path = DATA_DIR / 'udmurt_sample.jpg'
img_path.write_bytes(requests.get(image_url, timeout=60).content)
img = Image.open(img_path)
print(img.size)
plt.figure(figsize=(6, 8))
plt.imshow(img)
plt.axis('off');
"""),
        md("## 3. Запускаем Tesseract baseline"),
        code("""
raw_text = pytesseract.image_to_string(img, lang='rus')
print(raw_text[:2000])
save_artifact('lesson03_ocr_raw.txt', raw_text)
"""),
        md("## 4. Улучшаем препроцессинг и сравниваем"),
        code("""
gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
gray = cv2.fastNlMeansDenoising(gray, h=20)
thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
prep_path = DATA_DIR / 'udmurt_sample_preprocessed.png'
cv2.imwrite(str(prep_path), thr)

prep_img = Image.open(prep_path)
prep_text = pytesseract.image_to_string(prep_img, lang='rus')
print(prep_text[:2000])
save_artifact('lesson03_ocr_preprocessed.txt', prep_text)

plt.figure(figsize=(6, 8))
plt.imshow(prep_img, cmap='gray')
plt.axis('off');
"""),
        md("## 5. Быстрая диагностика качества без ground truth"),
        code("""
def ocr_diagnostics(text):
    chars = len(text)
    letters = re.findall(r'[А-Яа-яЁёӐ-ӿ]', text)
    tokens = re.findall(r'[А-Яа-яЁёӐ-ӿ]{2,}', text)
    weird = re.findall(r'[^А-Яа-яЁёӐ-ӿ0-9\\s.,:;!?()\\-—"«»]', text)
    return {
        'chars': chars,
        'cyrillic_letters': len(letters),
        'tokens_2plus': len(tokens),
        'unique_tokens_2plus': len(set(t.lower() for t in tokens)),
        'weird_char_share': len(weird) / max(chars, 1),
    }

diag = pd.DataFrame([
    {'version': 'raw', **ocr_diagnostics(raw_text)},
    {'version': 'preprocessed', **ocr_diagnostics(prep_text)},
])
show_df(diag)
save_artifact('lesson03_ocr_diagnostics.csv', diag)
"""),
        md("""
## Вопросы для отчёта

1. Можно ли читать результат глазами? Какие слова/буквы распознаются хуже всего?
2. Улучшил ли препроцессинг результат?
3. Почему русская OCR-модель может ошибаться на удмуртском?
4. Какие 30-50 строк стоит вручную разметить как ground truth для следующего шага?
"""),
    ],
    "04_asr_udmurt_whisper_tiny.ipynb": [
        md("""
# Занятие 4. ASR baseline на открытом аудио удмуртской речи

**Цель практики:** взять короткий открытый аудиофайл на удмуртском языке с Wikimedia Commons, прогнать маленькую ASR-модель и понять, получается ли осмысленная транскрипция.

Используем `openai/whisper-tiny` через Transformers. Это маленькая модель; она удобна для бесплатного Colab, но может плохо работать с языками, которых нет в её сильных режимах. Это и есть предмет анализа.
"""),
        code("""
!pip -q install transformers accelerate librosa soundfile pandas
"""),
        code(COMMON_SETUP + "\nimport librosa, soundfile as sf\nfrom IPython.display import Audio\nfrom transformers import pipeline\n"),
        md("## 1. Скачиваем открытое аудио из Wikimedia Commons"),
        code("""
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
FILE_TITLE = 'File:Udmurt.ogg'
params = {
    'action': 'query',
    'titles': FILE_TITLE,
    'prop': 'imageinfo',
    'iiprop': 'url|mime|size|extmetadata',
    'format': 'json',
}
r = requests.get(COMMONS_API, params=params, timeout=30)
r.raise_for_status()
page = next(iter(r.json()['query']['pages'].values()))
info = page['imageinfo'][0]
audio_url = info['url']
print(audio_url)
print('license:', info.get('extmetadata', {}).get('LicenseShortName', {}).get('value'))

audio_path = DATA_DIR / 'udmurt.ogg'
audio_path.write_bytes(requests.get(audio_url, timeout=60).content)
Audio(str(audio_path))
"""),
        md("## 2. Нормализуем аудио под ASR"),
        code("""
y, sr = librosa.load(audio_path, sr=16000, mono=True)
trimmed = y[: 30 * 16000]
wav_path = DATA_DIR / 'udmurt_16k.wav'
sf.write(wav_path, trimmed, 16000)
print('seconds:', round(len(trimmed) / 16000, 2))
Audio(str(wav_path))
"""),
        md("## 3. Запускаем Whisper tiny"),
        code("""
asr = pipeline('automatic-speech-recognition', model='openai/whisper-tiny')
result = asr(str(wav_path), generate_kwargs={'task': 'transcribe'})
print(result['text'])
save_artifact('lesson04_asr_whisper_tiny.txt', result['text'])
"""),
        md("## 4. Анализируем результат как baseline, а не как истину"),
        code("""
transcript = result['text']
analysis = {
    'chars': len(transcript),
    'tokens': len(transcript.split()),
    'cyrillic_share': len(re.findall(r'[А-Яа-яЁёӐ-ӿ]', transcript)) / max(len(transcript), 1),
    'latin_share': len(re.findall(r'[A-Za-z]', transcript)) / max(len(transcript), 1),
    'questions_for_human': [
        'на каком языке модель фактически распознала речь?',
        'есть ли совпадающие удмуртские слова?',
        'нужна ли ручная транскрипция хотя бы 30 секунд?',
        'какие шумы или особенности голоса мешают?',
    ],
}
print(json.dumps(analysis, ensure_ascii=False, indent=2))
save_artifact('lesson04_asr_analysis.json', json.dumps(analysis, ensure_ascii=False, indent=2))
"""),
        md("""
## Вопросы для отчёта

1. Похожа ли транскрипция на речь в аудио?
2. Модель ошиблась языком или просто дала шумный текст?
3. Какой минимальный набор ручной разметки нужен для честной оценки?
4. Что должно быть в протоколе записи для будущих полевых данных?
"""),
    ],
    "05_corpus_cleaning_datacard.ipynb": [
        md("""
# Занятие 5. Очистка, фильтрация и data card маленького корпуса

**Цель практики:** собрать мини-корпус через Wikipedia API, удалить дубли/короткие тексты, посчитать простые признаки качества и создать черновик data card.

Работает в бесплатном Colab на CPU.
"""),
        code(COMMON_SETUP),
        md("## 1. Скачиваем мини-корпус"),
        code("""
LANG = 'udm'
API = f'https://{LANG}.wikipedia.org/w/api.php'

def random_extracts(n=25):
    params = {
        'action': 'query',
        'generator': 'random',
        'grnnamespace': 0,
        'grnlimit': n,
        'prop': 'extracts|info',
        'explaintext': 1,
        'inprop': 'url',
        'format': 'json',
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json().get('query', {}).get('pages', {})
    return pd.DataFrame([{
        'title': p.get('title'),
        'url': p.get('fullurl'),
        'text': p.get('extract', ''),
    } for p in pages.values()])

df = random_extracts(25)
df.loc[len(df)] = df.iloc[0]  # искусственный дубль для практики
show_df(df[['title', 'url']], 10)
"""),
        md("## 2. Чистим текст и считаем признаки"),
        code("""
def normalize_space(text):
    return re.sub(r'\\s+', ' ', str(text)).strip()

clean = df.copy()
clean['text_clean'] = clean['text'].apply(normalize_space)
clean['chars'] = clean['text_clean'].str.len()
clean['tokens'] = clean['text_clean'].apply(lambda x: len(re.findall(r'\\w+', x)))
clean['cyrillic_share'] = clean['text_clean'].apply(lambda x: len(re.findall(r'[А-Яа-яЁёӐ-ӿ]', x)) / max(len(x), 1))
clean['duplicate'] = clean.duplicated('text_clean')
clean['keep'] = (clean['chars'] >= 300) & (clean['cyrillic_share'] >= 0.45) & (~clean['duplicate'])

show_df(clean[['title', 'chars', 'tokens', 'cyrillic_share', 'duplicate', 'keep']], 30)
filtered = clean[clean['keep']].copy()
save_artifact('lesson05_clean_corpus.csv', filtered[['title', 'url', 'text_clean', 'chars', 'tokens', 'cyrillic_share']])
"""),
        md("## 3. Делим на train/dev/test без обучения модели"),
        code("""
filtered = filtered.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(filtered)
filtered['split'] = 'train'
filtered.loc[filtered.index >= int(n * 0.8), 'split'] = 'dev'
filtered.loc[filtered.index >= int(n * 0.9), 'split'] = 'test'
show_df(filtered[['title', 'chars', 'split']], 30)
save_artifact('lesson05_splits.csv', filtered[['title', 'url', 'split', 'text_clean']])
"""),
        md("## 4. Черновик data card"),
        code("""
data_card = f'''# Data card: mini {LANG} Wikipedia corpus

## Source
Wikipedia API, language edition: {LANG}.wikipedia.org

## Size
- Raw documents: {len(df)}
- Kept documents: {len(filtered)}
- Total kept characters: {int(filtered['chars'].sum()) if len(filtered) else 0}

## Filtering
- Removed exact duplicates
- Removed documents shorter than 300 characters
- Required Cyrillic-script share >= 0.45

## Known limitations
- Wikipedia is not representative of all language varieties
- Articles may contain named entities, Russian borrowings, lists and formatting remnants
- Human language review is still required

## License
Check Wikipedia/Wikimedia terms and page histories before redistribution.
'''
print(data_card)
save_artifact('lesson05_DATA_CARD.md', data_card)
"""),
        md("""
## Вопросы для отчёта

1. Что фильтры удалили ошибочно?
2. Какие признаки качества не видны из автоматических метрик?
3. Что обязательно должен проверить носитель/эксперт?
4. Что нужно добавить в data card перед публикацией?
"""),
    ],
    "06_parallel_alignment_mt.ipynb": [
        md("""
# Занятие 6. Параллельные тексты, alignment и переводческий baseline

**Цель практики:** загрузить маленький татарско-русский параллельный корпус, проверить кандидаты выравнивания и попробовать простой multilingual embedding baseline.

Работает в бесплатном Colab на CPU. Если Hugging Face загрузка недоступна, тетрадка использует встроенный мини-набор для демонстрации.
"""),
        code("""
!pip -q install datasets sentence-transformers pandas scikit-learn
"""),
        code(COMMON_SETUP + "\nfrom sklearn.metrics.pairwise import cosine_similarity\n"),
        md("## 1. Загружаем небольшой фрагмент параллельного корпуса"),
        code("""
fallback = pd.DataFrame({
    'tat': [
        'Мин мәктәпкә барам.',
        'Бу китап бик кызык.',
        'Без бүген яңа проект башлыйбыз.',
        'Телне саклау өчен мәгълүмат кирәк.',
        'Укытучы сораулар бирә.',
    ],
    'rus': [
        'Я иду в школу.',
        'Эта книга очень интересная.',
        'Сегодня мы начинаем новый проект.',
        'Для сохранения языка нужны данные.',
        'Учитель задает вопросы.',
    ],
})

try:
    from datasets import load_dataset
    ds = load_dataset('AigizK/tatar-russian-parallel-corpora', split='train[:80]')
    pairs = ds.to_pandas()
    print(pairs.columns)
    # Пытаемся найти текстовые колонки автоматически.
    text_cols = [c for c in pairs.columns if pairs[c].dtype == 'object']
    pairs = pairs[text_cols[:2]].dropna().head(50)
    pairs.columns = ['tat', 'rus']
except Exception as e:
    print('HF loading failed, using fallback:', e)
    pairs = fallback

show_df(pairs, 10)
save_artifact('lesson06_parallel_pairs.csv', pairs)
"""),
        md("## 2. Создаем испорченный порядок и ищем alignment"),
        code("""
left = pairs['tat'].reset_index(drop=True)
right = pairs['rus'].sample(frac=1, random_state=7).reset_index(drop=True)
candidate_df = pd.DataFrame({'tat': left, 'rus_shuffled': right})
show_df(candidate_df, 10)
"""),
        md("## 3. Embedding baseline для поиска пар"),
        code("""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
emb_left = model.encode(left.tolist(), normalize_embeddings=True)
emb_right = model.encode(right.tolist(), normalize_embeddings=True)
sim = cosine_similarity(emb_left, emb_right)

matches = []
for i in range(len(left)):
    j = int(sim[i].argmax())
    matches.append({
        'tat': left.iloc[i],
        'best_rus': right.iloc[j],
        'score': float(sim[i, j]),
        'gold_rus': pairs['rus'].iloc[i],
        'correct_exact': right.iloc[j] == pairs['rus'].iloc[i],
    })
matches = pd.DataFrame(matches)
show_df(matches, 20)
print('exact alignment accuracy:', matches['correct_exact'].mean())
save_artifact('lesson06_alignment_candidates.csv', matches)
"""),
        md("## 4. Что делать с переводом"),
        code("""
low_conf = matches[matches['score'] < 0.5]
report = {
    'pairs': len(matches),
    'exact_alignment_accuracy': float(matches['correct_exact'].mean()),
    'low_confidence_pairs': len(low_conf),
    'next_steps': [
        'проверить top-20 кандидатов вручную',
        'сохранять score и источник каждой пары',
        'не обучать MT на непроверенных парах',
        'сделать dev/test из вручную подтвержденных пар',
    ],
}
print(json.dumps(report, ensure_ascii=False, indent=2))
save_artifact('lesson06_alignment_report.json', json.dumps(report, ensure_ascii=False, indent=2))
"""),
        md("""
## Вопросы для отчёта

1. Какие пары baseline нашел неверно?
2. Можно ли доверять score без ручной проверки?
3. Сколько проверенных пар нужно собрать до первого MT baseline?
4. Какие права и источники нужно сохранить рядом с парами?
"""),
    ],
    "07_spellchecker_normalization.ipynb": [
        md("""
# Занятие 7. Частотный словарь, нормализация и простой спелчекер

**Цель практики:** собрать маленький словарь из открытых текстов, искусственно внести ошибки и проверить простой baseline исправления через edit distance.

Работает в бесплатном Colab на CPU.
"""),
        code("""
!pip -q install rapidfuzz pandas
"""),
        code(COMMON_SETUP + "\nfrom rapidfuzz import process, fuzz\n"),
        md("## 1. Собираем слова из мини-корпуса"),
        code("""
LANG = 'udm'
API = f'https://{LANG}.wikipedia.org/w/api.php'

params = {
    'action': 'query',
    'generator': 'random',
    'grnnamespace': 0,
    'grnlimit': 20,
    'prop': 'extracts',
    'explaintext': 1,
    'format': 'json',
}
r = requests.get(API, params=params, timeout=30)
r.raise_for_status()
pages = r.json().get('query', {}).get('pages', {})
texts = [p.get('extract', '') for p in pages.values()]
tokens = []
for t in texts:
    tokens.extend(re.findall(r'[А-Яа-яЁёӐ-ӿ]{3,}', t.lower()))

freq = pd.Series(tokens).value_counts().reset_index()
freq.columns = ['word', 'count']
freq = freq[freq['count'] >= 1].head(300)
show_df(freq, 20)
save_artifact('lesson07_frequency_dictionary.csv', freq)
"""),
        md("## 2. Делаем искусственные ошибки"),
        code("""
alphabet = sorted(set(''.join(freq['word'].head(100).tolist())))

def corrupt(word):
    if len(word) < 4:
        return word
    i = random.randrange(len(word))
    op = random.choice(['delete', 'swap', 'replace'])
    if op == 'delete':
        return word[:i] + word[i+1:]
    if op == 'swap' and i < len(word) - 1:
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    return word[:i] + random.choice(alphabet or ['а']) + word[i+1:]

test_words = freq['word'].head(40).tolist()
typos = pd.DataFrame({'gold': test_words})
typos['typo'] = typos['gold'].apply(corrupt)
show_df(typos, 20)
"""),
        md("## 3. Исправляем через ближайшее слово"),
        code("""
vocab = freq['word'].tolist()

def suggest(word, limit=3):
    return process.extract(word, vocab, scorer=fuzz.WRatio, limit=limit)

rows = []
for _, row in typos.iterrows():
    sugg = suggest(row['typo'])
    rows.append({
        'typo': row['typo'],
        'gold': row['gold'],
        'top1': sugg[0][0] if sugg else None,
        'top1_score': sugg[0][1] if sugg else None,
        'top3': [s[0] for s in sugg],
    })

eval_df = pd.DataFrame(rows)
eval_df['top1_correct'] = eval_df['top1'] == eval_df['gold']
eval_df['top3_correct'] = eval_df.apply(lambda r: r['gold'] in r['top3'], axis=1)
show_df(eval_df, 40)
print('top1 accuracy:', eval_df['top1_correct'].mean())
print('top3 accuracy:', eval_df['top3_correct'].mean())
save_artifact('lesson07_spellchecker_eval.csv', eval_df)
"""),
        md("## 4. Где нормализация отличается от исправления"),
        code("""
normalization_questions = pd.DataFrame([
    {'case': 'вариант орфографии', 'should_fix?': 'не всегда', 'human_check': 'является ли форма допустимой нормой?'},
    {'case': 'OCR-ошибка', 'should_fix?': 'часто да', 'human_check': 'есть ли изображение-источник?'},
    {'case': 'диалектная форма', 'should_fix?': 'нет без решения сообщества', 'human_check': 'какой вариант нужен в корпусе?'},
    {'case': 'заимствование/имя', 'should_fix?': 'осторожно', 'human_check': 'это слово языка или шум?'},
])
show_df(normalization_questions)
save_artifact('lesson07_normalization_questions.csv', normalization_questions)
"""),
        md("""
## Вопросы для отчёта

1. Какие ошибки baseline исправляет хорошо?
2. Какие “ошибки” могут оказаться нормальными вариантами?
3. Что нужно хранить в словаре: частоты, источники, варианты, комментарии?
4. Кто должен принимать решение о норме?
"""),
    ],
    "08_final_package_review.ipynb": [
        md("""
# Занятие 8. Финальная интеграция и агентная проверка пакета

**Цель практики:** собрать минимальный финальный пакет проекта в Colab, проверить его чек-листом и получить список задач перед публикацией/летней школой.

Работает в бесплатном Colab на CPU.
"""),
        code(COMMON_SETUP),
        md("## 1. Создаем мини-пакет проекта"),
        code("""
PROJECT = DATA_DIR / 'final_package'
PROJECT.mkdir(exist_ok=True)

files = {
    'README.md': '''# Mini low-resource language project

Goal: create a tiny documented corpus prototype.
Language: Udmurt / replace with your language.
Status: classroom prototype, not production.
''',
    'DATA_CARD.md': '''# Data card

Sources: Wikipedia API / Wikimedia Commons / replace with your sources.
License: check source pages before redistribution.
Known limitations: small sample, not representative, needs human review.
''',
    'EVAL_REPORT.md': '''# Evaluation report

Metric: manual inspection + simple script diagnostics.
Result: baseline works partially.
Errors: needs language expert review.
''',
    'sources.csv': 'title,url,type,license_or_access,notes\\nExample,https://example.org,web,unknown,replace me\\n',
}
for name, content in files.items():
    (PROJECT / name).write_text(content, encoding='utf-8')

print('created files:')
for p in sorted(PROJECT.iterdir()):
    print('-', p.name)
"""),
        md("## 2. Проверяем структуру пакета"),
        code("""
required = ['README.md', 'DATA_CARD.md', 'EVAL_REPORT.md', 'sources.csv']
checks = []
for name in required:
    p = PROJECT / name
    checks.append({
        'check': f'{name} exists',
        'ok': p.exists(),
        'details': str(p),
    })
    if p.exists():
        txt = p.read_text(encoding='utf-8')
        checks.append({
            'check': f'{name} is not empty',
            'ok': len(txt.strip()) > 40,
            'details': f'{len(txt)} chars',
        })

sources = pd.read_csv(PROJECT / 'sources.csv')
for col in ['title', 'url', 'type', 'license_or_access', 'notes']:
    checks.append({'check': f'sources.csv has column {col}', 'ok': col in sources.columns, 'details': ''})

check_df = pd.DataFrame(checks)
show_df(check_df, 30)
save_artifact('lesson08_package_checks.csv', check_df)
"""),
        md("## 3. Агентный review без LLM: правила и задачи"),
        code("""
tasks = []

if not check_df['ok'].all():
    tasks.append('исправить отсутствующие или пустые обязательные файлы')

if 'unknown' in sources['license_or_access'].fillna('').str.lower().to_string():
    tasks.append('уточнить лицензии в sources.csv')

readme = (PROJECT / 'README.md').read_text(encoding='utf-8').lower()
if 'replace' in readme:
    tasks.append('заменить placeholder-описания в README.md')

data_card = (PROJECT / 'DATA_CARD.md').read_text(encoding='utf-8').lower()
if 'human review' in data_card or 'needs' in data_card:
    tasks.append('запланировать ручную проверку носителем/экспертом')

report = {
    'package_ready': len(tasks) == 0,
    'tasks_before_publication': tasks,
    'summer_school_angle': [
        'какой артефакт можно показать участникам',
        'какие данные нужно дособрать',
        'какие роли нужны в команде',
        'какие риски нельзя автоматизировать',
    ],
}
print(json.dumps(report, ensure_ascii=False, indent=2))
save_artifact('lesson08_agent_review.json', json.dumps(report, ensure_ascii=False, indent=2))
"""),
        md("## 4. Финальная таблица для защиты"),
        code("""
def status(ok):
    return 'готово' if ok else 'нужно доработать'

def exists(name):
    return (PROJECT / name).exists()

def nonempty(name):
    p = PROJECT / name
    return p.exists() and len(p.read_text(encoding='utf-8').strip()) > 40

defense = pd.DataFrame([
    {'artifact': 'README', 'status': status(nonempty('README.md')), 'next_step': 'уточнить цель и пользователей'},
    {'artifact': 'sources.csv', 'status': status(exists('sources.csv')), 'next_step': 'добавить реальные лицензии'},
    {'artifact': 'DATA_CARD', 'status': status(nonempty('DATA_CARD.md')), 'next_step': 'описать ограничения'},
    {'artifact': 'EVAL_REPORT', 'status': status(nonempty('EVAL_REPORT.md')), 'next_step': 'добавить примеры ошибок'},
])
show_df(defense)
save_artifact('lesson08_defense_table.csv', defense)
"""),
        md("""
## Вопросы для отчёта

1. Что в пакете уже можно показать внешнему человеку?
2. Что нельзя публиковать без дополнительной проверки?
3. Какие задачи переходят в летнюю школу?
4. Какие проверки стоит автоматизировать агентом?
"""),
    ],
}


README = """# Colab notebooks

These notebooks are classroom practices for the course sessions. They are designed for free Google Colab on CPU or, where useful, the free GPU tier. Each notebook keeps data small, writes artifacts under `/content/lowres_lab`, and asks students to inspect the output rather than trust a model blindly.

| Notebook | Practice | Open in Colab |
|---|---|---|
| `01_agents_for_language_preservation.ipynb` | real LangGraph mini-agent for source scouting, OCR baseline, and low-resource language diagnostics | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/01_agents_for_language_preservation.ipynb) |
| `02_web_scraping_sources.ipynb` | source table, API collection, basic noise checks | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/02_web_scraping_sources.ipynb) |
| `03_ocr_udmurt_commons.ipynb` | OCR baseline on an Udmurt Wikimedia Commons scan | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/03_ocr_udmurt_commons.ipynb) |
| `04_asr_udmurt_whisper_tiny.ipynb` | ASR baseline on open Udmurt audio | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/04_asr_udmurt_whisper_tiny.ipynb) |
| `05_corpus_cleaning_datacard.ipynb` | mini-corpus cleaning, splits, data card | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/05_corpus_cleaning_datacard.ipynb) |
| `06_parallel_alignment_mt.ipynb` | Tatar-Russian alignment baseline | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/06_parallel_alignment_mt.ipynb) |
| `07_spellchecker_normalization.ipynb` | frequency dictionary and edit-distance spellchecker | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/07_spellchecker_normalization.ipynb) |
| `08_final_package_review.ipynb` | final package checklist and review | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/08_final_package_review.ipynb) |

Main open sources used or referenced:

- Wikimedia Commons category `Udmurt Dunne` and file `Удномер.jpg`
- Wikimedia Commons file `Udmurt.ogg`
- Wikipedia API language editions: `udm`, `kv`, `sah`, `tt`, `mhr`
- Hugging Face dataset `AigizK/tatar-russian-parallel-corpora`
- Tatoeba downloads as an optional source for sentence pairs
- Mozilla Common Voice as an optional source for ASR data when access/terms are suitable
"""


def main():
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (OUT / name).write_text(json.dumps(nb(cells), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(f"wrote {len(NOTEBOOKS)} notebooks to {OUT}")


if __name__ == "__main__":
    main()
