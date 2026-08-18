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
import os, re, json, textwrap, math, statistics, random, io, time
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

Если Wikimedia временно отвечает `429 Too Many Requests`, подождите минуту и перезапустите ячейку. В тетрадке запросы сгруппированы, но публичные API все равно иногда ограничивают Colab-адреса.
"""),
        code("""
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
UDM_WIKI_API = 'https://udm.wikipedia.org/w/api.php'

COMMONS_CATEGORY = 'Category:Udmurt Dunne'
LANGUAGE = 'удмуртский'
FALLBACK_FILE_TITLE = 'File:Удномер.jpg'
RASTER_MIME_TYPES = {'image/jpeg', 'image/png', 'image/tiff', 'image/webp'}

def api_get(url, params, timeout=30, attempts=3):
    last_error = None
    for attempt in range(attempts):
        r = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={'User-Agent': 'lowres-course-colab/1.0 (contact: teaching-notebook)'},
        )
        if r.status_code == 429 and attempt < attempts - 1:
            wait = int(r.headers.get('Retry-After', 2 + attempt * 2))
            print(f'Wikimedia API rate limit: ждем {wait} сек. и пробуем еще раз')
            time.sleep(wait)
            continue
        try:
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2 + attempt * 2)
    raise last_error

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

def commons_imageinfo_batch(titles):
    if not titles:
        return []
    data = api_get(COMMONS_API, {
        'action': 'query',
        'titles': '|'.join(titles),
        'prop': 'imageinfo',
        'iiprop': 'url|mime|size|extmetadata',
        'format': 'json',
    })
    by_title = {}
    for page in data.get('query', {}).get('pages', {}).values():
        title = page.get('title')
        info = page.get('imageinfo', [{}])[0]
        meta = info.get('extmetadata', {})
        by_title[title] = {
            'title': title,
            'url': info.get('url'),
            'mime': info.get('mime'),
            'width': info.get('width'),
            'height': info.get('height'),
            'license': meta.get('LicenseShortName', {}).get('value'),
            'artist': meta.get('Artist', {}).get('value'),
            'description': meta.get('ImageDescription', {}).get('value'),
        }
    return [by_title[t] for t in titles if t in by_title]

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
    titles = [item['title'] for item in state['commons_files']]
    inspected = commons_imageinfo_batch(titles)
    selected = None
    for info in inspected:
        if is_raster_image(info) and selected is None:
            selected = info
    if selected is None:
        fallback = commons_imageinfo(FALLBACK_FILE_TITLE)
        inspected.append(fallback)
        selected = fallback
    state['selected_file'] = selected
    state['inspected_files'] = inspected
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
    for candidate in state.get('inspected_files', []):
        if candidate.get('title') != selected_title:
            candidates.append(candidate)

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
    inspected_files: List[Dict[str, Any]]
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
    return {'selected_file': next_state['selected_file'], 'inspected_files': next_state['inspected_files']}

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


def lesson01_dataset_scout_cells():
    languages_code = repr([
        {"language_ru": "татарский", "language_en": "Tatar", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "tat", "opus_code": "tt", "wiki_code": "tt"},
        {"language_ru": "башкирский", "language_en": "Bashkir", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "bak", "opus_code": "ba", "wiki_code": "ba"},
        {"language_ru": "чувашский", "language_en": "Chuvash", "family": "Тюркская", "branch": "огурская", "iso639_3": "chv", "opus_code": "chv", "wiki_code": "cv"},
        {"language_ru": "якутский / саха", "language_en": "Sakha / Yakut", "family": "Тюркская", "branch": "сибирская", "iso639_3": "sah", "opus_code": "sah", "wiki_code": "sah"},
        {"language_ru": "тувинский", "language_en": "Tuvan", "family": "Тюркская", "branch": "сибирская", "iso639_3": "tyv", "opus_code": "tyv", "wiki_code": "tyv"},
        {"language_ru": "хакасский", "language_en": "Khakas", "family": "Тюркская", "branch": "сибирская", "iso639_3": "kjh", "opus_code": "kjh", "wiki_code": None},
        {"language_ru": "алтайский", "language_en": "Altai", "family": "Тюркская", "branch": "сибирская", "iso639_3": "alt", "opus_code": "alt", "wiki_code": "alt"},
        {"language_ru": "кумыкский", "language_en": "Kumyk", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "kum", "opus_code": "kum", "wiki_code": None},
        {"language_ru": "карачаево-балкарский", "language_en": "Karachay-Balkar", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "krc", "opus_code": "krc", "wiki_code": "krc"},
        {"language_ru": "ногайский", "language_en": "Nogai", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "nog", "opus_code": "nog", "wiki_code": None},
        {"language_ru": "крымскотатарский", "language_en": "Crimean Tatar", "family": "Тюркская", "branch": "кыпчакско-огузская", "iso639_3": "crh", "opus_code": "crh", "wiki_code": "crh"},
        {"language_ru": "удмуртский", "language_en": "Udmurt", "family": "Уральская", "branch": "пермская", "iso639_3": "udm", "opus_code": "udm", "wiki_code": "udm"},
        {"language_ru": "коми-зырянский", "language_en": "Komi-Zyrian", "family": "Уральская", "branch": "пермская", "iso639_3": "kpv", "opus_code": "kpv", "wiki_code": "kv"},
        {"language_ru": "коми-пермяцкий", "language_en": "Komi-Permyak", "family": "Уральская", "branch": "пермская", "iso639_3": "koi", "opus_code": "koi", "wiki_code": "koi"},
        {"language_ru": "эрзянский", "language_en": "Erzya", "family": "Уральская", "branch": "мордовская", "iso639_3": "myv", "opus_code": "myv", "wiki_code": "myv"},
        {"language_ru": "мокшанский", "language_en": "Moksha", "family": "Уральская", "branch": "мордовская", "iso639_3": "mdf", "opus_code": "mdf", "wiki_code": "mdf"},
        {"language_ru": "марийский луговой", "language_en": "Meadow Mari", "family": "Уральская", "branch": "марийская", "iso639_3": "mhr", "opus_code": "mhr", "wiki_code": "mhr"},
        {"language_ru": "марийский горный", "language_en": "Hill Mari", "family": "Уральская", "branch": "марийская", "iso639_3": "mrj", "opus_code": "mrj", "wiki_code": "mrj"},
        {"language_ru": "карельский", "language_en": "Karelian", "family": "Уральская", "branch": "прибалтийско-финская", "iso639_3": "krl", "opus_code": "krl", "wiki_code": "krl"},
        {"language_ru": "вепсский", "language_en": "Veps", "family": "Уральская", "branch": "прибалтийско-финская", "iso639_3": "vep", "opus_code": "vep", "wiki_code": "vep"},
        {"language_ru": "хантыйский", "language_en": "Khanty", "family": "Уральская", "branch": "угорская", "iso639_3": "kca", "opus_code": None, "wiki_code": None},
        {"language_ru": "мансийский", "language_en": "Mansi", "family": "Уральская", "branch": "угорская", "iso639_3": "mns", "opus_code": "mns", "wiki_code": None},
        {"language_ru": "ненецкий", "language_en": "Nenets", "family": "Уральская", "branch": "самодийская", "iso639_3": "yrk", "opus_code": "yrk", "wiki_code": None},
        {"language_ru": "чеченский", "language_en": "Chechen", "family": "Северокавказская", "branch": "нахская", "iso639_3": "che", "opus_code": "ce", "wiki_code": "ce"},
        {"language_ru": "ингушский", "language_en": "Ingush", "family": "Северокавказская", "branch": "нахская", "iso639_3": "inh", "opus_code": "inh", "wiki_code": "inh"},
        {"language_ru": "аварский", "language_en": "Avar", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "ava", "opus_code": "av", "wiki_code": "av"},
        {"language_ru": "даргинский", "language_en": "Dargwa", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "dar", "opus_code": "dar", "wiki_code": None},
        {"language_ru": "лезгинский", "language_en": "Lezgian", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "lez", "opus_code": "lez", "wiki_code": "lez"},
        {"language_ru": "лакский", "language_en": "Lak", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "lbe", "opus_code": "lbe", "wiki_code": "lbe"},
        {"language_ru": "рутульский", "language_en": "Rutul", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "rut", "opus_code": "rut", "wiki_code": None},
        {"language_ru": "адыгейский", "language_en": "Adyghe", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "ady", "opus_code": "ady", "wiki_code": "ady"},
        {"language_ru": "кабардино-черкесский", "language_en": "Kabardian", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "kbd", "opus_code": "kbd", "wiki_code": "kbd"},
        {"language_ru": "абазинский", "language_en": "Abaza", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "abq", "opus_code": None, "wiki_code": None},
        {"language_ru": "бурятский", "language_en": "Buryat", "family": "Монгольская", "branch": "монгольская", "iso639_3": "bxr", "opus_code": "bxr", "wiki_code": "bxr"},
        {"language_ru": "калмыцкий", "language_en": "Kalmyk", "family": "Монгольская", "branch": "ойратская", "iso639_3": "xal", "opus_code": "xal", "wiki_code": "xal"},
        {"language_ru": "эвенкийский", "language_en": "Evenki", "family": "Тунгусо-маньчжурская", "branch": "тунгусская", "iso639_3": "evn", "opus_code": "evn", "wiki_code": None},
        {"language_ru": "нанайский", "language_en": "Nanai", "family": "Тунгусо-маньчжурская", "branch": "тунгусская", "iso639_3": "gld", "opus_code": "gld", "wiki_code": None},
        {"language_ru": "нивхский", "language_en": "Nivkh", "family": "изолят / палеоазиатская группа", "branch": "нивхская", "iso639_3": "niv", "opus_code": None, "wiki_code": None},
        {"language_ru": "чукотский", "language_en": "Chukchi", "family": "чукотско-камчатская", "branch": "чукотская", "iso639_3": "ckt", "opus_code": None, "wiki_code": None},
        {"language_ru": "корякский", "language_en": "Koryak", "family": "чукотско-камчатская", "branch": "чукотская", "iso639_3": "kpy", "opus_code": None, "wiki_code": None},
        {"language_ru": "алеутский", "language_en": "Aleut", "family": "эскимосско-алеутская", "branch": "алеутская", "iso639_3": "ale", "opus_code": "ale", "wiki_code": None},
        {"language_ru": "эскимосский / юпик", "language_en": "Yupik", "family": "эскимосско-алеутская", "branch": "эскимосская", "iso639_3": "ess", "opus_code": None, "wiki_code": None},
    ])
    return [
        md("""
# Занятие 1. Агент первичной разведки языковых данных

**Цель практики:** не OCR и не ASR, а стартовая карта проекта: какие языки берем в поле зрения и какие открытые данные уже можно найти.

Агент в этой тетрадке собирает первоначальную информацию:

1. берет редактируемый seed list основных живых языков народов России без диалектального уровня;
2. проверяет OPUS API на параллельные данные с русским;
3. проверяет OPUS на моноязычные строки/сегменты;
4. проверяет наличие языковой Википедии и ее размер;
5. собирает таблицу, которую можно открыть в Google Sheets и дальше править руками.

Готовый снапшот этой таблицы уже создан в Google Sheets: https://docs.google.com/spreadsheets/d/1XIW9BCxs4ENsQUhiK1HYGzpA1aLiYcwOOOfnmZNtcB8
"""),
        code("""
!pip -q install langgraph pandas requests openpyxl
"""),
        code(COMMON_SETUP + """
from typing import Any, Dict, List, TypedDict
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
"""),
        md("""
## 1. Seed list языков

Это не “истина навсегда”, а стартовая рабочая рамка для курса. Ее надо обсуждать и уточнять: какие языки добавить, где объединять варианты, где наоборот нельзя смешивать разные языковые сообщества.
"""),
        code(f"""
LANGUAGES = {languages_code}

seed_df = pd.DataFrame(LANGUAGES)
display(seed_df.groupby(['family', 'branch']).size().reset_index(name='languages'))
display(seed_df.head(12))
"""),
        md("## 2. Инструменты агента: OPUS API и Wikipedia API"),
        code("""
OPUS_API = 'https://opus.nlpl.eu/opusapi'

def api_get(url, params, attempts=3):
    last_error = None
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=30, headers={'User-Agent': 'lowres-course-dataset-scout/1.0'})
            if r.status_code == 429 and attempt < attempts - 1:
                wait = int(r.headers.get('Retry-After', 2 + attempt * 2))
                print('rate limit, wait', wait, 'sec')
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1 + attempt * 2)
    raise last_error

def as_int(value):
    if value in ('', None):
        return 0
    return int(value)

def query_opus_for_language(opus_code):
    if not opus_code:
        return {
            'opus_checked': False,
            'opus_ru_parallel_pairs': 0,
            'opus_ru_parallel_documents': 0,
            'opus_ru_parallel_corpora': '',
            'opus_mono_pairs_or_segments': 0,
            'opus_mono_documents': 0,
            'opus_mono_corpora': '',
        }
    data = api_get(OPUS_API, {
        'source': 'ru',
        'target': opus_code,
        'preprocessing': 'xml',
        'version': 'latest',
    })
    corpora = data.get('corpora', [])
    parallel = [c for c in corpora if {c.get('source'), c.get('target')} == {'ru', opus_code}]
    mono = [c for c in corpora if c.get('source') == opus_code and not c.get('target')]
    return {
        'opus_checked': True,
        'opus_ru_parallel_pairs': sum(as_int(c.get('alignment_pairs')) for c in parallel),
        'opus_ru_parallel_documents': sum(as_int(c.get('documents')) for c in parallel),
        'opus_ru_parallel_corpora': '; '.join(f"{c.get('corpus')} ({c.get('alignment_pairs') or 0})" for c in parallel),
        'opus_mono_pairs_or_segments': sum(as_int(c.get('alignment_pairs')) for c in mono),
        'opus_mono_documents': sum(as_int(c.get('documents')) for c in mono),
        'opus_mono_corpora': '; '.join(f"{c.get('corpus')} ({c.get('alignment_pairs') or 0})" for c in mono),
    }

def query_wikipedia_for_language(wiki_code):
    if not wiki_code:
        return {'wiki_checked': False, 'wiki_articles': '', 'wiki_pages': '', 'wiki_source_url': ''}
    try:
        data = api_get(f'https://{wiki_code}.wikipedia.org/w/api.php', {
            'action': 'query',
            'meta': 'siteinfo',
            'siprop': 'statistics',
            'format': 'json',
        })
        stats = data.get('query', {}).get('statistics', {})
        return {
            'wiki_checked': True,
            'wiki_articles': stats.get('articles', ''),
            'wiki_pages': stats.get('pages', ''),
            'wiki_source_url': f'https://{wiki_code}.wikipedia.org/',
        }
    except Exception as exc:
        return {
            'wiki_checked': False,
            'wiki_articles': '',
            'wiki_pages': '',
            'wiki_source_url': f'https://{wiki_code}.wikipedia.org/',
            'wiki_error': str(exc),
        }
"""),
        md("## 3. Plain Python агент: state, tools, observations, report"),
        code("""
def scout_language(row):
    observation = dict(row)
    observation.update(query_opus_for_language(row.get('opus_code')))
    observation.update(query_wikipedia_for_language(row.get('wiki_code')))
    observation['parallel_with_russian_source'] = 'https://opus.nlpl.eu/opusapi'
    observation['monolingual_source'] = 'OPUS monolingual rows; Wikipedia statistics where available'
    observation['checked_at_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return observation

def run_dataset_scout(languages):
    state = {
        'goal': 'собрать первичную карту языков и открытых датасетов',
        'sources': ['OPUS API', 'Wikipedia siteinfo API'],
        'languages_total': len(languages),
        'observations': [],
        'errors': [],
    }
    for i, lang in enumerate(languages, 1):
        print(f"[{i}/{len(languages)}] {lang['language_ru']}")
        try:
            state['observations'].append(scout_language(lang))
        except Exception as exc:
            state['errors'].append({'language_ru': lang['language_ru'], 'error': str(exc)})
    inventory = pd.DataFrame(state['observations'])
    inventory = inventory.sort_values(['family', 'branch', 'language_ru']).reset_index(drop=True)
    state['inventory'] = inventory
    state['summary'] = {
        'languages_total': len(languages),
        'languages_checked': len(inventory),
        'with_opus_ru_parallel': int((inventory['opus_ru_parallel_pairs'] > 0).sum()),
        'with_wikipedia': int((inventory['wiki_checked'] == True).sum()),
        'errors': len(state['errors']),
    }
    return state

plain_state = run_dataset_scout(LANGUAGES)
plain_state['summary']
"""),
        code("""
inventory = plain_state['inventory']
display(inventory.head(20))
display(inventory.groupby('family')[['opus_ru_parallel_pairs', 'opus_mono_pairs_or_segments']].sum().sort_values('opus_ru_parallel_pairs', ascending=False))

save_artifact('lesson01_language_dataset_inventory.csv', inventory)
save_artifact('lesson01_dataset_scout_summary.json', json.dumps(plain_state['summary'], ensure_ascii=False, indent=2))
"""),
        md("## 4. Та же логика в LangGraph"),
        code("""
class DatasetScoutState(TypedDict, total=False):
    languages: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    inventory: Any
    summary: Dict[str, Any]

def init_state_node(state: DatasetScoutState):
    return {'observations': [], 'errors': []}

def collect_node(state: DatasetScoutState):
    observations = []
    errors = []
    for lang in state['languages']:
        try:
            observations.append(scout_language(lang))
        except Exception as exc:
            errors.append({'language_ru': lang['language_ru'], 'error': str(exc)})
    return {'observations': observations, 'errors': errors}

def table_node(state: DatasetScoutState):
    inventory = pd.DataFrame(state['observations']).sort_values(['family', 'branch', 'language_ru']).reset_index(drop=True)
    return {'inventory': inventory}

def summary_node(state: DatasetScoutState):
    inventory = state['inventory']
    return {'summary': {
        'languages_total': len(state['languages']),
        'languages_checked': len(inventory),
        'with_opus_ru_parallel': int((inventory['opus_ru_parallel_pairs'] > 0).sum()),
        'with_wikipedia': int((inventory['wiki_checked'] == True).sum()),
        'errors': len(state.get('errors', [])),
    }}

workflow = StateGraph(DatasetScoutState)
workflow.add_node('init', init_state_node)
workflow.add_node('collect', collect_node)
workflow.add_node('table', table_node)
workflow.add_node('summary', summary_node)
workflow.set_entry_point('init')
workflow.add_edge('init', 'collect')
workflow.add_edge('collect', 'table')
workflow.add_edge('table', 'summary')
workflow.add_edge('summary', END)
agent = workflow.compile()

graph_state = agent.invoke({'languages': LANGUAGES})
graph_state['summary']
"""),
        md("""
## 5. Google Sheets

На занятии можно открыть готовый Google Sheet и править его как общий рабочий артефакт:

https://docs.google.com/spreadsheets/d/1XIW9BCxs4ENsQUhiK1HYGzpA1aLiYcwOOOfnmZNtcB8

В Colab эта тетрадка сохраняет CSV в `/content/lowres_lab/lesson01_language_dataset_inventory.csv`. Его можно загрузить в Google Sheets или использовать как основу для обновления общей таблицы.
"""),
        md("""
## 6. Как автоматизировать обновление

Разовый агент полезен для старта, но карта датасетов быстро устаревает: в OPUS появляются новые релизы, в Hugging Face загружают корпуса, национальные проекты открывают новые таблицы, а часть ссылок ломается.

Для этого нужен фоновый агент-монитор:

1. **Scheduler** запускает пайплайн по расписанию: например, раз в неделю или раз в месяц.
2. **Collector** заново обходит источники: OPUS, Wikipedia, Hugging Face, GitHub, национальные корпуса, сайты СМИ и архивов.
3. **State store** хранит предыдущий снимок таблицы: CSV в GitHub, Google Sheet, SQLite или маленький JSON.
4. **Diff checker** сравнивает старую и новую версии: новые языки, новые корпуса, рост/падение counts, ошибки API.
5. **Updater** обновляет Google Sheet только для безопасных полей: counts, даты проверки, ссылки на источники.
6. **Human review** получает спорные изменения: новый источник без понятной лицензии, резкое падение counts, объединение языков/вариантов, изменение классификации.

Самый простой стек для курса:

- `scripts/build_language_dataset_inventory.py` лежит в GitHub;
- GitHub Actions запускает его по cron;
- скрипт сохраняет новый CSV;
- отдельный шаг через Google Sheets API обновляет таблицу;
- если diff большой или появились ошибки, агент создает issue/комментарий для ручной проверки.

Colab для такого расписания не подходит: он хорош для занятия и ручного запуска, но не для надежного фонового мониторинга.
"""),
        code("""
def compare_inventory_snapshots(old_df, new_df):
    key = 'iso639_3'
    old = old_df.set_index(key)
    new = new_df.set_index(key)
    rows = []

    for code in sorted(set(old.index) | set(new.index)):
        if code not in old.index:
            rows.append({'iso639_3': code, 'change_type': 'new_language', 'needs_human_review': True})
            continue
        if code not in new.index:
            rows.append({'iso639_3': code, 'change_type': 'missing_language', 'needs_human_review': True})
            continue

        old_pairs = int(old.loc[code, 'opus_ru_parallel_pairs'])
        new_pairs = int(new.loc[code, 'opus_ru_parallel_pairs'])
        delta = new_pairs - old_pairs
        if delta != 0:
            rows.append({
                'iso639_3': code,
                'language_ru': new.loc[code, 'language_ru'],
                'change_type': 'parallel_count_changed',
                'old_pairs': old_pairs,
                'new_pairs': new_pairs,
                'delta': delta,
                'needs_human_review': abs(delta) > max(1000, old_pairs * 0.5),
            })

    return pd.DataFrame(rows)

# Мини-демо: имитируем, что через месяц OPUS нашел больше параллельных предложений для удмуртского.
old_snapshot = inventory.copy()
new_snapshot = inventory.copy()
new_snapshot.loc[new_snapshot['iso639_3'] == 'udm', 'opus_ru_parallel_pairs'] += 250

diff = compare_inventory_snapshots(old_snapshot, new_snapshot)
display(diff)
save_artifact('lesson01_inventory_diff_demo.csv', diff)
"""),
        md("""
### Пример GitHub Actions расписания

```yaml
name: update-language-dataset-inventory

on:
  schedule:
    - cron: "0 6 1 * *"  # 1 числа каждого месяца
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install pandas openpyxl requests
      - run: python scripts/build_language_dataset_inventory.py
      - name: Update Google Sheet
        run: python scripts/update_google_sheet.py
        env:
          GOOGLE_SERVICE_ACCOUNT_JSON: ${{ secrets.GOOGLE_SERVICE_ACCOUNT_JSON }}
          SPREADSHEET_ID: "1XIW9BCxs4ENsQUhiK1HYGzpA1aLiYcwOOOfnmZNtcB8"
```

В реальном проекте секреты Google API нельзя хранить в notebook. Их кладут в GitHub Secrets, Google Cloud Secret Manager или другой защищенный secret store.
"""),
        md("""
## Вопросы для отчета

1. Какие языковые семьи в таблице оказываются лучше всего покрыты параллельными данными с русским?
2. Где есть Википедия, но почти нет параллельных данных?
3. Где OPUS показывает нули: это значит “данных нет” или “мы не нашли правильный код/источник”?
4. Какие источники надо добавить следующими: национальные корпуса, сайты СМИ, библиотеки, архивы, Hugging Face, GitHub?
5. Какие поля можно обновлять автоматически, а какие требуют human review?
6. Как часто стоит запускать фонового агента для такой таблицы и почему?
"""),
    ]


NOTEBOOKS = {
    "01_agents_for_language_preservation.ipynb": lesson01_dataset_scout_cells(),
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
## 6. OCR как маленький агентский workflow

Этот блок переносит агентскую идею из вводного занятия туда, где ей место: в OCR-задачу. Здесь `state` хранит источник, версии OCR, диагностику и решение о следующем человеческом шаге.
"""),
        code("""
def decide_next_ocr_step(diag_df):
    best = diag_df.sort_values(['tokens_2plus', 'weird_char_share'], ascending=[False, True]).iloc[0]
    if best['tokens_2plus'] < 20:
        return 'Нужен другой скан или ручная разметка маленького ground truth: baseline почти не читает текст.'
    if best['weird_char_share'] > 0.1:
        return 'Нужна дополнительная предобработка и проверка символов.'
    return 'Можно отобрать 30-50 строк для ручной оценки ошибок OCR.'

ocr_agent_state = {
    'task': 'оценить OCR baseline для открытого удмуртского скана',
    'source_file': FILE_TITLE,
    'source_url': image_url,
    'versions': {
        'raw': raw_text,
        'preprocessed': prep_text,
    },
    'diagnostics': diag.to_dict('records'),
    'next_human_step': decide_next_ocr_step(diag),
}

print(json.dumps(ocr_agent_state, ensure_ascii=False, indent=2)[:3000])
save_artifact('lesson03_ocr_agent_state.json', json.dumps(ocr_agent_state, ensure_ascii=False, indent=2))
"""),
        md("""
## Вопросы для отчёта

1. Можно ли читать результат глазами? Какие слова/буквы распознаются хуже всего?
2. Улучшил ли препроцессинг результат?
3. Почему русская OCR-модель может ошибаться на удмуртском?
4. Какое поле `state` оказалось самым важным для решения о следующем шаге?
5. Какие 30-50 строк стоит вручную разметить как ground truth для следующего шага?
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
| `01_agents_for_language_preservation.ipynb` | agentic dataset scouting for languages of Russia using OPUS and Wikipedia APIs | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/01_agents_for_language_preservation.ipynb) |
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
