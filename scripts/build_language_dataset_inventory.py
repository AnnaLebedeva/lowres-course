import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/ann.lebedeva/Documents/lowres")
OUT = ROOT / "generated_datasets"
OUT.mkdir(exist_ok=True)

OPUS_API = "https://opus.nlpl.eu/opusapi"


LANGUAGES = [
    # Turkic
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
    # Uralic
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
    # North Caucasian
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
    # Mongolic and Tungusic
    {"language_ru": "бурятский", "language_en": "Buryat", "family": "Монгольская", "branch": "монгольская", "iso639_3": "bxr", "opus_code": "bxr", "wiki_code": "bxr"},
    {"language_ru": "калмыцкий", "language_en": "Kalmyk", "family": "Монгольская", "branch": "ойратская", "iso639_3": "xal", "opus_code": "xal", "wiki_code": "xal"},
    {"language_ru": "эвенкийский", "language_en": "Evenki", "family": "Тунгусо-маньчжурская", "branch": "тунгусская", "iso639_3": "evn", "opus_code": "evn", "wiki_code": None},
    {"language_ru": "нанайский", "language_en": "Nanai", "family": "Тунгусо-маньчжурская", "branch": "тунгусская", "iso639_3": "gld", "opus_code": "gld", "wiki_code": None},
    # Other families
    {"language_ru": "нивхский", "language_en": "Nivkh", "family": "изолят / палеоазиатская группа", "branch": "нивхская", "iso639_3": "niv", "opus_code": None, "wiki_code": None},
    {"language_ru": "чукотский", "language_en": "Chukchi", "family": "чукотско-камчатская", "branch": "чукотская", "iso639_3": "ckt", "opus_code": None, "wiki_code": None},
    {"language_ru": "корякский", "language_en": "Koryak", "family": "чукотско-камчатская", "branch": "чукотская", "iso639_3": "kpy", "opus_code": None, "wiki_code": None},
    {"language_ru": "алеутский", "language_en": "Aleut", "family": "эскимосско-алеутская", "branch": "алеутская", "iso639_3": "ale", "opus_code": "ale", "wiki_code": None},
    {"language_ru": "эскимосский / юпик", "language_en": "Yupik", "family": "эскимосско-алеутская", "branch": "эскимосская", "iso639_3": "ess", "opus_code": None, "wiki_code": None},
]


def api_json(url, params, attempts=3):
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    last_error = None
    for attempt in range(attempts):
        req = urllib.request.Request(full_url, headers={"User-Agent": "lowres-course-dataset-scout/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt * 2)
    raise last_error


def opus_summary(opus_code):
    if not opus_code:
        return {
            "opus_checked": False,
            "opus_ru_parallel_pairs": 0,
            "opus_ru_parallel_documents": 0,
            "opus_ru_parallel_corpora": "",
            "opus_mono_pairs_or_segments": 0,
            "opus_mono_documents": 0,
            "opus_mono_corpora": "",
        }

    data = api_json(OPUS_API, {
        "source": "ru",
        "target": opus_code,
        "preprocessing": "xml",
        "version": "latest",
    })
    corpora = data.get("corpora", [])
    parallel = [
        c for c in corpora
        if {c.get("source"), c.get("target")} == {"ru", opus_code}
    ]
    mono = [
        c for c in corpora
        if c.get("source") == opus_code and not c.get("target")
    ]

    def as_int(value):
        if value in ("", None):
            return 0
        return int(value)

    return {
        "opus_checked": True,
        "opus_ru_parallel_pairs": sum(as_int(c.get("alignment_pairs")) for c in parallel),
        "opus_ru_parallel_documents": sum(as_int(c.get("documents")) for c in parallel),
        "opus_ru_parallel_corpora": "; ".join(
            f"{c.get('corpus')} ({c.get('alignment_pairs') or 0})" for c in parallel
        ),
        "opus_mono_pairs_or_segments": sum(as_int(c.get("alignment_pairs")) for c in mono),
        "opus_mono_documents": sum(as_int(c.get("documents")) for c in mono),
        "opus_mono_corpora": "; ".join(
            f"{c.get('corpus')} ({c.get('alignment_pairs') or 0})" for c in mono
        ),
    }


def wiki_summary(wiki_code):
    if not wiki_code:
        return {"wiki_checked": False, "wiki_articles": "", "wiki_pages": "", "wiki_source_url": ""}
    try:
        data = api_json(f"https://{wiki_code}.wikipedia.org/w/api.php", {
            "action": "query",
            "meta": "siteinfo",
            "siprop": "statistics",
            "format": "json",
        })
        stats = data.get("query", {}).get("statistics", {})
        return {
            "wiki_checked": True,
            "wiki_articles": stats.get("articles", ""),
            "wiki_pages": stats.get("pages", ""),
            "wiki_source_url": f"https://{wiki_code}.wikipedia.org/",
        }
    except Exception as exc:
        return {
            "wiki_checked": False,
            "wiki_articles": "",
            "wiki_pages": "",
            "wiki_source_url": f"https://{wiki_code}.wikipedia.org/",
            "wiki_error": str(exc),
        }


def build_inventory():
    rows = []
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for item in LANGUAGES:
        print("checking", item["language_ru"])
        row = dict(item)
        row.update(opus_summary(item.get("opus_code")))
        row.update(wiki_summary(item.get("wiki_code")))
        row["parallel_with_russian_source"] = "https://opus.nlpl.eu/opusapi"
        row["monolingual_source"] = "OPUS monolingual rows; Wikipedia statistics where available"
        row["checked_at_utc"] = checked_at
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = build_inventory()
    df = df.sort_values(["family", "branch", "language_ru"]).reset_index(drop=True)
    csv_path = OUT / "russia_languages_dataset_inventory.csv"
    xlsx_path = OUT / "russia_languages_dataset_inventory.xlsx"
    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="inventory", index=False)
        meta = pd.DataFrame([
            {"field": "scope", "value": "Initial course inventory: major living languages of peoples of Russia, no dialect-level coverage."},
            {"field": "sources", "value": "OPUS API; Wikipedia siteinfo statistics."},
            {"field": "checked_at_utc", "value": df["checked_at_utc"].iloc[0] if len(df) else ""},
            {"field": "caveat", "value": "Counts are API snapshots and need linguistic/community review before publication."},
        ])
        meta.to_excel(writer, sheet_name="metadata", index=False)
    print(csv_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
