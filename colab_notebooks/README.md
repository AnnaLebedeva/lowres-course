# Colab notebooks

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
