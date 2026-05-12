# Computational Approaches to Text Analysis

Workshop materials for a three-hour introduction to computational text analysis methods. The workshop covers four approaches in sequence — manual coding, word frequency analysis, sentiment analysis, and topic modeling — all applied to the same interview dataset.

**Live site:** https://jinghanlib.github.io/computational-approaches-text-analysis/

## Workshop Sections

| Section | Method | Tool |
|---|---|---|
| 1 · Software QDA | Manual coding | Taguette (in class), QualCoder (take-home) |
| 2 · Word Counting | Frequency analysis | Voyant Tools |
| 3 · Sentiment Analysis | Rule-based scoring | VADER |
| 4 · Topic Modeling | Neural topic modeling | BERTopic + Ollama |

Each section has a conceptual overview page and a hands-on exercise page. The exercises are designed to work independently: students can complete them in sequence during the workshop or return to any section on their own.

## Dataset

50 interview paragraphs from six digital influencers (Alex, Ben, Gina, Linda, Margot, Otto) discussing ethics, sponsorship transparency, and audience relationships. Originally published by Renata Gonçalves Curty (UCSB) at [zenodo.org/records/18604273](https://zenodo.org/records/18604273); reformatted as CSV for use with the analysis scripts.

## Repository Structure

```
data/               Interview data (CSV and raw transcripts)
instructions/       QMD source files for each workshop page
outputs/            Pre-computed results (sentiment and topic modeling)
scripts/            Python scripts for running the analysis pipeline
docs/               Rendered HTML site (served via GitHub Pages)
slides.qmd          Reveal.js workshop slides
```

## Scripts

| Script | Description |
|---|---|
| `scripts/prepare_interviews.py` | Converts raw `.txt` transcripts to `data/interviews.csv` |
| `scripts/run_sentiment.py` | Runs VADER on `interviews.csv`, writes results to `outputs/interviews_sentiment/` |
| `scripts/run_bertopic.py` | Runs BERTopic pipeline (nomic-embed-text + HDBSCAN + llama3.1), writes results to `outputs/interviews_bertopic/` |

## Requirements

Python 3.11 and [Ollama](https://ollama.com) with `nomic-embed-text` and `llama3.1` pulled locally.

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.1
```

## Run the Pipeline

```bash
python3.11 scripts/run_sentiment.py
python3.11 scripts/run_bertopic.py
```

Pre-computed results are already included in `outputs/` and `docs/outputs/` for students who prefer to work with prepared data.
