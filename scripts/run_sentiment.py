#!/usr/bin/env python3
"""Run VADER sentiment analysis on interview data.

Usage:
    python3.11 scripts/run_sentiment.py

Reads:  data/interviews.csv
Writes: outputs/interviews_sentiment/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "interviews_sentiment"


def label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / "data" / "interviews.csv")
    analyzer = SentimentIntensityAnalyzer()

    scores = df["text"].map(analyzer.polarity_scores)
    df["compound"] = scores.map(lambda s: round(s["compound"], 4))
    df["positive"]  = scores.map(lambda s: round(s["pos"], 4))
    df["negative"]  = scores.map(lambda s: round(s["neg"], 4))
    df["neutral"]   = scores.map(lambda s: round(s["neu"], 4))
    df["sentiment"] = df["compound"].map(label)

    df.to_csv(OUTPUT_DIR / "sentiment_scores.csv", index=False)

    # ── Chart 1: distribution of compound scores ───────────────────────────────
    fig = px.histogram(
        df,
        x="compound",
        color="sentiment",
        nbins=20,
        title="Distribution of Compound Sentiment Scores (50 paragraphs)",
        labels={"compound": "Compound score (−1 negative → +1 positive)", "count": "Paragraphs"},
        color_discrete_map={"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"},
        category_orders={"sentiment": ["positive", "neutral", "negative"]},
    )
    fig.write_html(viz_dir / "sentiment_distribution.html", include_plotlyjs="cdn")

    # ── Chart 2: box plot per interviewee ──────────────────────────────────────
    fig2 = px.box(
        df,
        x="interviewee",
        y="compound",
        color="interviewee",
        points="all",
        title="Compound Score by Interviewee",
        labels={"compound": "Compound score", "interviewee": "Interviewee"},
    )
    fig2.add_hline(y=0.05,  line_dash="dot", line_color="#2ecc71",
                   annotation_text="positive threshold (+0.05)")
    fig2.add_hline(y=-0.05, line_dash="dot", line_color="#e74c3c",
                   annotation_text="negative threshold (−0.05)")
    fig2.update_layout(showlegend=False)
    fig2.write_html(viz_dir / "sentiment_by_interviewee.html", include_plotlyjs="cdn")

    # ── Print summary ──────────────────────────────────────────────────────────
    counts = df["sentiment"].value_counts()
    avg = df.groupby("interviewee")["compound"].mean().sort_values()

    print(f"\nSentiment summary — {len(df)} paragraphs")
    print(f"  Positive  (≥ +0.05) : {counts.get('positive', 0)}")
    print(f"  Neutral              : {counts.get('neutral', 0)}")
    print(f"  Negative  (≤ −0.05) : {counts.get('negative', 0)}")
    print(f"\nAverage compound score by interviewee:")
    for name, score in avg.items():
        print(f"  {name:<8} {score:+.3f}")
    print(f"\nResults written to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
