#!/usr/bin/env python3
"""Run BERTopic on interview data using the same pipeline as the example project.

Pipeline (matches jinghanlib/topic-modeling-slave-narratives exactly):
  nomic-embed-text (Ollama)  →  UMAP  →  HDBSCAN  →  c-TF-IDF  →  llama3.1 labels

Parameters are adjusted for 50 short interview paragraphs rather than
hundreds of long document chunks.

Usage:
    python3.11 scripts/run_bertopic.py
    python3.11 scripts/run_bertopic.py --skip-labels      # skip llama3.1 step
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "interviews_bertopic"

CUSTOM_STOPWORDS = {
    "actually",
    "also",
    "audience",
    "brand",
    "brands",
    "content",
    "definitely",
    "example",
    "feel",
    "followers",
    "honestly",
    "influencer",
    "influencers",
    "just",
    "know",
    "like",
    "lot",
    "make",
    "media",
    "people",
    "post",
    "really",
    "right",
    "said",
    "say",
    "social",
    "thing",
    "things",
    "think",
    "time",
    "want",
    "way",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="Minimum number of paragraphs to form a topic (HDBSCAN min_cluster_size).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="UMAP n_neighbors. Lower values capture finer local structure. "
             "Should be well below the number of documents.",
    )
    parser.add_argument(
        "--ollama-embedding-model",
        default="nomic-embed-text",
        help="Ollama embedding model name.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.1:latest",
        help="Ollama chat model for topic labeling.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip the llama3.1 labeling step (faster, uses c-TF-IDF names only).",
    )
    return parser.parse_args()


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_texts_ollama(texts: list[str], model: str) -> np.ndarray:
    """Embed texts one-by-one via the local Ollama API (matches example script)."""
    all_embeddings: list[list[float]] = []
    for index, text in enumerate(texts, start=1):
        response = requests.post(
            "http://127.0.0.1:11434/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=180,
        )
        response.raise_for_status()
        all_embeddings.append(response.json()["embedding"])
        if index % 10 == 0 or index == len(texts):
            print(f"  Embedded {index}/{len(texts)}")
    return np.array(all_embeddings, dtype=np.float32)


# ── LLM labeling ───────────────────────────────────────────────────────────────

def topic_prompt(topic_id: int, words: list[str], examples: list[str]) -> str:
    """Build the labeling prompt (same structure as example, different domain context)."""
    example_text = "\n\n".join(
        f"Example {i + 1}: {ex}" for i, ex in enumerate(examples)
    )
    return textwrap.dedent(
        f"""
        You are helping label topics from interviews with digital influencers
        about ethics, sponsorship, and audience relationships.
        Use clear, specific wording appropriate for a media studies course.
        Write a human-readable theme title, not a comma-separated list of words.
        Collapse word variants into one idea. For example,
        "sponsored, sponsorship, ad, hashtag, disclose" → "Sponsorship Disclosure and Labeling."

        Topic id: {topic_id}
        Top words: {", ".join(words)}

        Representative passages:
        {example_text}

        Return compact JSON with keys: label, description.
        """
    ).strip()


def call_ollama_model(prompt: str, model: str) -> dict[str, str]:
    """Call local Ollama chat model and parse JSON response (matches example script)."""
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        },
        timeout=180,
    )
    response.raise_for_status()
    content = response.json()["response"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}
    label = str(parsed.get("label", "")).strip()
    description = str(parsed.get("description", "")).strip()
    if not label:
        label = content.strip().splitlines()[0][:80] or "Unlabeled topic"
    if not description:
        description = label
    return {"label": label, "description": description}


def label_topics(topic_model, chunks: pd.DataFrame, topics: list[int], ollama_model: str) -> pd.DataFrame:
    labels: list[dict] = []
    unique_topics = sorted(t for t in set(topics) if t != -1)
    for topic_id in unique_topics:
        words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
        examples = chunks.loc[chunks["topic"] == topic_id, "text"].head(3).tolist()
        examples = [ex[:1200] for ex in examples]
        prompt = topic_prompt(topic_id, words, examples)
        result = call_ollama_model(prompt, ollama_model)
        # Fallback if label is empty or JSON blob
        if not result["label"] or result["label"].startswith("{"):
            result["label"] = ", ".join(words[:4]).title()
        print(f"  Topic {topic_id}: {result['label']}")
        labels.append({"topic": topic_id, **result, "top_words": ", ".join(words)})
    return pd.DataFrame(labels)


# ── Visualizations ─────────────────────────────────────────────────────────────

def save_visualizations(
    topic_model,
    chunks: pd.DataFrame,
    embeddings: np.ndarray,
    viz_dir: Path,
) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    # Document count bar chart (plain, readable — used as the homepage featured chart)
    label_lookup = (
        chunks[chunks["topic"] != -1]
        .groupby("topic")["topic_label"]
        .first()
        .to_dict()
        if "topic_label" in chunks.columns
        else {}
    )
    count_df = (
        chunks[chunks["topic"] != -1]
        .groupby("topic")
        .size()
        .reset_index(name="paragraphs")
        .sort_values("paragraphs", ascending=False)
    )
    count_df["label"] = count_df["topic"].map(lambda t: label_lookup.get(t, f"Topic {t}"))
    fig_counts = px.bar(
        count_df,
        x="label",
        y="paragraphs",
        color="label",
        title="Paragraphs per Topic",
        labels={"label": "Topic", "paragraphs": "Paragraphs assigned"},
    )
    fig_counts.update_layout(showlegend=False, xaxis_tickangle=-20)
    fig_counts.write_html(viz_dir / "topic_counts.html", include_plotlyjs="cdn")
    print("  Saved topic_counts.html")

    # BERTopic built-in charts
    plotters = {
        "topics.html": lambda: topic_model.visualize_topics(),
        "topic_barchart.html": lambda: topic_model.visualize_barchart(),
        "documents.html": lambda: topic_model.visualize_documents(
            chunks["text"].tolist(), embeddings=embeddings
        ),
    }
    for filename, plotter in plotters.items():
        try:
            plotter().write_html(viz_dir / filename, include_plotlyjs="cdn")
            print(f"  Saved {filename}")
        except Exception as exc:
            print(f"  Skipped {filename}: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

    # ── Load data ──────────────────────────────────────────────────────────────
    chunks = pd.read_csv(ROOT / "data" / "interviews.csv")
    print(f"Loaded {len(chunks)} paragraphs from data/interviews.csv")

    # ── Embed ──────────────────────────────────────────────────────────────────
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.ollama_embedding_model)
    embeddings_path = output_dir / f"embeddings_{safe_name}.npy"
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        if len(embeddings) == len(chunks):
            print(f"Loaded cached embeddings from {embeddings_path.name}")
        else:
            print("Cached embeddings are stale — re-embedding")
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print(f"Embedding {len(chunks)} paragraphs with {args.ollama_embedding_model} …")
        embeddings = embed_texts_ollama(chunks["text"].tolist(), args.ollama_embedding_model)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path.name}")

    # ── BERTopic ───────────────────────────────────────────────────────────────
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    # UMAP: n_neighbors tuned down for 50 docs (example uses 8 for ~hundreds)
    umap_model = UMAP(
        n_neighbors=args.n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=args.seed,
    )

    # HDBSCAN: min_cluster_size tuned down for 50 docs (example default is 10)
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # CountVectorizer with custom stopwords (matches example's enhanced cleaning)
    stopwords = sorted(set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS)
    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
    )

    print("Running BERTopic …")
    topics, _ = topic_model.fit_transform(chunks["text"].tolist(), embeddings=embeddings)
    chunks["topic"] = topics

    topic_info = topic_model.get_topic_info()
    n_outliers = int((chunks["topic"] == -1).sum())
    n_topics = int(chunks.loc[chunks["topic"] != -1, "topic"].nunique())
    print(f"Found {n_topics} topics, {n_outliers} outlier paragraphs (topic -1)")

    # ── Save core outputs ──────────────────────────────────────────────────────
    chunks.to_csv(output_dir / "topic_assignments.csv", index=False)
    topic_info.to_csv(output_dir / "topic_info.csv", index=False)

    # ── LLM labels ─────────────────────────────────────────────────────────────
    existing_labels_path = output_dir / "topic_labels_llm.csv"
    if not args.skip_labels:
        print(f"Labeling topics with {args.ollama_model} …")
        labels_df = label_topics(topic_model, chunks, topics, args.ollama_model)
        labels_df.to_csv(existing_labels_path, index=False)
    elif existing_labels_path.exists():
        # Reuse previously generated llama labels — don't overwrite them
        labels_df = pd.read_csv(existing_labels_path)
        print(f"Loaded existing labels from {existing_labels_path.name}")
    else:
        # No labels file and skip requested — fall back to c-TF-IDF names
        labels_df = topic_info[topic_info["Topic"] != -1][["Topic", "Name"]].rename(
            columns={"Topic": "topic", "Name": "label"}
        ).copy()
        labels_df["description"] = labels_df["label"]
        labels_df["top_words"] = labels_df["topic"].map(
            lambda t: ", ".join(w for w, _ in topic_model.get_topic(t)[:10])
        )
        labels_df.to_csv(existing_labels_path, index=False)

    # ── Human-readable review table (matches our existing format) ──────────────
    review_rows: list[dict] = []
    label_lookup = labels_df.set_index("topic")["label"].to_dict()
    for topic_id, group in chunks[chunks["topic"] != -1].groupby("topic"):
        examples = group.sort_values(
            by="topic_score" if "topic_score" in group.columns else group.columns[0]
        ).head(2)
        review_rows.append(
            {
                "topic": topic_id,
                "topic_label": label_lookup.get(topic_id, f"Topic {topic_id}"),
                "documents": len(group),
                "top_words": labels_df.loc[labels_df["topic"] == topic_id, "top_words"].iloc[0]
                if topic_id in labels_df["topic"].values
                else "",
                "interviewees": ", ".join(sorted(group["interviewee"].unique())),
                "example_ids": "; ".join(group["interview_id"].head(2)),
                "example_text": " || ".join(group["text"].head(2).str[:260]),
            }
        )
    review_df = pd.DataFrame(review_rows).sort_values("documents", ascending=False)
    review_df.to_csv(output_dir / "topic_review_table.csv", index=False)

    # Merge labels back into chunks so count chart can show readable names
    chunks["topic_label"] = chunks["topic"].map(label_lookup)

    # ── Visualizations ─────────────────────────────────────────────────────────
    print("Saving visualizations …")
    save_visualizations(topic_model, chunks, embeddings, output_dir / "visualizations")

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"{'Topic':>5}  {'Docs':>4}  {'Interviewees':<30}  Label")
    print(f"{'─' * 60}")
    for _, row in review_df.iterrows():
        print(f"{row['topic']:>5}  {row['documents']:>4}  {row['interviewees']:<30}  {row['topic_label']}")
    if n_outliers:
        print(f"\n  {n_outliers} paragraphs assigned to outlier topic (-1)")
    print(f"{'─' * 60}")
    print(f"Results written to {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
