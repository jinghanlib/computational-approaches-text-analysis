#!/usr/bin/env python3
"""Convert raw interview .txt transcripts to a single structured CSV.

Usage:
    python scripts/prepare_interviews.py

Reads:  data/Alex.txt, Ben.txt, Gina.txt, Linda.txt, Margot.txt, Otto.txt
Writes: data/interviews.csv  (one row per paragraph)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "interviews.csv"

NAMES = ["Alex", "Ben", "Gina", "Linda", "Margot", "Otto"]


def parse_txt(path: Path, name: str) -> list[dict]:
    """Return paragraph-level rows from a transcript file."""
    raw = path.read_text(encoding="utf-8-sig")  # strips BOM automatically
    rows: list[dict] = []
    para_num = 0

    for block in re.split(r"\n{2,}", raw):
        lines: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            # Drop pure interviewer prompts
            if re.match(r"^Interviewer\s*:", line, re.IGNORECASE):
                continue
            # Strip "Interviewee: " label, keep the spoken text
            line = re.sub(r"^Interviewee\s*:\s*", "", line, flags=re.IGNORECASE)
            lines.append(line)

        text = " ".join(lines).strip()
        if not text:
            continue

        para_num += 1
        rows.append(
            {
                "interview_id": f"{name.upper()[:4]}-{para_num:03d}",
                "interviewee": name,
                "paragraph_num": para_num,
                "text": text,
            }
        )

    return rows


def main() -> None:
    all_rows: list[dict] = []

    for name in NAMES:
        path = DATA_DIR / f"{name}.txt"
        if not path.exists():
            print(f"  Warning: {path.name} not found — skipping")
            continue
        rows = parse_txt(path, name)
        all_rows.extend(rows)
        print(f"  {name}: {len(rows)} paragraphs")

    with open(OUTPUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["interview_id", "interviewee", "paragraph_num", "text"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows → {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
