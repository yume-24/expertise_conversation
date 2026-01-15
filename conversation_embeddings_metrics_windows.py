#!/usr/bin/env python3
"""
conversation_embeddings_metrics_windows.py (rewritten)

Lexical trigram embeddings:
- Extract 3-word trigrams from utterances (in sequence order)
- Embed trigrams
- Reuse all the same conversation metrics from conversation_embedding_metrics.py

Usage:
  python conversation_embeddings_metrics_windows.py --input path/to/file_or_dir --output trigram_metrics.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Reuse your existing metric implementation
# Make sure conversation_embedding_metrics.py is in the same folder or on PYTHONPATH
from conversation_embedding_metrics import compute_conversation_metrics, iter_csv_files


# ---------------------------
# Trigram extraction helpers
# ---------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str) -> List[str]:
    """Simple, fast tokenizer: keeps alphanumerics + apostrophes, lowercases."""
    if text is None:
        return []
    return _WORD_RE.findall(str(text).lower())

def utterance_to_trigrams(tokens: List[str]) -> List[str]:
    """Return list of trigram strings like 'this is a'."""
    if len(tokens) < 3:
        return []
    return [" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)]

def expand_df_to_trigram_stream(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    speaker_col: str = "Speaker",
    utterance_col: str = "Utterance",
    min_chars: int = 1,
) -> pd.DataFrame:
    """
    Turn original utterance rows into a trigram-level stream:
    - Preserves conversation order by expanding each utterance into multiple trigram rows.
    - Creates new integer Sequence values so sorting is stable.
    """
    required = {sequence_col, speaker_col, utterance_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()

    # coerce types
    d[utterance_col] = d[utterance_col].fillna("").astype(str)
    d[speaker_col] = d[speaker_col].fillna("").astype(str)

    # Sequence should be numeric-sortable; coerce then sort
    d[sequence_col] = pd.to_numeric(d[sequence_col], errors="coerce")
    d = d.dropna(subset=[sequence_col])
    d = d.sort_values(sequence_col).reset_index(drop=True)

    out_rows = []
    # Multiply original sequence so per-utterance trigram index can be appended without collisions
    # (e.g., seq=57 -> 57000 + k)
    SEQ_MULT = 1000

    for _, row in d.iterrows():
        seq = int(row[sequence_col])
        spk = row[speaker_col]
        utt = row[utterance_col].strip()

        if len(utt) < min_chars:
            continue

        toks = tokenize(utt)
        tris = utterance_to_trigrams(toks)
        if not tris:
            continue

        for k, tri in enumerate(tris):
            out_rows.append({
                "Sequence": seq * SEQ_MULT + k,
                "Speaker": spk,
                "Utterance": tri,
            })

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        # Return empty with required cols so caller can handle
        return pd.DataFrame(columns=["Sequence", "Speaker", "Utterance"])

    out_df = out_df.sort_values("Sequence").reset_index(drop=True)
    return out_df


# ---------------------------
# CLI / batch
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="CSV file or directory containing CSVs")
    ap.add_argument("--output", type=str, required=True, help="Output CSV path for metrics")
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")

    # Optional: force the two speakers the metric function keeps
    ap.add_argument("--speaker-a", type=str, default=None, help="Optional: force Speaker A name")
    ap.add_argument("--speaker-b", type=str, default=None, help="Optional: force Speaker B name")
    ap.add_argument("--k", type=int, default=None, help="Optional: number of clusters for entropy/MI")

    # Trigram extraction controls
    ap.add_argument("--min-chars", type=int, default=1, help="Skip utterances shorter than this many characters")
    args = ap.parse_args()

    input_path = Path(args.input)
    files = iter_csv_files(input_path)
    if not files:
        raise SystemExit("No CSV files found.")

    # Load model once (compute_conversation_metrics creates/uses it internally)
    # compute_conversation_metrics expects a SentenceTransformer instance
    from sentence_transformers import SentenceTransformer
    print(f"Found {len(files)} file(s). Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    rows = []
    for f in files:
        convo_id = f.stem
        try:
            raw = pd.read_csv(f)

            tri_df = expand_df_to_trigram_stream(
                raw,
                sequence_col="Sequence",
                speaker_col="Speaker",
                utterance_col="Utterance",
                min_chars=args.min_chars,
            )

            if len(tri_df) < 4:
                raise ValueError(
                    f"{convo_id}: too few trigram points after expansion "
                    f"({len(tri_df)}). (Maybe very short B speaker, or tokenizer filtered too hard.)"
                )

            row = compute_conversation_metrics(
                df=tri_df,
                convo_id=convo_id,
                model=model,
                speaker_a=args.speaker_a,
                speaker_b=args.speaker_b,
                k_clusters=args.k,
            )
            rows.append(row)
            print(f"✓ {convo_id} (trigrams={int(row.get('n_utterances', np.nan))})")

        except Exception as e:
            print(f"✗ {f.name}: {e}")

    out_df = pd.DataFrame(rows)

    # Stable column order: identifiers first
    front = ["convo_id", "speaker_A", "speaker_B", "n_utterances", "n_A", "n_B"]
    cols = front + [c for c in out_df.columns if c not in front]
    out_df = out_df[cols]

    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics to: {out_path.resolve()}")
    print(out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
