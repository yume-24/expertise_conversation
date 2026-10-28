"""Canonical transcript ordering and speaker labels from notebook 01."""
import re
import numpy as np
import pandas as pd

def parse_wired_path(path, project_root):
    """
    Example:
    data/wired/wired_astro/wired_astro_12.csv
    """
    folder_name = path.parent.name
    file_stem = path.stem

    video_id = re.sub(r"^wired_", "", folder_name)

    match = re.search(r"_(\d+)$", file_stem)
    file_number = int(match.group(1)) if match else None

    return {
        "dataset": "wired",
        "video_id": video_id,
        "file_number": file_number,
        "conversation_id": file_stem,
        "source_file": str(path.relative_to(project_root))
    }


def count_words(text):
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            str(text),
        )
    )


def create_utterances_table(wired_raw, centroid_min_words=5):
    """
    Create one canonical row per original transcript utterance.

    The function does not concatenate adjacent utterances.
    """

    utterances = (
        wired_raw
        .copy()
        .reset_index(drop=False)
        .rename(columns={"index": "_original_dataframe_row"})
    )

    # Accommodate either the original CSV column names or the
    # standardized names already present in wired_raw.
    rename_map = {}

    aliases = {
        "Sequence": "sequence",
        "Utterance": "text",
        "Speaker": "speaker_raw",
        "Notes": "notes",
    }

    for old_name, new_name in aliases.items():
        if (
            old_name in utterances.columns
            and new_name not in utterances.columns
        ):
            rename_map[old_name] = new_name

    utterances = utterances.rename(columns=rename_map)

    # If canonical speaker has not already been created, use
    # the original speaker label.
    if "speaker" not in utterances.columns:
        utterances["speaker"] = utterances["speaker_raw"]

    # Stable source order.
    if "sequence" not in utterances.columns:
        utterances["sequence"] = (
            utterances
            .groupby(
                ["dataset", "conversation_id"],
                sort=False,
            )
            .cumcount()
            .add(1)
        )

    utterances["_sequence_numeric"] = pd.to_numeric(
        utterances["sequence"],
        errors="coerce",
    )

    utterances["_sequence_numeric"] = (
        utterances["_sequence_numeric"]
        .fillna(utterances["_original_dataframe_row"])
    )

    conversation_keys = [
        "dataset",
        "conversation_id",
    ]

    sort_columns = [
        *conversation_keys,
        "_sequence_numeric",
        "_original_dataframe_row",
    ]

    utterances = (
        utterances
        .sort_values(
            sort_columns,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # Clean text without altering its linguistic contents.
    utterances["text"] = (
        utterances["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    if utterances["text"].eq("").any():
        empty_rows = utterances.loc[
            utterances["text"].eq(""),
            [
                *conversation_keys,
                "sequence",
            ],
        ]

        raise ValueError(
            "Empty utterance text found:\n"
            + empty_rows.to_string(index=False)
        )

    # Reconstruct turn membership without concatenating anything.
    # A new turn begins whenever the speaker changes.
    speaker_changed = (
        utterances
        .groupby(
            conversation_keys,
            sort=False,
        )["speaker"]
        .transform(
            lambda speakers:
                speakers.ne(speakers.shift())
        )
    )

    utterances["turn_id"] = (
        speaker_changed
        .astype(int)
        .groupby(
            [
                utterances[column]
                for column in conversation_keys
            ]
        )
        .cumsum()
    )

    # Position of the utterance inside its speaker turn.
    utterances["utterance_in_turn"] = (
        utterances
        .groupby(
            [
                *conversation_keys,
                "turn_id",
            ],
            sort=False,
        )
        .cumcount()
        .add(1)
    )

    # Position within the whole conversation.
    utterances["utterance_number"] = (
        utterances
        .groupby(
            conversation_keys,
            sort=False,
        )
        .cumcount()
        .add(1)
    )

    utterances["utterance_id"] = (
        utterances["dataset"].astype(str)
        + "::"
        + utterances["conversation_id"].astype(str)
        + "::utterance_"
        + utterances["utterance_number"]
            .astype(str)
            .str.zfill(3)
    )

    utterances["n_words"] = (
        utterances["text"]
        .map(count_words)
        .astype(int)
    )

    # Keep all utterances in the canonical table. This flag only
    # determines the primary geometry subset later.
    utterances["include_geometry_primary"] = (
        utterances["n_words"] >= centroid_min_words
    )

    # This index will correspond exactly to rows of E_utterances.
    utterances["embedding_idx"] = np.arange(
        len(utterances),
        dtype=int,
    )

    preferred_columns = [
        "dataset",
        "video_id",
        "conversation_id",
        "file_number",
        "level",
        "level_label",
        "utterance_id",
        "utterance_number",
        "turn_id",
        "utterance_in_turn",
        "sequence",
        "normalized_time",
        "speaker",
        "speaker_raw",
        "speaker_role",
        "text",
        "n_words",
        "include_geometry_primary",
        "embedding_idx",
        "notes",
        "source_file",
        "_original_dataframe_row",
    ]

    # Retain only columns that actually exist in this dataset.
    output_columns = [
        column
        for column in preferred_columns
        if column in utterances.columns
    ]

    return utterances[output_columns].copy()


def load_transcripts(config):
    from .config import ROOT
    paths = sorted(config.input.glob('wired_*/*.csv'))
    if not paths:
        raise FileNotFoundError(f'No transcripts under {config.input}')
    raw_frames = []
    for path in paths:
        df = pd.read_csv(path)
        # Preserve notebook source-file labels even when input files are external.
        metadata = parse_wired_path(path, ROOT if path.is_relative_to(ROOT) else config.input)
        df['raw_row_id'] = np.arange(len(df))
        for column, value in metadata.items():
            df[column] = value
        raw_frames.append(df)
    wired_raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    wired_raw = wired_raw.rename(columns={'Speaker': 'speaker_raw', 'Utterance': 'text'})
    wired_raw['text'] = wired_raw['text'].astype('string').str.replace(r'\s+', ' ', regex=True).str.strip()
    wired_raw = wired_raw[wired_raw['text'].notna() & wired_raw['text'].ne('')].copy()
    levels = ['child', 'teenager', 'undergraduate', 'graduate', 'expert']
    level_map = {number: (number - 12, label) for number, label in enumerate(levels, 12)}
    wired_raw['level'] = wired_raw['file_number'].map(lambda x: level_map[x][0])
    wired_raw['level_label'] = wired_raw['file_number'].map(lambda x: level_map[x][1])
    wired_raw['level_label'] = pd.Categorical(wired_raw['level_label'], categories=levels, ordered=True)
    is_expert = wired_raw['speaker_raw'].str.contains(r'\bSpeaker\s*1\b', case=False, na=False)
    wired_raw['speaker'] = np.where(is_expert, 'A', 'B')
    wired_raw['speaker_role'] = np.where(is_expert, 'expert', 'partner')
    return create_utterances_table(wired_raw, config.centroid_min_words)
