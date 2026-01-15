import argparse
import pandas as pd

# Columns that must swap A<->B when we swap speakers
SWAP_PAIRS = [
    ("speaker_A", "speaker_B"),
    ("n_A", "n_B"),
    ("cross_nn_A_to_B", "cross_nn_B_to_A"),
    ("influence_A_toward_B", "influence_B_toward_A"),
    ("turn_jump_A_to_B_mean", "turn_jump_B_to_A_mean"),
    ("hull_area_A", "hull_area_B"),
    ("containment_A_in_B", "containment_B_in_A"),
    ("entropy_A", "entropy_B"),
    ("entropy_slope_A", "entropy_slope_B"),
    ("align_lag_A_to_B", "align_lag_B_to_A"),
]

# Columns that must flip sign when we swap speakers
FLIP_SIGN = [
    "influence_asymmetry_AminusB",
    "align_lag_asymmetry_AminusB",
]

def ensure_speaker1_is_A(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # row mask: Speaker 1 is currently in B but not in A
    mask = df["speaker_B"].astype(str).str.contains("Speaker 1", na=False) & \
           ~df["speaker_A"].astype(str).str.contains("Speaker 1", na=False)

    if mask.sum() == 0:
        return df

    # Swap paired columns
    for a, b in SWAP_PAIRS:
        if a in df.columns and b in df.columns:
            tmp = df.loc[mask, a].copy()
            df.loc[mask, a] = df.loc[mask, b].values
            df.loc[mask, b] = tmp.values

    # Flip sign for asymmetry columns
    for col in FLIP_SIGN:
        if col in df.columns:
            df.loc[mask, col] = -df.loc[mask, col]

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input metrics CSV")
    ap.add_argument("--output", required=True, help="Output CSV with Speaker 1 forced to A when present")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df2 = ensure_speaker1_is_A(df)
    df2.to_csv(args.output, index=False)

    changed = ((df["speaker_B"].astype(str).str.contains("Speaker 1", na=False)) &
               (~df["speaker_A"].astype(str).str.contains("Speaker 1", na=False))).sum()

    print(f"Done. Rows swapped: {changed}")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
