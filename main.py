import clean_corpus_make_graph
from clean_corpus_make_graph import clean_corpus, read_corpus, text_to_graph, draw_top_interactive
from graph_utils import *
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
EXPERT_DIR = DATA_DIR / "expert"
NOVICE_DIR = DATA_DIR / "novice"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.txt")):
        text = read_corpus(str(p))
        cleaned = clean_corpus(text, stop_words=None)
        G = text_to_graph(cleaned)
        df = graph_stats(G, graph_id=p.stem)
        rows.append(df)
    return pd.concat(rows) if rows else pd.DataFrame()

def main():
    test_file = '/Users/yume/PycharmProjects/horizons_2/data/novice/speechlike_dreams_novice.txt'
    txt = read_corpus(test_file)
    graph = text_to_graph(txt)
    draw_top_interactive(graph, 200)

    expert_df = process_folder(EXPERT_DIR)
    novice_df = process_folder(NOVICE_DIR)

    expert_df.to_csv(OUTPUT_DIR / "experts_graph_stats.csv")
    novice_df.to_csv(OUTPUT_DIR / "novices_graph_stats.csv")

    print("Experts:\n", expert_df.head(), "\n")
    print("Novices:\n", novice_df.head())

    import matplotlib.pyplot as plt
    import seaborn as sns

    def violin_comparison(expert_df, novice_df):
        # Label the groups and combine
        expert_df["group"] = "Expert"
        novice_df["group"] = "Novice"
        combined = pd.concat([expert_df, novice_df])

        # Keep only numeric columns
        numeric_cols = combined.select_dtypes(include=["number"]).columns

        # Plot settings
        num_metrics = len(numeric_cols)
        fig, axes = plt.subplots(nrows=(num_metrics + 2) // 3, ncols=3, figsize=(15, num_metrics * 0.8))
        axes = axes.flatten()

        for i, metric in enumerate(numeric_cols):
            sns.violinplot(
                data=combined,
                x="group",
                y=metric,
                ax=axes[i],
                inner="box",  # show quartiles inside violin
                cut=0,
                palette="muted"
            )
            axes[i].set_title(metric)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Expert vs Novice Graph Metric Distributions", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    violin_comparison(expert_df, novice_df)

    expert_df = process_folder(EXPERT_DIR)
    novice_df = process_folder(NOVICE_DIR)

    expert_df["label"] = 1
    novice_df["label"] = 0

    full_df = pd.concat([expert_df, novice_df])
    full_df.to_csv(OUTPUT_DIR / "features_for_classification.csv")

    print("Saved:", OUTPUT_DIR / "features_for_classification.csv")
    print(full_df.head())


if __name__ == '__main__':
    main()