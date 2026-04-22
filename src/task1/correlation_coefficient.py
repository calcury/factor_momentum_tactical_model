import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

HIGH_CORR_THRESHOLD = 0.9
LOW_CORR_THRESHOLD = -0.7
STOCK_STATIC_FILE = "data/stock_static.csv"
OUTPUT_TXT = 'result/task1/correlation_groups.txt'
OUTPUT_HEATMAP = 'result/task1/stock_correlation_heatmap.png'


class Corr():
    def __init__(self, high_threshold=HIGH_CORR_THRESHOLD, low_threshold=LOW_CORR_THRESHOLD, output_txt=OUTPUT_TXT, output_heatmap=OUTPUT_HEATMAP):
        self.HIGH_CORR_THRESHOLD = high_threshold
        self.LOW_CORR_THRESHOLD = low_threshold
        self.OUTPUT_TXT = output_txt
        self.OUTPUT_HEATMAP = output_heatmap

    def load_data(self, path=STOCK_STATIC_FILE):
        self.df = pd.read_csv(path, index_col=0)

    def compute_similarity(self, show_heatmap=False):

        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        zscore_df = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
        zscore_df = zscore_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        features = zscore_df.to_numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized_features = features / np.where(norms == 0, 1.0, norms)
        similarity_values = normalized_features @ normalized_features.T

        sim_matrix = pd.DataFrame(
            similarity_values,
            index=zscore_df.index,
            columns=zscore_df.index
        )

        # Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, annot=False, cmap='coolwarm',
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Stock Similarity Matrix (Z-score + Cosine)')
        plt.tight_layout()

        # Save image
        plt.savefig(self.OUTPUT_HEATMAP)
        if show_heatmap:
            plt.show()

        sol = sim_matrix.where(
            np.triu(np.ones(sim_matrix.shape), k=1).astype(bool))
        df_pairs = (
            sol.rename_axis(index='Stock_A', columns='Stock_B')
            .stack()
            .reset_index(name='Similarity')
        )

        self.top_high = df_pairs.sort_values(
            by='Similarity', ascending=False).head(10)
        self.top_low = df_pairs.sort_values(
            by='Similarity', ascending=True).head(10)

        self.high_corr_pairs = df_pairs[df_pairs['Similarity'] >= self.HIGH_CORR_THRESHOLD].sort_values(
            by='Similarity', ascending=False
        )
        self.low_corr_pairs = df_pairs[df_pairs['Similarity'] <= self.LOW_CORR_THRESHOLD].sort_values(
            by='Similarity', ascending=True
        )

    def print_top_pairs(self):
        print("=== high similarity ===")
        print(self.top_high)

        print()

        print("=== low similarity ===")
        print(self.top_low)

    def save_results(self):
        with open(self.OUTPUT_TXT, 'w', encoding='utf-8') as f:
            f.write("Stock Similarity Group Detection\n")
            f.write("=" * 40 + "\n")
            f.write("Method: Z-score normalized cosine similarity\n")
            f.write(
                f"High similarity threshold: >= {self.HIGH_CORR_THRESHOLD}\n")
            f.write(
                f"Low similarity threshold: <= {self.LOW_CORR_THRESHOLD}\n\n")

            f.write(
                f"High similarity groups ({len(self.high_corr_pairs)} pairs)\n")
            f.write("-" * 40 + "\n")
            if self.high_corr_pairs.empty:
                f.write("No high similarity pairs found.\n\n")
            else:
                for _, row in self.high_corr_pairs.iterrows():
                    f.write(
                        f"{row['Stock_A']} <-> {row['Stock_B']}: "
                        f"{row['Similarity']:.6f}\n"
                    )
                f.write("\n")

            f.write(
                f"Low similarity groups ({len(self.low_corr_pairs)} pairs)\n")
            f.write("-" * 40 + "\n")
            for _, row in self.low_corr_pairs.iterrows():
                f.write(
                    f"{row['Stock_A']} <-> {row['Stock_B']}: "
                    f"{row['Similarity']:.6f}\n"
                )


if __name__ == "__main__":
    corr = Corr()
    corr.load_data()
    corr.compute_similarity(show_heatmap=True)
    corr.print_top_pairs()
    corr.save_results()
