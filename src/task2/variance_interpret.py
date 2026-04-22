import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CONFIG_PATH = "config/prediction_model_config.json"
COMP_RANGE = (1, 20)
OUT_DIR = "result/task2/pca_variance_comparison.png"


class VarianceInterpretation():
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.save_path = OUT_DIR

    def load_returns_from_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        sp100 = pd.read_csv(
            cfg["data"]["sp100_path"], index_col=0, parse_dates=True
        ).sort_index() - 1.0
        hsi = pd.read_csv(
            cfg["data"]["hsi_path"], index_col=0, parse_dates=True
        ).sort_index() - 1.0
        self.returns = {"sp100": sp100, "hsi": hsi}

    def compute_explained_variance(self, data: pd.DataFrame, max_components: int):
        if data is None or data.size == 0:
            return np.array([])

        n_samples, n_features = data.shape
        if n_samples < 2 or n_features < 1:
            return np.array([])

        n_comp = min(n_samples, n_features)
        n_comp = min(n_comp, int(max_components))

        scaler = StandardScaler()
        X = scaler.fit_transform(data.values)
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(X)
        return pca.explained_variance_ratio_

    def plot(self, all_evr, comp_range):
        comp_start, comp_end = comp_range
        plt.figure(figsize=(10, 5))

        for name, evr in all_evr.items():
            plot_evr = evr[comp_start: comp_end +
                           1] if len(evr) > comp_start else np.array([])
            x = np.arange(comp_start + 1, comp_start + 1 + len(plot_evr))
            plt.plot(x, plot_evr, marker="o", linestyle="-", label=name)

        plt.xlabel("Number of Principal Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA Explained Variance Ratio Comparison")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150)
        plt.close()

    def compute_and_plot(self):
        _, comp_end = COMP_RANGE

        # 计算所有市场的方差解释率
        results = {}
        for name, data in self.returns.items():
            max_comp = comp_end + 1
            evr = self.compute_explained_variance(data, max_comp)
            results[name] = evr

        self.plot(results, COMP_RANGE)


if __name__ == "__main__":
    variance_interp = VarianceInterpretation(CONFIG_PATH)
    variance_interp.load_returns_from_config()
    variance_interp.compute_and_plot()
