import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_returns_from_config(config_path="line_prediction_model_config.json"):
    json_path = Path(__file__).with_name(config_path)
    cfg = json.loads(json_path.read_text(encoding="utf-8"))

    sp100 = pd.read_csv(
        cfg["data"]["sp100_path"], index_col=0, parse_dates=True
    ).sort_index() - 1.0
    hsi = pd.read_csv(
        cfg["data"]["hsi_path"], index_col=0, parse_dates=True
    ).sort_index() - 1.0
    return {"sp100": sp100, "hsi": hsi}


def compute_explained_variance(returns: pd.DataFrame, max_components: int):
    n_samples, n_features = returns.shape
    if n_samples < 2 or n_features < 1:
        return np.array([])

    n_comp = min(n_samples, n_features)
    n_comp = min(n_comp, int(max_components))

    scaler = StandardScaler()
    X = scaler.fit_transform(returns.values)
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X)
    return pca.explained_variance_ratio_


def plot_multi_variance(
    all_evr: dict,
    comp_range: tuple,
    save_path: Path,
):
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
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


CONFIG_PATH = "line_prediction_model_config.json"
COMP_RANGE = (1, 20)
OUT_DIR = "./data/task2/pca_variance"


if __name__ == "__main__":
    returns = load_returns_from_config(CONFIG_PATH)
    out_dir = Path(OUT_DIR)
    comp_start, comp_end = COMP_RANGE

    # 计算所有市场的方差解释率
    results = {}
    for name, data in returns.items():
        max_comp = comp_end + 1
        evr = compute_explained_variance(data, max_comp)
        results[name] = evr

    # 画在一张图里
    save_path = out_dir / "pca_variance_comparison.png"
    plot_multi_variance(results, COMP_RANGE,
                        save_path=save_path)
