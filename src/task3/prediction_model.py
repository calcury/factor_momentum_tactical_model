import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

CONFIG_PATH = "config/prediction_model_config.json"


class LinePredictionPipeline:
    def __init__(self, cfg, split_ranges=None):
        self.cfg = cfg
        self.split_ranges = split_ranges

    @classmethod
    def from_config_path(cls, split_ranges=None):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg, split_ranges=split_ranges)

    def set_split_ranges(self, train_range, val_range, test_range):
        self.split_ranges = {
            "train": train_range,
            "val": val_range,
            "test": test_range,
        }

    @staticmethod
    def _normalize_range(n_total, rg, name):
        if rg is None or len(rg) != 2:
            raise ValueError(f"{name} 区间必须是长度为 2 的元组/列表，例如 (0, 50)")

        start, end = rg
        if start is None:
            start = 0
        if end is None:
            end = n_total

        start = int(start)
        end = int(end)

        if start < 0 or end < 0:
            raise ValueError(f"{name} 区间不能为负数: {(start, end)}")
        if start >= end:
            raise ValueError(f"{name} 区间必须满足 start < end: {(start, end)}")
        if end > n_total:
            raise ValueError(f"{name} 区间超出数据范围: {(start, end)}，总长度为 {n_total}")

        return start, end

    @staticmethod
    def _map_raw_range_to_valid_positions(valid_raw_pos, rg, name):
        start, end = rg
        pos = np.where((valid_raw_pos >= start) & (valid_raw_pos < end))[0]
        if len(pos) == 0:
            raise ValueError(
                f"{name} 在原始区间 {(start, end)} 内没有可用样本。"
                "请调整切分区间，或检查特征窗口/目标窗口导致的裁剪。"
            )
        return int(pos[0]), int(pos[-1]) + 1

    def _resolve_split_ranges(self, n_total, split_ranges=None):
        ranges = split_ranges if split_ranges is not None else self.split_ranges
        if ranges is None:
            raise ValueError(
                "未提供切分区间。请在初始化时传入 split_ranges，"
                "或先调用 set_split_ranges(train_range, val_range, test_range)。"
            )

        if isinstance(ranges, (list, tuple)) and len(ranges) == 3:
            train_rg, val_rg, test_rg = ranges
        elif isinstance(ranges, dict):
            train_rg = ranges.get("train")
            val_rg = ranges.get("val")
            test_rg = ranges.get("test")
        else:
            raise ValueError(
                "split_ranges 格式错误。支持 {'train':(a,b),'val':(c,d),'test':(e,f)} "
                "或 ((a,b),(c,d),(e,f))"
            )

        tr_s, tr_e = self._normalize_range(n_total, train_rg, "train")
        va_s, va_e = self._normalize_range(n_total, val_rg, "val")
        te_s, te_e = self._normalize_range(n_total, test_rg, "test")

        if not (tr_e <= va_s <= va_e <= te_s <= te_e):
            raise ValueError(
                "区间必须按时间先后且不重叠，要求满足: train_end <= val_start <= val_end <= test_start <= test_end"
            )

        return {
            "train": (tr_s, tr_e),
            "val": (va_s, va_e),
            "test": (te_s, te_e),
        }

    @staticmethod
    def get_xy(df):
        # Get feature columns by excluding target and return columns.
        separate_cols = ["target", "target_return", "next_day_return"]
        feats = [c for c in df.columns if c not in separate_cols]
        return df[feats].values, df["target_return"].values, feats

    @staticmethod
    def calc_metrics(df):
        # Calculate strategy metrics and prediction metrics.
        p_ret, t_ret = df["pred_target_return"], df["target_return"]
        y_signal = (p_ret > 0).astype(int)
        strat_r = df["next_day_return"] * y_signal

        metrics = {
            "strategy": {
                "ann_ret": strat_r.mean() * 252 if len(strat_r) else 0,
                "ann_vol": strat_r.std(ddof=1) * np.sqrt(252) if len(strat_r) > 1 else 0,
                "sharpe": 0,
                "max_dd": 0,
            },
            "return_prediction": {
                "mae": np.mean(np.abs(t_ret - p_ret)),
                "corr": np.corrcoef(t_ret, p_ret)[0, 1]
                if np.std(t_ret) > 0 and np.std(p_ret) > 0
                else None,
            },
        }

        strategy = metrics["strategy"]
        if strategy["ann_vol"] > 0:
            strategy["sharpe"] = strategy["ann_ret"] / strategy["ann_vol"]

        if len(strat_r):
            nav = (1 + strat_r).cumprod()
            start_nav = ((nav - nav.cummax()) / nav.cummax()).min()
            metrics["strategy"]["max_dd"] = start_nav

        return metrics

    def train_models(self, df_tr, df_val):
        # Train regression model using grid search and return the best one.
        Xt_r, yt_r, _ = self.get_xy(df_tr)
        Xv_r, yv_r, _ = self.get_xy(df_val)

        best_reg = {"score": -np.inf}
        grid = self.cfg.get("regression_model", {}).get(
            "grid", {"alpha": [1.0]})
        for p in ParameterGrid(grid):
            reg = Ridge(**p).fit(Xt_r, yt_r)
            mae = -np.mean(np.abs(yv_r - reg.predict(Xv_r)))
            if mae > best_reg["score"]:
                best_reg = {"score": mae, "params": p, "model": reg}

        return best_reg

    @staticmethod
    def eval_data(data, start, end, is_rolling, reg_p, win=0, min_t=0):
        # Train and evaluate regression model on selected range.
        separate_cols = ["target", "target_return", "next_day_return"]
        feats = [c for c in data.columns if c not in separate_cols]
        rows = []

        eval_range = range(start, end) if is_rolling else [None]

        for i in eval_range:
            if is_rolling:
                train = data.iloc[max(0, i - win) if win >
                                  0 else max(0, i - min_t): i]
                if train.empty:
                    continue
                Xt = train[feats].values
                yt = train["target_return"].values
                Xi = data.iloc[i:i + 1][feats].values

                p_ret = Ridge(**reg_p).fit(Xt, yt).predict(Xi)[0]

                rows.append(
                    {
                        "date": data.index[i],
                        "target_return": data.iloc[i]["target_return"],
                        "pred_target_return": p_ret,
                        "next_day_return": data.iloc[i]["next_day_return"],
                    }
                )
            else:
                X = data[feats].values
                p_ret = Ridge(**reg_p).fit(X,
                                           data["target_return"].values).predict(X)
                return pd.DataFrame(
                    {
                        "target_return": data["target_return"],
                        "pred_target_return": p_ret,
                        "next_day_return": data["next_day_return"],
                    },
                    index=data.index,
                )

        return pd.DataFrame(rows).set_index("date") if is_rolling else None

    def run_market(self, name, returns, split_ranges=None):
        out = self.cfg[f"{name}_output"]
        out_dir = Path(out["directory"])
        out_dir.mkdir(parents=True, exist_ok=True)

        h = int(self.cfg["target"]["horizon_days"])
        thresh = float(self.cfg["target"]["direction_threshold"])
        idx_ret = returns.mean(axis=1)
        df = pd.DataFrame(
            {
                "next_day_return": idx_ret.shift(-1),
                "target_return": idx_ret.rolling(h).sum().shift(-h + 1),
            },
            index=returns.index,
        )
        df["target"] = (df["target_return"] > thresh).astype(int)

        # 切分区间按原始 CSV 行号解释。
        n_total = len(returns)
        resolved_ranges = self._resolve_split_ranges(n_total, split_ranges)
        tr_s, tr_e = resolved_ranges["train"]
        va_s, va_e = resolved_ranges["val"]
        te_s, te_e = resolved_ranges["test"]

        train_idx = returns.index[tr_s:tr_e]

        scaler = StandardScaler()
        pca = PCA(
            n_components=int(self.cfg["features"]["n_factors"]),
            random_state=int(self.cfg["model"]["random_state"]),
        )

        scaler.fit(returns.loc[train_idx].values)
        pca.fit(scaler.transform(returns.loc[train_idx].values))

        f_data = pca.transform(scaler.transform(returns.values))
        f_df = pd.DataFrame(f_data, index=returns.index)
        f_df.columns = [f"f_{i + 1}" for i in range(f_df.shape[1])]

        feats = [f_df, pd.Series(returns.std(axis=1), name="cross_dispersion")]
        for w in self.cfg["features"]["factor_windows"]:
            feats.append(f_df.rolling(w).mean().add_suffix(f"_mom_{w}"))
        for w in self.cfg["features"]["vol_windows"]:
            feats.append(f_df.rolling(w).std().add_suffix(f"_vol_{w}"))
        for w in self.cfg["features"]["index_mom_windows"]:
            feats.append(idx_ret.rolling(w).mean().rename(f"index_mom_{w}"))

        full_dataset = pd.concat(feats, axis=1).join(df, how="inner")
        dataset = full_dataset.dropna()
        valid_raw_pos = np.where(full_dataset.notna().all(axis=1))[0]

        tr_s_v, tr_e_v = self._map_raw_range_to_valid_positions(
            valid_raw_pos, (tr_s, tr_e), "train"
        )
        va_s_v, va_e_v = self._map_raw_range_to_valid_positions(
            valid_raw_pos, (va_s, va_e), "val"
        )
        te_s_v, te_e_v = self._map_raw_range_to_valid_positions(
            valid_raw_pos, (te_s, te_e), "test"
        )

        tr_df = dataset.iloc[tr_s_v:tr_e_v]
        val_df = dataset.iloc[va_s_v:va_e_v]
        te_df = dataset.iloc[te_s_v:te_e_v]

        if tr_df.empty or val_df.empty or te_df.empty:
            raise ValueError("train/val/test 中存在空区间，请检查 split_ranges")

        reg_best = self.train_models(tr_df, val_df)
        roll = self.cfg.get("rolling_oos", {})
        is_roll = roll.get("enabled", False)
        win = int(roll.get("window_size", len(tr_df)))
        min_t = int(roll.get("min_train_size", len(tr_df)))

        if is_roll:
            val_pred = self.eval_data(
                dataset, va_s_v, va_e_v, True, reg_best["params"], win, min_t)
            te_pred = self.eval_data(
                dataset, te_s_v, te_e_v, True, reg_best["params"], win, min_t)
        else:
            val_pred = self.eval_data(val_df, 0, 0, False, reg_best["params"])
            Xtr, ytr, _ = self.get_xy(pd.concat([tr_df, val_df]))
            _ = Ridge(**reg_best["params"]).fit(Xtr, ytr)
            te_pred = self.eval_data(te_df, 0, 0, False, reg_best["params"])

        val_met = self.calc_metrics(val_pred)
        te_met = self.calc_metrics(te_pred)

        val_pred.to_csv(
            out_dir / out["val_prediction_file"], encoding="utf-8-sig")
        te_pred.to_csv(
            out_dir / out["test_prediction_file"], encoding="utf-8-sig")

        plots = [
            (val_pred, "Val", "val_return_compare_plot"),
            (te_pred, "Test", "test_return_compare_plot"),
        ]
        for p_df, title_prefix, fname in plots:
            plt.figure(figsize=(10, 4))
            plt.plot(p_df.index, p_df["target_return"], label="actual")
            plt.plot(p_df.index, p_df["pred_target_return"], label="predicted")
            plt.title(f"{name.upper()} {title_prefix}: Pred vs Actual")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / out[fname], dpi=150)
            plt.close()

        summary = {
            "market": name,
            "split": [len(tr_df), len(val_df), len(te_df)],
            "val": val_met,
            "test": te_met,
        }

        metrics_path = out_dir / out["metrics_file"]
        metrics_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "market": name,
            "split": summary["split"],
            "reg_params": reg_best["params"],
            "val_strategy": val_met["strategy"],
            "test_strategy": te_met["strategy"],
        }

    def load_market_returns(self):
        sp100 = pd.read_csv(
            self.cfg["data"]["sp100_path"], index_col=0, parse_dates=True
        ).sort_index() - 1.0
        hsi = pd.read_csv(
            self.cfg["data"]["hsi_path"], index_col=0, parse_dates=True
        ).sort_index() - 1.0
        return {"sp100": sp100, "hsi": hsi}

    def run_all_markets(self):
        markets = self.load_market_returns()
        results = []
        for name, data in markets.items():
            results.append(self.run_market(name, data))
        return results


if __name__ == "__main__":
    pipeline = LinePredictionPipeline.from_config_path(
        "line_prediction_model_config.json")

    cfg_ranges = pipeline.cfg.get("split_ranges")
    if cfg_ranges is not None:
        pipeline.set_split_ranges(
            tuple(cfg_ranges["train"]),
            tuple(cfg_ranges["val"]),
            tuple(cfg_ranges["test"]),
        )
    elif pipeline.split_ranges is None:
        raise ValueError(
            "请为 LinePredictionPipeline 提供 split_ranges，"
            "例如 set_split_ranges((0, 50), (50, 80), (80, None))"
        )

    for res in pipeline.run_all_markets():
        print(f"=== {res['market']} ===")
        print(
            f"Split (Train/Val/Test): {res['split'][0]}/{res['split'][1]}/{res['split'][2]}")
        print(f"Best Reg Params: {res['reg_params']}")
        print(f"Strategy Metrics (Val) : {res['val_strategy']}")
        print(f"Strategy Metrics (Test): {res['test_strategy']}\n")
