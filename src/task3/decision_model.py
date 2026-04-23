import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

CONFIG_PATH = "config/prediction_model_config.json"

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class FullThrottleEngine:
    def __init__(self, split_ranges=None):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.split_ranges = split_ranges

        decision_cfg = self.cfg.get("decision_v2", {})
        tx_cfg = self.cfg.get("transaction_cost", {})
        risk_cfg = self.cfg.get("risk", {})

        self.initial_capital = float(self.cfg.get("initial_capital", 100000.0))
        self.comm_rates = {
            'sp100': float(tx_cfg.get('sp100_commission', 0.001)),
            'hsi': float(tx_cfg.get('hsi_commission', 0.0015)),
        }
        annual_mgmt_fee = float(risk_cfg.get("annual_management_fee", 0.0005))
        annualization_days = float(risk_cfg.get("annualization_days", 252))
        self.mgmt_fee = annual_mgmt_fee / annualization_days

        # 策略参数，支持通过 config 中的 decision_v2 区块覆盖
        self.momentum_lookback = int(decision_cfg.get("momentum_lookback", 3))
        self.momentum_weight = float(decision_cfg.get("momentum_weight", 0.5))
        self.target_exposure = float(decision_cfg.get("target_exposure", 0.98))
        self.rebalance_threshold = float(
            decision_cfg.get("rebalance_threshold", 0.15))
        self.min_nav_eps = float(decision_cfg.get("min_nav_eps", 1e-12))

        output_cfg = self.cfg.get("decision_v2_output", {})
        self.output_dir = Path(output_cfg.get(
            "directory", "data/task3/decision_v2"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_names = {
            "train_nav": output_cfg.get("train_nav_file", "train_nav.csv"),
            "val_nav": output_cfg.get("val_nav_file", "val_nav.csv"),
            "test_nav": output_cfg.get("test_nav_file", "test_nav.csv"),
            "train_plot": output_cfg.get("train_nav_plot", "train_nav_vs_benchmark.png"),
            "val_plot": output_cfg.get("val_nav_plot", "val_nav_vs_benchmark.png"),
            "test_plot": output_cfg.get("test_nav_plot", "test_nav_vs_benchmark.png"),
            "train_weight_plot": output_cfg.get("train_weight_plot", "train_weights.png"),
            "val_weight_plot": output_cfg.get("val_weight_plot", "val_weights.png"),
            "test_weight_plot": output_cfg.get("test_weight_plot", "test_weights.png"),
        }

        compare_cfg = self.cfg.get("decision_v2_model_compare", {})
        self.compare_model_label = compare_cfg.get("label", "无动量模型")
        self.compare_nav_files = {
            "train": compare_cfg.get("train_nav_file", ""),
            "val": compare_cfg.get("val_nav_file", "no_motion_val_nav.csv"),
            "test": compare_cfg.get("test_nav_file", "no_motion_test_nav.csv"),
        }

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

    def _load_prediction_frame(self, market):
        out_cfg = self.cfg[f"{market}_output"]
        out_dir = Path(out_cfg["directory"])

        # 兼容现有产物：将 val/test 预测拼接后按日期排序，供 tuple 切分。
        val_path = out_dir / out_cfg["val_prediction_file"]
        test_path = out_dir / out_cfg["test_prediction_file"]
        val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

        df = pd.concat([val_df, test_df], axis=0)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        return df

    def _load_compare_model_nav(self, split):
        file_name = self.compare_nav_files.get(split)
        if not file_name:
            return None

        path = Path(file_name)
        if not path.is_absolute():
            path = self.output_dir / path

        if not path.exists():
            return None

        df_cmp = pd.read_csv(path, index_col=0, parse_dates=True)
        if "NAV" not in df_cmp.columns:
            return None
        return df_cmp["NAV"]

    def _validate_columns(self, df, asset):
        required_cols = {"pred_target_return",
                         "target_return", "next_day_return"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{asset} 缺少必要列: {sorted(missing)}")

    def run(self, split="val", split_ranges=None):
        # 1. 载入 Task2 的预测数据
        if split not in {"train", "val", "test"}:
            raise ValueError("split 仅支持 'train'、'val' 或 'test'")

        df_sp_full = self._load_prediction_frame("sp100")
        df_hsi_full = self._load_prediction_frame("hsi")

        self._validate_columns(df_sp_full, "sp100")
        self._validate_columns(df_hsi_full, "hsi")

        idx_all = df_sp_full.index.intersection(
            df_hsi_full.index).sort_values()
        resolved_ranges = self._resolve_split_ranges(
            len(idx_all), split_ranges)
        s, e = resolved_ranges[split]
        idx = idx_all[s:e]

        if len(idx) == 0:
            raise ValueError(f"{split} 区间为空，请检查 split_ranges")

        df_sp = df_sp_full.loc[idx]
        df_hsi = df_hsi_full.loc[idx]

        cash = self.initial_capital
        units = {'sp100': 0.0, 'hsi': 0.0}
        history = []

        # 基准组合口径与主策略一致：初始建仓支付佣金，持有期扣管理费
        b_sp = self.initial_capital / (1 + self.comm_rates['sp100'])
        b_hsi = self.initial_capital / (1 + self.comm_rates['hsi'])
        b50_sp = 0.5 * self.initial_capital / (1 + self.comm_rates['sp100'])
        b50_hsi = 0.5 * self.initial_capital / (1 + self.comm_rates['hsi'])
        b_5050 = b50_sp + b50_hsi

        for i, date in enumerate(idx):
            nav = cash + sum(units.values())

            # --- 2. 暴力决策引擎 ---
            # 计算两者的“进攻潜力”
            potentials = {}
            for asset, df in [('sp100', df_sp), ('hsi', df_hsi)]:
                pred_r = df.at[date, 'pred_target_return']
                # 动能过滤：最近 3 天是否在涨
                start = max(0, i - self.momentum_lookback)
                mom = df['target_return'].iloc[start:i].sum()
                potentials[asset] = pred_r + \
                    (mom * self.momentum_weight)  # 预测 + 动能补偿

            # 确定目标：谁强买谁，且必须满仓
            best_asset = max(potentials, key=lambda x: potentials[x])

            target_weights = {'sp100': 0.0, 'hsi': 0.0}
            if potentials[best_asset] > 0:
                target_weights[best_asset] = self.target_exposure  # 满仓出击
            else:
                # 如果两个都在跌，空仓避险（这是赢过全仓的关键！）
                target_weights = {'sp100': 0.0, 'hsi': 0.0}

            # --- 3. 极速调仓 (修正后的 units 逻辑) ---
            for asset in ['sp100', 'hsi']:
                curr_w = units[asset] / nav if nav > self.min_nav_eps else 0.0
                tw = target_weights[asset]

                # 只有当目标发生根本性改变（如换赛道）才交易，节省佣金
                if abs(tw - curr_w) > self.rebalance_threshold:
                    target_val = np.floor(nav * tw)
                    diff = target_val - units[asset]
                    fee = abs(diff) * self.comm_rates[asset]

                    if diff > 0 and cash >= (diff + fee):  # 买入
                        cash -= (diff + fee)
                        units[asset] += diff
                    elif diff < 0:  # 卖出
                        cash += (abs(diff) - fee)
                        units[asset] += diff

            # --- 4. 结算 ---
            nav_after_trade = cash + sum(units.values())
            sp_w = units['sp100'] / \
                nav_after_trade if nav_after_trade > self.min_nav_eps else 0.0
            hsi_w = units['hsi'] / \
                nav_after_trade if nav_after_trade > self.min_nav_eps else 0.0
            history.append({'Date': date, 'NAV': nav_after_trade, 'SP100_W': sp_w, 'HSI_W': hsi_w,
                            'B_SP': b_sp, 'B_HSI': b_hsi, 'B_50': b_5050})

            if i < len(idx) - 1:
                r_sp, r_hsi = df_sp.iloc[i +
                                         1]['next_day_return'], df_hsi.iloc[i+1]['next_day_return']
                units['sp100'] *= (1 + r_sp - self.mgmt_fee)
                units['hsi'] *= (1 + r_hsi - self.mgmt_fee)
                b_sp *= (1 + r_sp - self.mgmt_fee)
                b_hsi *= (1 + r_hsi - self.mgmt_fee)
                b50_sp *= (1 + r_sp - self.mgmt_fee)
                b50_hsi *= (1 + r_hsi - self.mgmt_fee)
                b_5050 = b50_sp + b50_hsi

        return pd.DataFrame(history).set_index('Date')

    def plot(self, res, split="val", save=True, show_plot=False):
        split_upper = split.upper()
        plt.figure(figsize=(14, 7))
        plt.plot(res['NAV'], label='终极进攻策略', color='red', lw=3)

        compare_nav = self._load_compare_model_nav(split)
        if compare_nav is not None:
            plt.plot(compare_nav, label=self.compare_model_label,
                     color='blue', lw=2.5, alpha=0.9)

        plt.plot(res['B_SP'], label='全仓 SP100', alpha=0.4, ls='--')
        plt.plot(res['B_HSI'], label='全仓 HSI', alpha=0.4, ls='--')
        plt.plot(res['B_50'], label='50/50 分仓', alpha=0.6, ls=':')
        plt.title(f"{split_upper} 投资组合价值对比 (期末: ${res['NAV'].iloc[-1]:,.2f})")
        plt.legend()
        plt.grid(alpha=0.3)

        if save:
            plot_name = self.file_names[f"{split}_plot"]
            plt.savefig(self.output_dir / plot_name,
                        dpi=160, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()

    def plot_weights(self, res, split="val", save=True, show_plot=False):
        split_upper = split.upper()
        plt.figure(figsize=(14, 6))
        plt.stackplot(
            res.index,
            res['SP100_W'],
            res['HSI_W'],
            labels=['SP100 权重', 'HSI 权重'],
            alpha=0.85,
            colors=['#2A9D8F', '#E76F51']
        )
        plt.plot(res.index, 1 - res['SP100_W'] - res['HSI_W'],
                 label='现金权重', color='#264653', lw=1.8)
        plt.ylim(0, 1.02)
        plt.title(f"{split_upper} 资产配置变化图")
        plt.legend(loc='upper left')
        plt.grid(alpha=0.25)

        if save:
            plot_name = self.file_names[f"{split}_weight_plot"]
            plt.savefig(self.output_dir / plot_name,
                        dpi=160, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()

    def save_nav(self, res, split="val"):
        nav_name = self.file_names[f"{split}_nav"]
        res.to_csv(self.output_dir / nav_name)


if __name__ == "__main__":
    engine = FullThrottleEngine()

    cfg_ranges = engine.cfg.get("split_ranges")
    if cfg_ranges is not None:
        engine.set_split_ranges(
            tuple(cfg_ranges["train"]),
            tuple(cfg_ranges["val"]),
            tuple(cfg_ranges["test"]),
        )
    elif engine.split_ranges is None:
        raise ValueError(
            "请为 FullThrottleEngine 提供 split_ranges，"
            "例如 set_split_ranges((0, 50), (50, 80), (80, None))"
        )

    for split in ["train", "val", "test"]:
        results = engine.run(split=split)
        engine.save_nav(results, split=split)
        engine.plot(results, split=split, save=True)
        engine.plot_weights(results, split=split, save=True)
