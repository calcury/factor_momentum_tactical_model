import json
from pathlib import Path

import pandas as pd

try:
    from .decision_model import FullThrottleEngine
    from .prediction_model import LinePredictionPipeline
except ImportError:
    from decision_model import FullThrottleEngine
    from prediction_model import LinePredictionPipeline

CONFIG_PATH = "config/prediction_model_config.json"


ROLLING_WINDOWS = [
    {"name": "test1", "train": (0, 50), "val": (50, 100), "test": (100, 150)},
    {"name": "test2", "train": (50, 100), "val": (
        100, 150), "test": (150, 200)},
    {"name": "test3", "train": (100, 150), "val": (
        150, 200), "test": (200, 248)},
]


class Backtesting():
    def __init__(self):
        pass

    def _read_pred_len(self, cfg, market):
        out_cfg = cfg[f"{market}_output"]
        out_dir = Path(out_cfg["directory"])
        val_df = pd.read_csv(out_dir / out_cfg["val_prediction_file"])
        test_df = pd.read_csv(out_dir / out_cfg["test_prediction_file"])
        return len(val_df), len(test_df)

    def run_backtesting(self, show_plot=False):
        pipeline = LinePredictionPipeline.from_config_path(CONFIG_PATH)
        decision_engine = FullThrottleEngine(CONFIG_PATH)

        root_out = Path("result/task3")
        root_out.mkdir(parents=True, exist_ok=True)

        summary_rows = []

        for window in ROLLING_WINDOWS:
            run_name = window["name"]
            tr_rg, va_rg, te_rg = window["train"], window["val"], window["test"]
            print(f"\n=== {run_name} ===")
            print(
                f"raw split ranges: train={tr_rg}, val={va_rg}, test={te_rg}")

            pipeline.set_split_ranges(tr_rg, va_rg, te_rg)
            pred_results = pipeline.run_all_markets()
            for res in pred_results:
                print(
                    f"[{res['market']}] split={res['split']} reg_params={res['reg_params']}")

            sp_val_len, sp_test_len = self._read_pred_len(
                pipeline.cfg, "sp100")
            hsi_val_len, hsi_test_len = self._read_pred_len(
                pipeline.cfg, "hsi")
            val_len = min(sp_val_len, hsi_val_len)
            test_len = min(sp_test_len, hsi_test_len)
            if val_len < 2:
                raise ValueError("val 预测样本过少，无法为决策模型构造切分")
            if test_len <= 0:
                raise ValueError("test 预测样本为空，无法执行回测")

            decision_engine.set_split_ranges(
                (0, 1),
                (1, val_len),
                (val_len, val_len + test_len),
            )

            run_out = root_out / run_name
            run_out.mkdir(parents=True, exist_ok=True)
            decision_engine.output_dir = run_out

            result = decision_engine.run(split="test")
            decision_engine.save_nav(result, split="test")
            decision_engine.plot(result, split="test",
                                 save=True, show_plot=show_plot)
            decision_engine.plot_weights(
                result, split="test", save=True, show_plot=show_plot)

            # 保存窗口信息
            meta = {
                "name": run_name,
                "raw_split_ranges": {
                    "train": tr_rg,
                    "val": va_rg,
                    "test": te_rg,
                },
                "prediction_rows": {
                    "val": int(val_len),
                    "test": int(test_len),
                },
                "final_nav": float(result["NAV"].iloc[-1]),
            }
            (run_out / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            summary_rows.append(
                {
                    "run": run_name,
                    "raw_train": str(tr_rg),
                    "raw_val": str(va_rg),
                    "raw_test": str(te_rg),
                    "pred_val_rows": int(val_len),
                    "pred_test_rows": int(test_len),
                    "final_nav": float(result["NAV"].iloc[-1]),
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(root_out / "rolling_summary.csv",
                          index=False, encoding="utf-8-sig")
        print("\nBacktesting finished. Summary saved to data/task3/backtesting/rolling_summary.csv")


if __name__ == "__main__":
    backtester = Backtesting()
    backtester.run_backtesting(show_plot=False)
