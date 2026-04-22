import pandas as pd
import numpy as np
import re

SP100_FILE = "data/sp100.csv"
HSI_FILE = "data/hsi.csv"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0
SAVE_PATH = ["data", "result/task1"]


class FeatureAnalysis():
    def __init__(self, sp100_file=SP100_FILE, hsi_file=HSI_FILE, save_path=SAVE_PATH):
        self.sp100_file = sp100_file
        self.hsi_file = hsi_file
        self.save_path = save_path

    def process_index_data(self, file_path, index_name):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        returns = df - 1.0

        # 日均收益率
        daily_mean = returns.mean()

        # 年化收益率
        annual_return = daily_mean * TRADING_DAYS

        # 年化波动率
        daily_std = returns.std()
        annual_volatility = daily_std * np.sqrt(TRADING_DAYS)

        # 夏普比率
        sharpe_ratio = (annual_return - RISK_FREE_RATE) / annual_volatility

        # 累计收益率
        cumulative_return = (1 + returns).prod() - 1

        # 最大回撤
        def max_drawdown(series):
            cum_values = (1 + series).cumprod()
            running_max = cum_values.cummax()
            drawdown = (cum_values - running_max) / running_max
            return drawdown.min()

        max_dd = returns.apply(max_drawdown)

        # 正收益天数占比
        total_days = returns.shape[0]
        win_rate = (returns > 0).sum() / total_days

        stats = pd.DataFrame({
            'Daily mean': daily_mean,
            'Annualized rate of return': annual_return,
            'Annualized volatility': annual_volatility,
            'Sharpe ratio': sharpe_ratio,
            'Cumulative return': cumulative_return,
            'Max drawdown': max_dd,
            'Win rate': win_rate
        })

        normalized_index = []
        for stock_col in stats.index:
            matched = re.search(r"(\d+)", str(stock_col))
            if matched:
                stock_id = matched.group(1)
                normalized_index.append(f"stock_{stock_id}_{index_name}")
            else:
                fallback = str(stock_col).strip().replace(" ", "_")
                normalized_index.append(f"stock_{fallback}_{index_name}")

        stats.index = normalized_index
        stats.index.name = 'Stock'

        return stats

    def rank_by_composite_score(self, stats_df):
        scored = stats_df.copy()
        clean = scored.replace([np.inf, -np.inf], np.nan).copy()

        metric_cols = [
            'Sharpe ratio',
            'Annualized rate of return',
            'Cumulative return',
            'Win rate',
            'Max drawdown',
            'Annualized volatility'
        ]
        for col in metric_cols:
            clean[col] = clean[col].fillna(clean[col].median())

        factor_score = pd.DataFrame(index=clean.index)
        factor_score['Sharpe ratio'] = clean['Sharpe ratio'].rank(
            pct=True, ascending=True)
        factor_score['Annualized rate of return'] = clean['Annualized rate of return'].rank(
            pct=True, ascending=True)
        factor_score['Cumulative return'] = clean['Cumulative return'].rank(
            pct=True, ascending=True)
        factor_score['Win rate'] = clean['Win rate'].rank(
            pct=True, ascending=True)
        factor_score['Max drawdown'] = clean['Max drawdown'].rank(
            pct=True, ascending=True)
        factor_score['Annualized volatility'] = clean['Annualized volatility'].rank(
            pct=True, ascending=False)

        weights = {
            'Sharpe ratio': 0.35,
            'Annualized rate of return': 0.20,
            'Cumulative return': 0.15,
            'Win rate': 0.10,
            'Max drawdown': 0.15,
            'Annualized volatility': 0.05
        }

        scored['Composite score'] = sum(
            factor_score[col] * w for col, w in weights.items()
        )

        return scored.sort_values('Composite score', ascending=False)

    def save_results(self, all_stats_sorted):
        for path in self.save_path:
            all_stats_sorted.to_csv(f"{path}/stock_static.csv")

    def process_and_save(self):
        sp100_stats = self.process_index_data(self.sp100_file, "SP100")
        hsi_stats = self.process_index_data(self.hsi_file, "HSI")

        all_stats = pd.concat([sp100_stats, hsi_stats])
        self.all_stats_sorted = self.rank_by_composite_score(all_stats)

        self.save_results(self.all_stats_sorted)

    def print_results(self, top_n=10):
        print("\n========== best-performing stocks ==========")
        print(self.all_stats_sorted.head(1))

        print("\n========== worst-performing stocks ==========")
        print(self.all_stats_sorted.tail(1))


if __name__ == "__main__":
    analysis = FeatureAnalysis()
    analysis.process_and_save()
    analysis.print_results(top_n=10)
