import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SP100_FILE = "data/sp100.csv"
HSI_FILE = "data/hsi.csv"
SAVE_PATH = "result/task2/avg_return.csv"
FIGURE_PATH = "result/task2/avg_return_comparison.png"


class AverageReturn():
    def __init__(self, sp100_file=SP100_FILE, hsi_file=HSI_FILE):
        self.sp100_file = sp100_file
        self.hsi_file = hsi_file

    def compute_average_return(self):
        sp100_raw = pd.read_csv(self.sp100_file, index_col=0, parse_dates=True)
        hsi_raw = pd.read_csv(self.hsi_file, index_col=0, parse_dates=True)

        # mean - 1
        sp100_daily_index_return = sp100_raw.mean(axis=1) - 1
        hsi_daily_index_return = hsi_raw.mean(axis=1) - 1

        sp100_daily_index_return = np.log(sp100_raw.mean(axis=1))
        hsi_daily_index_return = np.log(hsi_raw.mean(axis=1))

        self.index_returns = pd.DataFrame({
            'sp100': sp100_daily_index_return,
            'hsi': hsi_daily_index_return
        })

    def plot_average_return(self, show_plot=True):
        plt.figure(figsize=(14, 6))
        plt.plot(self.index_returns['sp100'], label='sp100', alpha=0.7)
        plt.plot(self.index_returns['hsi'], label='hsi', alpha=0.7)
        plt.title('Daily Index Returns Comparison')
        plt.xlabel('Date')
        plt.ylabel('Daily Return Rate')
        plt.axhline(0, linewidth=0.8, linestyle='--')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(FIGURE_PATH)
        if show_plot:
            plt.show()

    def save_average_return(self, index_returns, save_path=SAVE_PATH):
        index_returns.to_csv(save_path)


if __name__ == "__main__":
    avg_return = AverageReturn()
    avg_return.compute_average_return()
    avg_return.plot_average_return(show_plot=True)
    avg_return.save_average_return(avg_return.index_returns)
