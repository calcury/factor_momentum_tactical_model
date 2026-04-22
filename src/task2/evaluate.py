import pandas as pd

HSI_PATH = "data/hsi.csv"
SP100_PATH = "data/sp100.csv"
HSI_PRED_PATH = "result/task2/prediction_result_hsi/hsi_factor_val_predictions.csv"
SP100_PRED_PATH = "result/task2/prediction_result_sp100/sp100_factor_val_predictions.csv"


class PredictionEvaluation():
    def __init__(self):
        self.hsi_real = pd.read_csv(HSI_PATH)
        self.sp_real = pd.read_csv(SP100_PATH)
        self.hsi_pred = pd.read_csv(HSI_PRED_PATH)
        self.sp_pred = pd.read_csv(SP100_PRED_PATH)

    def pearson_corr(self, actual, pred):
        if len(actual) == 0:
            return float('nan')
        a = pd.Series(actual).astype(float)
        f = pd.Series(pred).astype(float)
        return a.corr(f)

    def evaluate(self):
        # normalize/parse date columns
        if 'date' in self.hsi_pred.columns:
            self.hsi_pred['date'] = pd.to_datetime(self.hsi_pred['date'])
        if 'date' in self.sp_pred.columns:
            self.sp_pred['date'] = pd.to_datetime(self.sp_pred['date'])

        # source files use Chinese header '日期'
        hsi_rel_dates = pd.to_datetime(self.hsi_real['日期'])
        sp_rel_dates = pd.to_datetime(self.sp_real['日期'])

        # find common dates (intersection)
        common_hsi_dates = pd.Index(hsi_rel_dates).intersection(
            pd.Index(self.hsi_pred['date']))
        common_sp_dates = pd.Index(sp_rel_dates).intersection(
            pd.Index(self.sp_pred['date']))

        hsi_eva = self.hsi_pred[self.hsi_pred['date'].isin(
            common_hsi_dates)].reset_index(drop=True)
        sp_eva = self.sp_pred[self.sp_pred['date'].isin(
            common_sp_dates)].reset_index(drop=True)

        # compute Pearson correlation for target vs prediction if columns exist
        hsi_pearson = None
        sp_pearson = None
        if {'target_return', 'pred_target_return'}.issubset(hsi_eva.columns):
            hsi_pearson = self.pearson_corr(hsi_eva['target_return'].values,
                                            hsi_eva['pred_target_return'].values)
        if {'target_return', 'pred_target_return'}.issubset(sp_eva.columns):
            sp_pearson = self.pearson_corr(sp_eva['target_return'].values,
                                           sp_eva['pred_target_return'].values)

        print("=" * 40)
        print("Prediction Evaluation Results:")
        print(f"HSI Pearson r: {hsi_pearson}")
        print(f"SP100 Pearson r: {sp_pearson}")


if __name__ == "__main__":
    evaluator = PredictionEvaluation()
    evaluator.evaluate()
