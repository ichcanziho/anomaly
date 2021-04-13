import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class Metrics:

    @staticmethod
    def get_roc(dataset):
        ys = pd.read_csv(dataset)
        y_test = ys['real'].values
        y_pred = ys['prediction'].values
        try:
            auc = roc_auc_score(y_test, y_pred)
            if auc <= 0.5:
                auc = 1 - auc
        except:  # If a problem is presented, return the base AUC score
            auc = 0.5

        return auc

    @staticmethod
    def get_ave(dataset):
        y_true = pd.read_csv(dataset, sep=',').iloc[:, 0]
        y_score = pd.read_csv(dataset, sep=',').iloc[:, 1]
        baseline = list(y_true).count('positive') / len(y_true)
        try:
            AUC_PR = average_precision_score(y_true, y_score, pos_label='positive')
            if AUC_PR < baseline:
                AUC_PR = average_precision_score(y_true, -1 * y_score, pos_label='positive')
        except:  # If a problem is presented, return the base AUC_PR score
            AUC_PR = baseline
        return AUC_PR
