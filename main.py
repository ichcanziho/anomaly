# SKLEARN classifiers
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import FactorAnalysis
from sklearn.neighbors import KernelDensity
# PYOD classifiers
from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.knn import KNN
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.sod import SOD
from pyod.models.lscp import LSCP
from pyod.models.lmdd import LMDD
from pyod.models.vae import VAE
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
# OTHERS classifiers
from core.OCKRA.m_ockra import m_OCKRA
from brminer import BRM
# Main methods
from core import ADRanker
from core import Metrics
from core import Plots


def main():

    # PART 1:
    # Getting the predictions for each classifier
    # SK means: The classifier is from sklearn or works like sklearn
    # PY means: The classifier is from pyod or works like pyod

    models = {'SK_EE': EllipticEnvelope(),
              'SK_GM': GaussianMixture(),
              'SK_IF': IsolationForest(),
              'SK_OCSVM': OneClassSVM(),
              'SK_FA': FactorAnalysis(),
              'SK_KD': KernelDensity(),
              'PY_PCA': PCA(),
              'PY_COF': COF(),
              'PY_LODA': LODA(),
              'PY_LOF': LOF(),
              'PY_HBOS': HBOS(),
              'PY_MCD': MCD(),
              'PY_AvgKNN': KNN(method='mean'),
              'PY_LargestKNN': KNN(method='largest'),
              'PY_MedKNN': KNN(method='median'),
              'PY_AvgBagging': FeatureBagging(combination='average'),
              'PY_MaxBagging': FeatureBagging(combination='max'),
              'PY_CBLOF': CBLOF(n_clusters=10, n_jobs=4),
              'PY_COPOD': COPOD(),
              'PY_SOD': SOD(),
              'PY_LSCPwithLODA': LSCP([LODA(), LODA()]),
              'PY_AveLMDD': LMDD(dis_measure='aad'),
              'PY_VarLMDD': LMDD(dis_measure='var'),
              'PY_IqrLMDD': LMDD(dis_measure='iqr'),
              'PY_VAE': VAE(encoder_neurons=[8, 4, 2]),
              'PY_AutoEncoder': AutoEncoder(hidden_neurons=[6, 3, 3, 6]),
              'SK_BRM': BRM(bootstrap_sample_percent=70),
              'SK_OCKRA': m_OCKRA(),
              'PY_SoGaal': SO_GAAL(),
              'PY_MoGaal': MO_GAAL()
              }
    ranker = ADRanker(data="datasets", models=models)
    ranker.get_predictions()

    # PART 2:
    # After predictions, we can evaluate our classifiers using different scores
    # You can add manually a new metric by modifying 'metrics.py'

    ranker.get_scores(scores={'auc': Metrics.get_roc, 'ave': Metrics.get_ave})

    # PART 3:
    # Finally, it is time to summarize the results by plotting different graphs
    # You can add your own graphs by modifying ' plots.py'
    plot = Plots()
    plot.make_plot_basic(paths=['results/scores/auc/no/results.csv',
                                'results/scores/auc/minmax/results.csv',
                                'results/scores/auc/std/results.csv',
                                'results/scores/ave/no/results.csv',
                                'results/scores/ave/minmax/results.csv',
                                'results/scores/ave/std/results.csv'
                                ],
                         scalers=['Without scaler', 'Min max scaler', 'Standard scaler',
                                  'Without scaler', 'Min max scaler', 'Standard scaler'])

    plot.make_cd_plot(paths=['results/scores/auc/minmax/results.csv', 'results/scores/ave/no/results.csv',
                             'results/scores/auc/no/results.csv', 'results/scores/ave/no/results.csv',
                             'results/scores/auc/std/results.csv', 'results/scores/ave/std/results.csv'],
                      names=['CD auc minmax scale', 'CD ave minmax scale',
                             'CD auc no scale', 'CD ave no scale',
                             'CD auc std scale', 'CD ave std scale'],
                      titles=['CD diagram - AUC with min max scaling',
                              'CD diagram - Average precision with min max scaling',
                              'CD diagram - AUC without scaling',
                              'CD diagram - Average precision without scaling',
                              'CD diagram - AUC with standard scaling',
                              'CD diagram - Average precision with  standard scaling'])


if __name__ == '__main__':
    main()

