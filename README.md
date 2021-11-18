# Comparison-of-Classifiers-for-Unsupervised-Anomaly-Detection

A comparison of 30 Unsupervised Anomaly Detection Classifieres among 90 Datasets.
- Code Author: [Gabriel Ichcanziho Pérez Landa](https://github.com/ichcanziho)
- Paper Authors: [Gabriel Ichcanziho Pérez Landa](https://www.linkedin.com/in/ichcanziho/), [Virginia Itzel Contreras Miranda](https://www.linkedin.com/in/itzel-contreras-5323abba/), [Daniela Macias Arregoyta](https://www.linkedin.com/in/daniela-macias-arregoyta/) and [Miguel Angel Medina-Pérez](https://sites.google.com/site/miguelmedinaperez/) 
- Date of creation: January 22th, 2021
- Code Author Email: ichcanziho@outlook.com

### Abstract

The problem of detecting anomalies in an unsuper-vised way is one of the most addressed in the fieldof machine learning. The anomaly detection algo-rithms are of the utmost importance, their field ofapplication is critical and ranges from cybersecu-rity to medicine. Finding these anomalies can bevery useful as it allows you to prevent problems. Despite the great relevance of this topic, few articles compare different algorithms in depth. This document compares 30 anomaly detection algorithms using 90 publicly available databases. Theanomaly detection algorithms  belong to different families, being neural networks, probabilistic models, proximity-based, etc. The metrics usedin this paper are Area Under the Curve and Average Precision, developing the analysis with no scaling, min-max scaling, and standard scaling the databases. The results show that, with all the variants, only six out of 30 analyzed algorithms have the best performance, without statistically significant differences among them.  A Critical Difference diagram was created to show this comparison. In the end, only three anomaly detection algorithms were determined as the best, outperforming the rest. This research is useful for fraud detection, intrude detection, etc., and could be applied to several fields of study.

### Installation

This repostory requires [Pip](https://docs.python.org/3/installing/index.html) to install the requirements.
Before install all the libraries, it is very recomendable to make a new [Virtual ENV](https://docs.python.org/3/library/venv.html) to isolate the new libraries.


To run the program you must have some libraries, you can install it using the next command:

```sh
$ pip install -r requirements.txt
```

## Implemented classifiers

- RandomForestClassifier()
- BRM()
- GaussianMixture()
- IsolationForest()
- OneClassSVM()
- EllipticEnvelope()
- KNN(method="mean")
- KNN(method="largest")
- KNN(method="median")
- PCA()
- COF()
- LODA()
- LOF()
- HBOS()
- MCD()
- FeatureBagging(combination='average')
- FeatureBagging(combination='max')
- CBLOF()
- FactorAnalysis()
- KernelDensity()
- COPOD()
- SOD()
- LSCP()
- LMDD(dis_measure='aad')
- LMDD(dis_measure='var')
- LMDD(dis_measure='iqr')
- SO_GAAL()
- MO_GAAL()
- VAE()
- AutoEncoder()
- OCKRA()

## Run the program

The code is divided into three main parts:

1: Getting model's predictions
```sh
$ models = {'model_name': model()}
$ ranker = ADRanker(data="datasets", models=models)
$ ranker.get_predictions()
```
2: Scoring model's predictions
```sh
$ ranker.get_scores(scores={'auc': Metrics.get_roc, 'ave': Metrics.get_ave})
```
You can add your own metrics by modifying 
```sh
$ core/metrics.py
```
3: Summarize the results by plotting different graphs
```sh
$ plot = Plots()
$ plot.make_plot_basic(paths=[], scalers=[])
$ plot.make_cd_plot(paths=[], names=[], titles=[])
```

To generate the AUC and Average Precision results of the 30 models, run:
```sh
$ main.py
```

## Additional information

By default the code uses 3 cpus to run in parallel each scaler: minmax, std and no scaler
you can modify this value inside ADRanker class
```sh
$  # core/AnomalyDetection.py
$  class ADRanker(UtilMethods):
$
$   def __init__(self, data, models):
$       self.n_cpus = 3
```


Feel free to contact me if you have any dubts.
