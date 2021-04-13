from core import UtilMethods
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed


class ADRanker(UtilMethods):

    def __init__(self, data, models):
        self.n_cpus = 3
        self.input_folder = data
        self.datasets = self.get_folders(self.input_folder)
        self.models = models
        self.prediction_path = "results/predictions"
        self.scores_path = "results/scores"
        self.data_excepts = ["glass-0-1-6_vs_2", "glass-0-1-6_vs_5", "shuttle-c0-vs-c4", "shuttle-c2-vs-c4",
                             "yeast-1_vs_7"]

    @staticmethod
    def split_dataset(train, test, scaler='no'):
        ohe = OneHotEncoder(sparse=True)
        objInTrain = len(train)
        allData = pd.concat([train, test], ignore_index=True, sort=False, axis=0)
        AllDataWithoutClass = allData.iloc[:, :-1]
        AllDataWithoutClassOnlyNationals = AllDataWithoutClass.select_dtypes(include=['object'])
        AllDataWithoutClassNoNominal = AllDataWithoutClass.select_dtypes(exclude=['object'])

        encAllDataWithoutClassNominal = ohe.fit_transform(AllDataWithoutClassOnlyNationals)
        encAllDataWithoutClassNominalToPanda = pd.DataFrame(encAllDataWithoutClassNominal.toarray())

        if AllDataWithoutClassOnlyNationals.shape[1] > 0:
            codAllDataAgain = pd.concat([encAllDataWithoutClassNominalToPanda, AllDataWithoutClassNoNominal],
                                        ignore_index=True, sort=False, axis=1)
        else:
            codAllDataAgain = AllDataWithoutClass

        X_train = codAllDataAgain[:objInTrain]
        y_train = train.values[:, -1]

        X_test = codAllDataAgain[objInTrain:]
        y_test = test.values[:, -1]

        if scaler == 'no':
            X_train = X_train.values
            X_test = X_test.values
            return X_train, X_test, y_train, y_test
        else:
            if scaler == 'minmax':
                actual_scaler = MinMaxScaler()

            else:
                actual_scaler = StandardScaler()

            X_train = pd.DataFrame(actual_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index,
                                   columns=X_train.columns)
            X_test = pd.DataFrame(actual_scaler.transform(X_test[X_test.columns]), index=X_test.index,
                                  columns=X_test.columns)
            X_train = X_train.values
            X_test = X_test.values
            return X_train, X_test, y_train, y_test

    def get_prediction(self, train_file, test_file, model, name, scaler, dataset):

        try:
            self.create_folder(f'{self.prediction_path}/{scaler}/{dataset}')

            def prediction(x_test, current_model, model_name):
                if 'PY' in model_name:
                    y_prediction = current_model.decision_function(x_test)
                else:
                    y_prediction = current_model.score_samples(x_test)
                return y_prediction

            train = pd.read_csv(train_file, sep=',')
            test = pd.read_csv(test_file, sep=',')

            X_train, X_test, y_train_len, y_test = self.split_dataset(train, test, scaler)
            y_train = ['negative' for _ in range(len(y_train_len))]

            if 'PY' in name:
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)
            y_pred = prediction(X_test, model, name)
            predictions_frame = pd.DataFrame({"real": y_test,
                                              "prediction": y_pred})
            predictions_frame.to_csv(f'{self.prediction_path}/{scaler}/{dataset}/{name.split("_")[1]}.csv', index=False)
        except:
            file = open('error_log.txt', 'a')
            file.write(f"{model},{dataset}, {scaler}\n")

    def get_predictions(self):

        def get_tra_tst_files(path):
            files = self.get_files(path)
            tra_f = [file for file in files if "tra" in file][0]
            tra_f = f'{path}/{tra_f}'
            tst_f = [file for file in files if "tst" in file][0]
            tst_f = f'{path}/{tst_f}'
            return tra_f, tst_f

        for name, model in tqdm(self.models.items(), desc="Loading model..."):
            for dataset in tqdm(self.datasets, desc="Loading dataset..."):
                if dataset in self.data_excepts:
                    continue
                tra, tst = get_tra_tst_files(f'{self.input_folder}/{dataset}')

                Parallel(n_jobs=self.n_cpus) \
                    (delayed(self.get_prediction)
                     (train_file=tra,
                      test_file=tst,
                      model=model,
                      name=name,
                      scaler=scaler,
                      dataset=dataset)
                     for scaler in ['no', 'minmax', 'std'])

    def get_scores(self, scores):
        for score_name, score in scores.items():
            print("score:", score_name)
            for scaler in self.get_folders(self.prediction_path):
                results = {}
                for dataset in tqdm(self.get_folders(f'{self.prediction_path}/{scaler}'), desc="datasets..."):
                    metric_results = []
                    classifier_names = []
                    for classifier in self.get_files(f'{self.prediction_path}/{scaler}/{dataset}'):
                        value = score(f'{self.prediction_path}/{scaler}/{dataset}/{classifier}')
                        metric_results.append(value)
                        classifier_names.append(classifier.split(".")[0])
                    results[dataset] = metric_results

                out = pd.DataFrame(results)
                out.insert(0, 'model', classifier_names)
                out.to_csv(f'{self.scores_path}/{score_name}/{scaler}/results.csv', index=False)


