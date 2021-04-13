import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import Orange
from autorank import autorank


class Plots:

    @staticmethod
    def get_box_plot_data(labels, bp):
        rows_list = []
        for i in range(len(labels)):
            dict1 = {'label': labels[i], 'lower_whisker': bp['whiskers'][i * 2].get_ydata()[1],
                     'lower_quartile': bp['boxes'][i].get_ydata()[1], 'median': bp['medians'][i].get_ydata()[1],
                     'upper_quartile': bp['boxes'][i].get_ydata()[2],
                     'upper_whisker': bp['whiskers'][(i * 2) + 1].get_ydata()[1]}
            rows_list.append(dict1)
        return pd.DataFrame(rows_list)

    @staticmethod
    def transform_data(results):
        base = results.copy()
        models = list(results.model)
        results = results.drop(columns=['model'])
        results = results.values
        results = results.T
        bp = plt.boxplot(results)
        info = Plots.get_box_plot_data(models, bp)
        plt.close()
        base['median'] = list(info['median'])
        base = base.sort_values(by=['median'])
        models = list(base.model)
        base = base.drop(columns=['model', 'median'])
        base = base.values
        base = base.T
        return base, models

    @staticmethod
    def make_plot_basic(paths, scalers):

        locs = [(0, 0), (1, 0), (2, 0),
                (0, 1), (1, 1), (2, 1)]
        datas = []
        names = []

        for path in paths:
            a = pd.read_csv(path)
            data, name = Plots.transform_data(a)
            datas.append(data)
            names.append(name)
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))

        tests = ['']*6
        i = 0
        for data, name, scaler in zip(datas, names, scalers):
            sns.boxplot(ax=axes[locs[i]], data=data)
            axes[locs[i]].set_title(scaler)
            axes[locs[i]].set_xticklabels(names[i], rotation=90)
            i += 1

        axes[(0, 1)].set_yticklabels(tests[0])
        axes[(1, 1)].set_yticklabels(tests[0])
        axes[(2, 1)].set_yticklabels(tests[0])

        plt.suptitle("Area Under the ROC Curve                                              Average Precision",
                     fontweight="bold")
        plt.savefig('results/images/summarize.png', bbox_inches='tight')
        plt.savefig("results/images/sum.svg")
        plt.show()
        plt.close()

    @staticmethod
    def make_bar_plot(paths, scaler_names):

        Plots.make_plot_basic(paths, scaler_names)

    @staticmethod
    def saveCD(data, name='test', title='title'):
        models = list(data.model)
        data = data.drop(columns=['model'])
        data = data.transpose()
        data.columns = models
        result = autorank(data, alpha=0.05, verbose=False)
        critical_distance = result.cd
        rankdf = result.rankdf
        avranks = rankdf.meanrank
        ranks = list(avranks.values)
        names = list(avranks.index)
        names = names[:30]
        avranks = ranks[:30]
        Orange.evaluation.graph_ranks(avranks, names, cd=critical_distance, width=10, textspace=1.5, labels=True)
        plt.suptitle(title)
        plt.savefig('results/images/svg/' + name + ".svg", format="svg")
        plt.savefig('results/images/png/' + name + ".png", format="png")
        plt.show()
        plt.close()

    @staticmethod
    def make_cd_plot(paths, names, titles):
        for results, name, title in zip(paths, names, titles):
            r = pd.read_csv(results)
            Plots.saveCD(r, name, title)
