import pprint
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectPercentile
from sklearn import svm
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
import datetime
import scipy.sparse
# from kmeans_smote import KMeansSMOTE
from numpy import *
# import statsmodels.api as sm
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import graphviz
import re

def plot_resample_comparison(X, y):
    from imblearn.base import BaseSampler
    
    def plot_resampling(X, y, sampling, ax):
        X_res, y_res = sampling.fit_resample(X, y)
        ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
        # make nice plotting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        return Counter(y_res)

    class FakeSampler(BaseSampler):

        _sampling_type = 'bypass'

        def _fit_resample(self, X, y):
            return X, y


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

    sampler = FakeSampler()
    plot_resampling(X, y, sampler, ax1)
    ax1.set_title('Original data - y={}'.format(Counter(y)))
                 
    sampler = SMOTENC(categorical_features=[0, 2], random_state=0)
    plot_resampling(X, y, sampler, ax2)
    ax.set_title('Resampling using {}'.format(sampler.__class__.__name__))
    fig.tight_layout()


def plot_data(df, cols=None, type=None):
    """plot type
    1- distribution plot
    3- correlation matrix
    5- categorical plot"""

    if type == 1:
        plt.figure(figsize=(18, 9))
        df[cols].boxplot()
        plt.title("Numerical variables in this dataset", fontsize=20)
        plt.savefig('output/Numerical.png')
    elif type == 2:
        fig = plt.figure()
        # ax = fig.add_subplot(1,3,i)
        var = cols[0]
        data = pd.concat([df['expire_flag'], df[var]], axis=1)
        # f, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=var, hue='expire_flag', data=data)
        # fig.axis(ymin=0, ymax=800000)
        plt.savefig('output/'+ var + '_count.png')
    elif type == 3:
        fig = plt.figure()
        corrmat = df[cols].corr()
        # print(data_corr)
        # data_corr.fillna(value=np.nan, inplace=True)
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True, xticklabels=False, yticklabels=1)
        plt.title('Correlation Matric')
        plt.savefig('output/Correlation_Matric.png')
        # plt.show()
    elif type == 4:
        f, ax = plt.subplots(figsize=(5, 4))
        def change_width(ax, new_value) :
            for patch in ax.patches :
                current_width = patch.get_width()
                diff = current_width - new_value

                # we change the bar width
                patch.set_width(new_value)

                # we recenter the bar
                patch.set_x(patch.get_x() + diff * .5)

        # Draw a nested barplot by species and sex
        sns.barplot(data=df, x="model", y="auc", hue="type")
        change_width(ax, .25)
        plt.gca().legend().set_title('')
        plt.legend(loc='best')
        plt.xlabel("Models")
        plt.ylabel("AUC values")
        # plt.suptitle('Model Comparison Between Control Group and Experimental Group',size=16)
        # plt.savefig('output/Model_Comparison.png')
        plt.show()

    elif type == 5:
        from scipy.stats import norm
        fig = plt.figure()
        # ax = fig.add_subplot(1,3,i)
        # fig, ax = plt.subplots()
        var = cols[0]
        data = pd.concat([df['expire_flag'], df[var]], axis=1)
        # print(df[var].value_count:())
        # f, ax = plt.subplots(figsize=(8, 6))
        sns.distplot(data[data['expire_flag'] == 0][var],
                     color='r',
                     kde_kws={"lw": 2, "label": "expire_flag_0"})
        sns.distplot(data[data['expire_flag'] == 1][var],
                     color='b',
                     kde_kws={"lw": 2, "label": "expire_flag_1"})
        plt.tight_layout()
        plt.savefig('output/'+ var + '_distribution.png')



def plot_feature_importance(model, X_train_new, model_name):
    print(model_name)
    color = {'DecisionTree': 'skyblue',
            'LogisticRegression': 'navy',
            'RandomForest': 'darkorange'
             }
    output_path = 'output/feature-' + str(model_name) + '.png'

    classifier = model_name.split('-')[0]
    print(classifier)

    if classifier == 'DecisionTree':
        feature_importance = model.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        # sorted_idx = np.argsort(feature_importance)[::-1]
    elif classifier == 'LogisticRegression':
        feature_importance = abs(model.coef_[0])
    else:
        feature_importance = model.feature_importances_

    print(feature_importance)

    # feature_importance = model.feature_importances_
    # customized number 
    num_features = 10 
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # feature_importance = feature_importance[feature_importance>20]
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(num_features) + .5
    print(feature_importance)
    print(sorted_idx)

    featfig = plt.figure()
    featax = featfig.add_subplot(111)
    featax.barh(pos, feature_importance[sorted_idx[-num_features:]], align='center', color='skyblue')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(X_train_new.columns)[sorted_idx[-num_features:]], fontsize=11)
    featax.set_xlabel('Relative Feature Importance - {}'.format(model_name))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_auc(y_test, y_score, model_name):
    fig = plt.figure()
    output_path = 'output/auc_' + model_name + '.png'

    # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    fpr, tpr, _ = roc_curve(y_test, y_score)
    # roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(y_test, y_score)
    print("auc score for test data:", roc_auc)
    
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig(output_path)
    # print('{} plotted'.format(output_path))


def visualize_tree(model, data_feature_names):
    import pydotplus
    import collections
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    # Visualize data
    dot_data = export_graphviz(model,
                                    feature_names=data_feature_names,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("../output/tree")

def plot():
    x = np.arange(8,25,1)
    y = aucs
    ymax = max(y)
    xpos = y.index(ymax)
    xmax = x[xpos]
    # print(y)
    print(clf, 'max(x,y): ', xmax, ',', ymax)
    print('plotting...')
    plt.subplot(3, 1, num+1)
    plt.xlabel("Number of features selected - {}".format(clf))
    plt.plot(np.arange(8,25,1), aucs, color=color[i], label=score_func) # linestyle
    plt.plot(xmax, ymax, 'ro', marker='x', markersize = 10)
    # Not ticks everywhere
    if num in [0,1]:
        plt.tick_params(labelbottom='off')

    # Same limits for everybody!
    plt.xlim(8,25)
    plt.ylim(0.5,1.0)


        ## Add title
    # plt.title(clf, loc='left', fontsize=10, fontweight=0, color='black')
    # print(aucs)
    
    # plt.text(0.5, 0.02, 'Number of features selected', ha='center', va='center')
    plt.text(0.06, 0.5, 'AUC score of number of selected features', ha='center', va='center', rotation='vertical')
    
    plt.legend()
    plt.savefig('../output/feature_selection_backup.png')
    plt.show()


def plot_classifier_comparison():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    h = .02  # step size in the mesh

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]

    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()