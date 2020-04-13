import pprint
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc as auc_func
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
from kmeans_smote import KMeansSMOTE
from numpy import *
import statsmodels.api as sm
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import graphviz
import re

def plot_data(df, cols=None, type=None):
    if type == 1:
        plt.figure(figsize=(18, 9))
        df[cols].boxplot()
        plt.title("Numerical variables in this dataset", fontsize=20)
        plt.savefig('../output/Numerical.png')
    elif type == 2:
        fig = plt.figure()
        # ax = fig.add_subplot(1,3,i)
        var = cols[0]
        data = pd.concat([df['expire_flag'], df[var]], axis=1)
        # print(df[var].value_count:())
        # f, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=var, hue='expire_flag', data=data)
        # fig.axis(ymin=0, ymax=800000)
        plt.savefig('../output/Categorical.png')
    elif type == 3:
        fig = plt.figure()
        data_corr = df.corr()
        print(data_corr)
        # data_corr.fillna(value=np.nan, inplace=True)
        sns.heatmap(data_corr, xticklabels=1, yticklabels=1)
        plt.title('Correlation Matric')
        plt.savefig('../output/Correlation_Matric.png')
        plt.show()
    elif type == 4:
        fig = plt.figure()
        sns.pointplot(x='model', y='auc', data=df)
        plt.title('Model Comparison')
        plt.savefig('../output/Model_Comparison.png')
        # plt.show()


def plot_feature_importance(model, X=None, text=None):
    # print(model)
    color = {'DecisionTreeClassifier': 'skyblue',
            'LogisticRegression': 'navy',
            'RandomForestClassifier': 'darkorange'
             }
    model_name = re.split('\(', str(model))[0]
    output_path = '../output/file/feature-' + model_name + '-' + text + '.png'

    if model_name == 'DecisionTreeClassifier':
        feature_importance = model.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        # sorted_idx = np.argsort(feature_importance)[::-1]
    elif model_name == 'LogisticRegression':
        feature_importance = abs(model.coef_[0])
    else:
        feature_importance = model.feature_importances_

    # print(feature_importance)
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    feature_importance = feature_importance[feature_importance>0]
    print(feature_importance)
    print(X.columns)
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    featfig = plt.figure()
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(pos, feature_importance[sorted_idx], align='center', color=color[model_name])
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
    featax.set_xlabel('Relative Feature Importance - {}'.format(model_name))

    plt.tight_layout()
    plt.savefig(output_path)
    # plt.show()


def plot_auc(model, fpr, tpr, roc_auc, text):
    model_name = re.split('\(', str(model))[0]
    output_path = '../output/file/auc-' + model_name + '-' + text + '.png'
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    # plt.show()


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