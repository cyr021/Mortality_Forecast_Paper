#!/usr/bin/python
# -*- coding:utf-8 -*-

from src.models.model_utils import *
from src.data.data_utils import load_data
from sklearn import model_selection
import pprint
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import graphviz
import re
from src.data.data_utlis import *
warnings.filterwarnings('ignore')
# pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',15)
pd.set_option('display.width',10000)
sns.set(style="darkgrid")

filename = '../data/out1028.csv'
score_funcs = ['mutual_info_classif','f_classif']
clfs = ['lg','rf','tree']
color = ['skyblue','green']
col_types = {'expire_flag': 'category',
             'gender': 'category',
            'insurance': 'category',
             'ethnicity': 'category',
             'religion': 'category',
             'marital_status': 'category',
             'language': 'category',
             'admission_type': 'category',
             'admission_location': 'category'
             }
print('current dir is: ', os.getcwd())
seed = 5

results = []
names = []
scoring = 'accuracy'

plt.figure()

# evaluate each model in turn
for score_func in score_funcs:
    for num, clf in enumerate(clfs):
        results = []
        for k in range(8,25):
            name = score_func + '-' + str(k)
            selector = SelectKBest(score_func, k=k)
            X, y = load_data(filename, col_types, selector=selector)
            x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.6)
            model = get_model(clf)
            kfold = model_selection.KFold(n_splits=5, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (clf, cv_results.mean(), cv_results.std())
            print(msg)

            # plot_feature_importance(model, X=X_pd, text=text)
            # auc_df = pd.DataFrame({'model': ['LR', 'RF', 'Tree'],
            #                        'auc': roc_lst})
            # plot_data(auc_df, type=4)
            # fpr, tpr, thresholds = roc_curve(y_test, y_score)
            # roc_auc = auc_func(fpr, tpr)
            # plot_auc(model, fpr, tpr, roc_auc, text=text)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
# max(x,y):  17 , 0.9378774805867127
# max(x,y):  19 , 0.9393155018694277
# max(x,y):  21 , 0.8549036525740581





