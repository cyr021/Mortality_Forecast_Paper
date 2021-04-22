# -*- coding: utf-8 -*-
"""
@author: B
"""
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
import matplotlib.pyplot as plt
import argparse
import time
from src.tools.plot_utils import *
from src.data.data_utils import *
from src.models.model_utils import train, calculate_hltest, test_data
from config import *
warnings.filterwarnings('ignore')

# extract the arguments
parser = argparse.ArgumentParser(description='run parameter optimization for mortality forecast model')
parser.add_argument("--train", default=True)
args = parser.parse_args()

# a set of configuration parameters based on training result
group_types = ['control', 'experimental']
names = ["Logistic Regression", "Decision Tree", "Random Forest"]
models = ['LogisticRegression', 'DecisionTree', "RandomForest"]

# pipeline
for group_type in group_types:
	for i, model in enumerate(models):
		df = load_data(group_type=group_type, usecols='usecols_'+group_type)
		X_train, X_test, y_train, y_test = preprocessing(
								 df,
								 list_cat_var=['admissiontype','gender'],
								 list_ts_var='list_ts_var_'+group_type)
		if args.train:
			train(X_train, X_test, y_train, y_test, model, show_params_refer=True)
		else:
			test_data(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
				  k='k_'+group_type[0], model_name=model, group_type=group_type)

# compute sapsii and plot auc
df = load_data(filename='data/sapsii.csv', usecols=['sapsii', 'sapsii_prob', 'expire_flag'], sapsii=True)
sapsii_score = roc_auc_score(y_true=np.array(df['expire_flag']), y_score=np.array(df['sapsii_prob']))
plot_auc(y_test=np.array(df['expire_flag']), y_score=np.array(df['sapsii_prob']), model_name='SAPSII')
print('sapsii score: {}'.format(sapsii_score))

