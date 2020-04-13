from kmeans_smote import KMeansSMOTE
from scipy.stats import norm
from scipy import stats
import warnings
import datetime
import scipy.sparse
from kmeans_smote import KMeansSMOTE
from numpy import *
import statsmodels.api as sm
from sklearn.tree import export_gr
import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler

def load_data(filename, col_types, selector=None, auto=True):
    """
    args:
        filename - a dir where one file located
        selector - a selector for one specific feature selection method
    """
    df = pd.read_csv(filename, header = 0, sep=',', dtype=col_types)
    print(df[['hadm_id','expire_flag','religion']].groupby(['religion','expire_flag']).count())
    print('数据类型：\n', df.dtypes)

   # missing value
    features = pd.DataFrame({'a': df.apply(lambda x:sum(x.isnull())/df.shape[0])}).\
        sort_values('a')
    features = features[features.a < 0.5]

    full_cols = ['Hct_max', 'Hb_max', 'MCHC_max', 'RBC_max',  'RBC_min', 'adlocation', 'ethnicity', 'age', 'icu_no', 'insurance', 'adtype', 'expire_flag', 'gender', 'icu_me']

    # outliers
    df = df.loc[:, features.index]
    df.at[df[df['age'] > 200].index, 'age'] = 80

    # plot_data(df.loc[:, df.columns.isin(full_cols)], type=3)
    # plot_data(df, type=3)

    # unimportant features
    X = df.drop(['expire_flag','dislocation','dob','admittime'], axis=1)
    y = df['expire_flag'].astype('category')

    ############################################################################### feature selection
    # full_cols = [
    #     '51279_itemid_min',
    #     '51277_itemid_min',
    #     '51265_itemid_min',
    #     '51250_itemid_min',
    #     'age',
    #     '51249_itemid_min',
    #     '51248_itemid_min',
    #     '51222_itemid_min',
    #     '51222_itemid_max',
    #     '51221_itemid_min',
    #     '51006_itemid_min',
    #     '50912_itemid_min'
    # ]
    # print(list(col_types.keys()))
    # print(list(col_types.keys()).append('expire_flag'))
    # plot_data(X, cols=['age', 'icu_no', 'icu_me'], type=1)
    # plot_data(df, cols=['gender'], type=2)
    # plot_data(df, cols=full_cols, type=3)

    # one-hot encoding, fill na, standard scale
    X = pd.get_dummies(X).fillna(method='ffill').fillna(df.mean()).drop(['subject_id', 'hadm_id'], axis=1)
    names = X.columns
    X = pd.DataFrame(scale(X))
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=names)
    # print(names(X))
    # print(X.head(3))
    # X = selector.fit_transform(X, y)
    # X = pd.DataFrame(X, columns=names[selector.get_support()])
    # X = pd.DataFrame(X, columns=['Hct_max', 'Hb_max', 'MCHC_max', 'RBC_max',  'RBC_min', 'age'])
    # print(X.head(1))

    # unbalanced data
    [print('Class {} has {} instances before oversampling'.format(label, count))
     for label, count in zip(*np.unique(y, return_counts=True))]

    kmeans_smote = KMeansSMOTE(
        kmeans_args={
            'n_clusters': 3
        },
        smote_args={
            'k_neighbors': 10
        },
        random_state=1
    )
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)
    # print('type(X_resampled): ', type(X_resampled))
    # X =pd.DataFrame(X, columns=names[selector.get_support()])

    [print('Class {} has {} instances after oversampling'.format(label, count))
     for label, count in zip(*np.unique(y_resampled, return_counts=True))]

    X_resampled = pd.DataFrame(scale(X_resampled))
    X_resampled = StandardScaler().fit_transform(X_resampled)

    return X_resampled, y_resampled, X