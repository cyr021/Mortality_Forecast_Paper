from scipy.stats import norm
from scipy import stats
import warnings
import datetime
import scipy.sparse
# from kmeans_smote import KMeansSMOTE
from numpy import *
# import statsmodels.api as sm
# from sklearn.tree import export_gr
import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler
from src.tools.utils import *
import numpy as np
from src.tools.plot_utils import *
from imblearn.over_sampling import SMOTE, SMOTENC

def collect_sapsii(conn):
          
    db_get_query_from_file(conn, "src/data/SQL/collect_sapsii.sql")
    
    print("SAPSii collected.")

def load_data(group_type=None, usecols=None, filename=None, sapsii=False):
    """
    Args:
        selector - a selector for one specific feature selection method
        group_type - control or experimental
        sapsii - load sapsii data or experimental/control

    Returns:
        tidy dataframe removing useless columns

    Exp:
        expire_flag
        0    225
        1     86
    """
    df = pd.read_csv(filename, header = 0, sep=',', encoding='gb18030', 
                        usecols=usecols, float_precision='round_trip')

    if sapsii is False:
        filename = 'data/mpkl-' + group_type + '.csv'
        for column in df:
            if df.dtypes[column] == np.float64:
                df.groupby(["expire_flag"])[column].mean().to_csv(r'data/statistics-'+group_type+'.csv', header=True, index=True, mode='a')
                df.groupby(["expire_flag"])[column].std().to_csv(r'data/statistics-'+group_type+'.csv', header=True, index=True, mode='a')
            else:
                df.groupby(["expire_flag",column])[column].count().to_csv(r'data/statistics-'+group_type+'.csv', header=True, index=True, mode='a')

            print('statistices for age: \n', 'mean\n', df['age'].groupby(df['expire_flag']).mean(),
                '\n', 'std\n', df['age'].groupby(df['expire_flag']).std())
            plot_data(df, cols=['gender'], type=2)
            plot_data(df, cols=['age'], type=5)
            plot_data(df, cols=['admissiontype'], type=2)
            plot_data(df, cols=['age','heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min', 'tempc_max',
                                'tempc_min', 'pao2fio2_vent_min', 'urineoutput', 'bun_min', 'bun_max',
                                'wbc_min', 'wbc_max', 'potassium_min', 'potassium_max', 'sodium_min',
                                'sodium_max', 'bicarbonate_min', 'bicarbonate_max', 'bilirubin_min',
                                'bilirubin_max', 'mingcs','expire_flag'], type=3
                      )
    else:
        plot_data(df, cols=['sapsii'], type=5)
        print('statistices for sapsii: \n', 'mean\n', df['sapsii'].groupby(df['expire_flag']).mean(), '\n', 'std\n', df['sapsii'].groupby(df['expire_flag']).std())

    # print('列名:\n', df.columns, '\n\n',
    #       '数据类型：\n', df.dtypes, '\n\n',
    #       '描述列:\n', df[d_column].describe(), '\n\n',
    #       '列值：\n', df[u_column].unique(), '\n\n',
    #       '统计列：\n', df['icustay_id'].groupby(df['expire_flag']).count(), '\n\n',
    #       '缺失值：\n', pd.DataFrame({'feature': df.apply(lambda x:sum(x.isnull())/df.shape[0])}).sort_values('feature', ascending=False))

    return df

def preprocessing(df, list_cat_var, list_ts_var):
    
    local_output_dir = 'data/processed/'
    
    print("#------- Custom filling NaN -------#")
    df_fill, list_ts_var = custom_fillna(df, list_ts_var)
    print('DONE')
    
    print("#------- Splitting into train and test sets -------#")
    X_train, y_train, X_test, y_test = self_train_test_split(df_fill)
    print('DONE')

    print("#------- oversampling -------#")
    X_train, y_train, X_test, y_test = smote_impute(X_train, y_train, X_test, y_test)
    print('DONE')
    
    print("#------- scales features -------#")
    X_train, y_train, X_test, y_test = scale_features(X_train, y_train, X_test, y_test, list_cat_var, list_ts_var)
    print('DONE')

    print("Preprocessing ended")
    return X_train, X_test, y_train, y_test
   
def custom_fillna(df, list_ts_var):
    """
    Function to:
    - fill na with the average value of the normal range for this item for each patient

    Args:
        df (dataframe)

    Returns:
        array
    """
    print('len(list_ts_var):', len(list_ts_var))
    df_mean = df[list_ts_var].apply(np.mean, axis=0)
    dict_refer = df_mean.to_dict()

    df_null = pd.DataFrame({'feature': df.apply(lambda x:sum(x.isnull())/df.shape[0])}).sort_values('feature', ascending=False)
    df_null = df_null[df_null.feature>0.50]
    removecols = df_null.index.tolist()
    print("移除缺失率超过50%的feature： {}".format(removecols))

    df = df.drop(removecols, axis=1)
    df = df.fillna(value=dict_refer)
    list_ts_var = list(set(list_ts_var).difference(set(removecols)))
    print('len(list_ts_var):', len(list_ts_var))
    print(df.columns)

    return df, list_ts_var

def self_train_test_split(df):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.25, random_state=0)
    X_train, y_train, X_test, y_test = train.loc[:, train.columns != 'expire_flag'], train['expire_flag'], test.loc[:, train.columns != 'expire_flag'], test['expire_flag']

    print("Number transactions X_train dataset: ", X_train.shape)
    print("Number transactions y_train dataset: ", y_train.shape)
    print("Number transactions X_test dataset: ", X_test.shape)
    print("Number transactions y_test dataset: ", y_test.shape)

    return X_train, y_train, X_test, y_test

def label_encoding_cat_variables(X_train, y_train, X_test, y_test, list_cat_var):

    from sklearn import preprocessing

    for colname in list_cat_var:
        le = preprocessing.LabelEncoder()
        X_train[colname] = le.fit_transform(X_train[colname])
    #     Y_train[colname] = le.fit_transform(Y_train[colname])
        X_test[colname] = le.fit_transform(X_test[colname])
    #     Y_test[colname] = le.fit_transform(Y_test[colname])

    return X_train, y_train, X_test, y_test

def scale_features(X_train, y_train, X_test, y_test, list_cat_var, list_ts_var):
    """
    Function to scale data by:
    - Label encoding 'list_cat_var' variables
    - MinMax scaling 'list_ts_var' variables

    Args:
        X_train (dataframe)
        y_train (dataframe)
        X_test (dataframe)
        y_test (dataframe)
        list_cat_var (array)

    Returns:
        scaled input
        array of LabelEncoder objects
    """
    X_train_scaled = X_train.copy()
    y_train_scaled = y_train.copy()
    X_test_scaled = X_test.copy()
    y_test_scaled = y_test.copy()

    # Label encoding cat variables
    (X_train_scaled, y_train_scaled,
     X_test_scaled, y_test_scaled) = label_encoding_cat_variables(X_train_scaled, y_train_scaled,
                                                             X_test_scaled, y_test_scaled, list_cat_var)

    # Scaling variables
    list_vars = list_ts_var
    var_scaler = MinMaxScaler([0, 1])

    X_train_scaled[list_vars] = var_scaler.fit_transform(X_train_scaled[list_vars])
    
    X_test_scaled[list_vars] = var_scaler.transform(X_test_scaled[list_vars])

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

def smote_impute(X_train, y_train, X_test, y_test):
    # unbalanced data
    # print('categorical features: {}'.format(X_train.columns))
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
    # print(sorted(Counter(Y_train_scaled).items()))
    print(X_train.columns)

    # sm = SMOTE(random_state=2)
    sm = SMOTENC(categorical_features=[19,20], random_state=0)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
    X_test_res, y_test_res = sm.fit_sample(X_test, y_test.ravel())

    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

    # from imblearn.over_sampling import SMOTE, ADASYN
    # X_train_scaled_resampled, Y_train_scaled_resampled = SMOTENC(categorical_features=[0, 2], random_state=0).fit_resample(X_train_scaled, Y_train_scaled)
    # X_test_scaled_resampled, Y_test_scaled_resampled =  SMOTENC(categorical_features=[0, 2], random_state=0).fit_resample(X_test_scaled, Y_test_scaled)

    # print('Dataset after resampling:')
    # print(sorted(Counter(Y_train_scaled_resampled).items()))
    # plot_resampling(X, y, sampler, ax1)

    return X_train_res, y_train_res.ravel(), X_test_res, y_test_res.ravel()

