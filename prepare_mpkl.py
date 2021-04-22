# -*- coding: utf-8 -*-
"""
@author: B
"""
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns 
import re
%matplotlib inline 
import numpy as np

usecols = ['subject_id', 'hadm_id', 'icustay_id', 'age','heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
					 'tempc_max','tempc_min', 'pao2fio2_vent_min', 'urineoutput', 'bun_min',
					 'bun_max','wbc_min', 'wbc_max', 'potassium_min', 'potassium_max', 'sodium_min',
			 'sodium_max', 'bicarbonate_min', 'bicarbonate_max', 'bilirubin_min',
			 'bilirubin_max', 'mingcs', 'aids', 'hem', 'mets', 'admissiontype',
			 'gender', 't_day', 'expire_flag'
			 ]
cols = ['subject_id', 'hadm_id', 'icustay_id','heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
					 'tempc_max','tempc_min', 'pao2fio2_vent_min', 'urineoutput', 'bun_min',
					 'bun_max','wbc_min', 'wbc_max', 'potassium_min', 'potassium_max', 'sodium_min',
			 'sodium_max', 'bicarbonate_min', 'bicarbonate_max', 'bilirubin_min',
			 'bilirubin_max', 'mingcs', 'aids', 'hem', 'mets']

df = pd.read_csv(u'data/mpkl-experimental-raw.csv', encoding='gb18030', usecols=usecols, float_precision='round_trip')
new_names = ['heartrate_max_1', 'heartrate_min_1', 'sysbp_max_1', 'sysbp_min_1',
					 'tempc_max_1','tempc_min_1', 'pao2fio2_vent_min_1', 'urineoutput_1', 'bun_min_1',
					 'bun_max_1','wbc_min_1', 'wbc_max_1', 'potassium_min_1', 'potassium_max_1', 'sodium_min_1',
			 'sodium_max_1', 'bicarbonate_min_1', 'bicarbonate_max_1', 'bilirubin_min_1',
			 'bilirubin_max_1', 'mingcs_1', 'aids_1', 'hem_1', 'mets_1']
old_names = ['heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min',
					 'tempc_max','tempc_min', 'pao2fio2_vent_min', 'urineoutput', 'bun_min',
					 'bun_max','wbc_min', 'wbc_max', 'potassium_min', 'potassium_max', 'sodium_min',
			 'sodium_max', 'bicarbonate_min', 'bicarbonate_max', 'bilirubin_min',
			 'bilirubin_max', 'mingcs', 'aids', 'hem', 'mets']
df1 = df[df['t_day']==1]
df1.rename(columns=dict(zip(old_names, new_names)), inplace=True)

new_names = ['heartrate_max_2', 'heartrate_min_2', 'sysbp_max_2', 'sysbp_min_2',
					 'tempc_max_2','tempc_min_2', 'pao2fio2_vent_min_2', 'urineoutput_2', 'bun_min_2',
					 'bun_max_2','wbc_min_2', 'wbc_max_2', 'potassium_min_2', 'potassium_max_2', 'sodium_min_2',
			 'sodium_max_2', 'bicarbonate_min_2', 'bicarbonate_max_2', 'bilirubin_min_2',
			 'bilirubin_max_2', 'mingcs_2', 'aids_2', 'hem_2', 'mets_2']
df2 = df[df['t_day']==2]
df2.rename(columns=dict(zip(old_names, new_names)), inplace=True)

new_names = ['heartrate_max_3', 'heartrate_min_3', 'sysbp_max_3', 'sysbp_min_3',
					 'tempc_max_3','tempc_min_3', 'pao3fio3_vent_min_3', 'urineoutput_3', 'bun_min_3',
					 'bun_max_3','wbc_min_3', 'wbc_max_3', 'potassium_min_3', 'potassium_max_3', 'sodium_min_3',
			 'sodium_max_3', 'bicarbonate_min_3', 'bicarbonate_max_3', 'bilirubin_min_3',
			 'bilirubin_max_3', 'mingcs_3', 'aids_3', 'hem_3', 'mets_3']
df3 = df[df['t_day']==3]
df3.rename(columns=dict(zip(old_names, new_names)), inplace=True)

df_new = df1.merge(df2[['subject_id', 'hadm_id', 'icustay_id','heartrate_max_2', 'heartrate_min_2', 'sysbp_max_2', 'sysbp_min_2',
					 'tempc_max_2','tempc_min_2', 'pao2fio2_vent_min_2', 'urineoutput_2', 'bun_min_2',
					 'bun_max_2','wbc_min_2', 'wbc_max_2', 'potassium_min_2', 'potassium_max_2', 'sodium_min_2',
			 'sodium_max_2', 'bicarbonate_min_2', 'bicarbonate_max_2', 'bilirubin_min_2',
			 'bilirubin_max_2', 'mingcs_2', 'aids_2', 'hem_2', 'mets_2']],how='inner',left_on=['subject_id', 'hadm_id', 'icustay_id'],right_on=['subject_id', 'hadm_id', 'icustay_id'])
df_new = df_new.merge(df3[['subject_id', 'hadm_id', 'icustay_id','heartrate_max_3', 'heartrate_min_3', 'sysbp_max_3', 'sysbp_min_3',
					 'tempc_max_3','tempc_min_3', 'pao3fio3_vent_min_3', 'urineoutput_3', 'bun_min_3',
					 'bun_max_3','wbc_min_3', 'wbc_max_3', 'potassium_min_3', 'potassium_max_3', 'sodium_min_3',
			 'sodium_max_3', 'bicarbonate_min_3', 'bicarbonate_max_3', 'bilirubin_min_3',
			 'bilirubin_max_3', 'mingcs_3', 'aids_3', 'hem_3', 'mets_3']],how='inner',left_on=['subject_id', 'hadm_id', 'icustay_id'],right_on=['subject_id', 'hadm_id', 'icustay_id'])
df_new = df_new.drop(['t_day'], axis=1)
df_new.to_csv('data/mpkl-experimental.csv', index=False)