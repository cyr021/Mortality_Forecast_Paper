# -*- coding: utf-8 -*-
"""
@author: B
"""
k_control = [20, 19, 18]
usecols_control = ['age','heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min','tempc_max',
'tempc_min', 'pao2fio2_vent_min', 'urineoutput', 'bun_min','bun_max','wbc_min', 'wbc_max', 
'potassium_min', 'potassium_max', 'sodium_min', 'sodium_max', 'bicarbonate_min', 
'bicarbonate_max', 'bilirubin_min',
'bilirubin_max', 'mingcs', 'aids', 'hem', 'mets', 'admissiontype',
'gender', 'expire_flag'
]
list_ts_var_control = ['age','heartrate_max', 'heartrate_min', 'sysbp_max', 'sysbp_min', 'tempc_max',
						 'tempc_min', 'urineoutput', 'bun_min', 'bun_max',
						 'wbc_min', 'wbc_max', 'potassium_min', 'potassium_max', 'sodium_min',
						 'sodium_max', 'bicarbonate_min', 'bicarbonate_max', 'mingcs']
k_experimental = [48, 34 , 53]
usecols_experimental = ['age', 'admissiontype', 'gender', 'heartrate_max_day1',
 'heartrate_min_day1', 'sysbp_max_day1', 'sysbp_min_day1', 'tempc_max_day1',
 'tempc_min_day1', 'pao2fio2_vent_min_day1', 'urineoutput_day1', 'bun_min_day1',
 'bun_max_day1', 'wbc_min_day1', 'wbc_max_day1', 'potassium_min_day1',
 'potassium_max_day1', 'sodium_min_day1', 'sodium_max_day1', 'bicarbonate_min_day1',
 'bicarbonate_max_day1', 'bilirubin_min_day1', 'bilirubin_max_day1', 'mingcs_day1',
 'aids_day1', 'hem_day1', 'mets_day1', 'heartrate_max_day2', 'heartrate_min_day2', 'sysbp_max_day2', 'sysbp_min_day2',
 'tempc_max_day2', 'tempc_min_day2', 'pao2fio2_vent_min_day2', 'urineoutput_day2',
 'bun_min_day2', 'bun_max_day2', 'wbc_min_day2', 'wbc_max_day2', 'potassium_min_day2',
 'potassium_max_day2', 'sodium_min_day2', 'sodium_max_day2', 'bicarbonate_min_day2',
 'bicarbonate_max_day2', 'bilirubin_min_day2', 'bilirubin_max_day2', 'mingcs_day2',
 'aids_day2', 'hem_day2', 'mets_day2', 'heartrate_max_day3', 'heartrate_min_day3',
 'sysbp_max_day3', 'sysbp_min_day3', 'tempc_max_day3', 'tempc_min_day3',
 'pao2fio2_vent_min_day3', 'urineoutput_day3', 'bun_min_day3', 'bun_max_day3',
 'wbc_min_day3', 'wbc_max_day3', 'potassium_min_day3', 'potassium_max_day3',
 'sodium_min_day3', 'sodium_max_day3', 'bicarbonate_min_day3',
 'bicarbonate_max_day3', 'bilirubin_min_day3', 'bilirubin_max_day3', 'mingcs_day3',
 'aids_day3', 'hem_day3', 'mets_day3', 'expire_flag'
				 ]
list_ts_var_experimental = ['age', 'heartrate_max_day1',
					 'heartrate_min_day1', 'sysbp_max_day1', 'sysbp_min_day1', 'tempc_max_day1',
					 'tempc_min_day1', 'pao2fio2_vent_min_day1', 'urineoutput_day1', 'bun_min_day1',
					 'bun_max_day1', 'wbc_min_day1', 'wbc_max_day1', 'potassium_min_day1',
					 'potassium_max_day1', 'sodium_min_day1', 'sodium_max_day1', 'bicarbonate_min_day1',
					 'bicarbonate_max_day1', 'bilirubin_min_day1', 'bilirubin_max_day1', 'mingcs_day1',
					 'aids_day1', 'hem_day1', 'mets_day1', 'heartrate_max_day2', 'heartrate_min_day2', 'sysbp_max_day2', 'sysbp_min_day2',
					 'tempc_max_day2', 'tempc_min_day2', 'pao2fio2_vent_min_day2', 'urineoutput_day2',
					 'bun_min_day2', 'bun_max_day2', 'wbc_min_day2', 'wbc_max_day2', 'potassium_min_day2',
					 'potassium_max_day2', 'sodium_min_day2', 'sodium_max_day2', 'bicarbonate_min_day2',
					 'bicarbonate_max_day2', 'bilirubin_min_day2', 'bilirubin_max_day2', 'mingcs_day2',
					 'aids_day2', 'hem_day2', 'mets_day2', 'heartrate_max_day3', 'heartrate_min_day3',
					 'sysbp_max_day3', 'sysbp_min_day3', 'tempc_max_day3', 'tempc_min_day3',
					 'pao2fio2_vent_min_day3', 'urineoutput_day3', 'bun_min_day3', 'bun_max_day3',
					 'wbc_min_day3', 'wbc_max_day3', 'potassium_min_day3', 'potassium_max_day3',
					 'sodium_min_day3', 'sodium_max_day3', 'bicarbonate_min_day3',
					 'bicarbonate_max_day3', 'bilirubin_min_day3', 'bilirubin_max_day3', 'mingcs_day3',
					 'aids_day3', 'hem_day3', 'mets_day3'
					 ]