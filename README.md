## Introduction
In this project, we selected ICU patients with pelvic and/or acetabular fractures based on ICD-9 codes from MIMIC-III database. Then, we used three mainstream machine learning algorithms (logistic regression, decision tree and random forest) to establish models, and compare their predictive performance with that of the Simplified Acute Physiology Score (SAPS), a standard prediction system currently used in clinic, via the Receiver Operating Characteristic (ROC) curve and Area Under ROC curve (AUROC). To improve the performance of machine learning-based models, we expanded the range of variable selection by the combination of SAPS-based parameters and more laboratory indicators.

This project is a python implementation of our work.

## How to use the project
### Source code
Source code is made up of 2 main scripts:

1. `prepare_mpkl`: recreate the processed data to be used for the ML model
2. `main.py`: train multiple machine learning models on control and experimental groups, and then compare the performance to that of SAPS

Below you can find more details on these scripts.

### Data Creation
#### Database build
Starting from a fresh system which has GNU Make installed, PostgreSQL installed, and a local copy of this repository, an instance of the the local mimic database with the usefule raw data from PhysioNet. Source code for the local database refers to [MiMic code repository](https://github.com/MIT-LCP/mimic-code).

#### Data prepration
We selected the patients who stay in ICU over 3 days to ensure the timeliness of the mortality prediction.

Variables used in SAPS:
-  Age, GCS
-  VITALS: Heart rate, systolic blood pressure, temperature, respiration rate
-  FLAGS: ventilation/cpap
-  IO: urine output
-  LABS: blood urea nitrogen, hematocrit, WBC, glucose, potassium, sodium, HCO3

The processed data collected is composed of 3 parts: 

1. **Static data**: demographic data including the age, date of birth, various date of death, admission time, discharge time and admission type
2. **ICU acquired data**: ICU related statistcs such as ICU stay duration and the average ICU stays
3. **Hospital acquired data**: chartevents in the first 3 days after being sent to ICU/transformation for patients with Severe pelvic and acetabular fractures

The data will be written in one `csv` formatted file, followed by being loaded in the next data modeling step.

### Pre-Processing
This phase will perform the following transformations on raw data:
- Removing features having over 50% missing value
- Fixing outliers or remove outliers
- One-hot encoding for categorical data
- Filling NaN with mean value at feature level
- Standard scale features
- Oversampling to balance data using KMeansSMOTE method
- Feature selection using mutual_info_classif and f_classif

## License
Licensed under the Apache License, Version 2.0.
