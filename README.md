## Introduction
In this project, selected ICU patients diagnosed as pelvic and/or acetabular fractures with ICD-9 rules in MIMIC-III electronic medical record database as the object of study; three mainstream machine learning algorithms, logistic regression, decision tree and random forest, were used to establish models, combined with clinical experience and two different methods of variable selection. All the models were compared with the Simplified Acute Physiology Score (SAPS) system via the Receiver Operating Characteristic (ROC) curve and Area Under ROC curve (AUROC), the method of evaluating the predictive performance of the models. 

## Data
We used the Medical Information Mart for Intensive Care (MIMIC)-III, an open source clinical electronic medical record database, which collected about 50,000 ICU patients from the Beth Israel Deaconess Medical Center of Harvard University since 2001.

## Evaluation Metric
The project uses **aucroc curve** as evaluation metric. It is one of the most important evaluation metrics for checking any classification model’s performance. AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.

## Problem Definition
Severe pelvic and acetabular fractures are mostly high-energy injuries, which are mainly characterized by the deep anatomical position of pelvis and acetabulum, and their complicated anatomical relationship. Besides, this kind of fractures is often accompanied by comminution, displacement, and difficult reduction. In addition, fractures of the pelvis and acetabulum are often combined with other injuries and often prone to complications. All these factors result in high risk of surgery and mortality. Therefore, a data-based auxiliary tool is urgently needed to predict the risk of death of patients in order to better guide clinical decision-making by clinicians.

## Methods
Machine leaning methods
- logistic regression
- decistion tree
- random forest

Traditional method
- SAPS

## Pipeline
### Source code
Source code is made up of 5 main scripts:

1. `main_data.R`: clean the data and create the processed data to be used for the ML model
2. `main_model.py`: train multiple machine learning models and compare the performance to that of SAPS

Below you can find more details on these scripts.

### Data Creation
We selected the patients who stay in ICU over 3 days to ensure the timeliness of the mortality prediction.

The processed data is composed of 3 parts: 

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

### Train Model
- LogisticRegression(penalty='l2')
- DecisionTreeClassifier(random_state=0, max_features='sqrt')
- RandomForestClassifier(bootstrap=True, criterion='gini',
                                 max_depth=8, max_features='sqrt', min_impurity_decrease=0.0,
                                 min_samples_leaf=20, min_samples_split=100,
                                 min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                 random_state=9, verbose=0)

## Conclusion
In conclusion, compared with traditional regression analysis, machine learning algorithms are more accurate and efficient in predicting in-hospital mortality risk of ICU patients with orthopedic trauma, which may provide more effective decision support for clinical practice. 