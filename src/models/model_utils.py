from scipy.optimize import minimize
from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
from pipelinehelper import PipelineHelper
from src.tools.plot_utils import *
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2
import statsmodels.api as sm

def train(X_train, X_test, y_train, y_test, model='LogisticRegression', show_params_refer=0):

    if model == 'LogisticRegression':
        pipe = Pipeline([
        ('selecter', SelectKBest(mutual_info_classif)),
        ('classifier', LogisticRegression(penalty='l2', random_state=0))])

        param_grid = {'selecter__k': range(4, 55) #range(4, 21)
        }
    elif model == 'DecisionTree':
        pipe = Pipeline([
        ('selecter', SelectKBest(mutual_info_classif)),
        ('classifier', DecisionTreeClassifier(max_depth=5))]
        )

        param_grid = {'selecter__k':  range(19)
        }
    else:
        pipe = Pipeline([
        ('selecter', SelectKBest(mutual_info_classif)),
        ('classifier', RandomForestClassifier(max_features='sqrt'))]
        )

        param_grid = {'selecter__k':  range(4, 55),
            'classifier__n_estimators': range(50, 300, 25),
            'classifier__max_depth': range(10, 50, 5),
            'classifier__min_samples_leaf': range(14, 40, 2)
    }
    
    score = 'roc_auc'
    print("# Tuning hyper-parameters for %s" % model)
    print()
    
    # save the outputs of gridsearch
    sys.stdout = open('results.csv', 'w')
    clf = GridSearchCV(pipe, n_jobs=1,
                       param_grid=param_grid,
                       verbose=2,
                       scoring='%s' % score,
                       cv=3)
    
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    clf.fit(X, y)
    sys.stdout.close()
    # joblib.dump(grid.best_params_, 'best_tfidf.pkl', compress = 1) 
    # from sklearn.externals import joblib
    # joblib.dump(grid.best_estimator_, 'rick.pkl')

    # print("Best parameters set found on development set for %s:" % model)
    # print()
    # print(clf.best_params_)
    # print()
    # print("Grid scores on development set for %s:" % model)
    # print()
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()
    
    # print("Best parameters set found on development set for %s:" % model)
    # print()
    # print("best_score:", clf.best_score_, "best_params:", clf.best_params_)
    plot_auc(y_test=clf.best_estimator_.predict(X), y_score=y, model_name=model)


def test_data(X_train, X_test, y_train, y_test, k=0, model_name=None, group_type=None):
    print("#------- feature importances for %s and %s group-------#" % (model_name, group_type))

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    X = pd.DataFrame(X, columns=X_train.columns)
    # y = pd.DataFrame(y, columns=y_train.dtype.names)

    # Create and fit selector
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)

    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices=True)
    X_new = X.iloc[:,cols]

    if model_name == 'LogisticRegression':
        model = LogisticRegression(penalty='l2', random_state=0)

        # # Hosmer-Lemeshow檢驗
        # result = sm.Logit(y, X_new)
        # result = result.fit()
        # y_prob = result.predict(X_new)
        
        # print('result.coef: \n', result.summary())

        # final = calculate_hltest(y_prob, y)
        # print('hltest results: \n', final)

    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier(max_depth=5)
    else:
        if group_type == 'control':
            model = RandomForestClassifier(max_depth=20,max_features='sqrt',min_samples_leaf=14,n_estimators=100)
        else:
            model = RandomForestClassifier(max_depth=40, max_features='sqrt', min_samples_leaf=14, n_estimators=50)
    
    model.fit(X_new, y)
    plot_feature_importance(model, X_new, model_name='%s-%s'% (model_name, group_type))


def calculate_hltest_old(df):
    """Test a model before putting it into production and verify that the model we have assumed is 
    correctly specified with the right assumptions."""
    
    df = df.sort_values('sapsii')
    df['score_decile'] = pd.qcut(df['sapsii'], 10)
    obsevents_pos = df['expire_flag'].groupby(df.score_decile).sum()
    obsevents_neg = df['sapsii'].groupby(df.score_decile).count() - obsevents_pos
    expevents_pos = df['sapsii'].groupby(df.score_decile).sum()
    expevents_neg = df['sapsii'].groupby(df.score_decile).count() - expevents_pos
    hl = (((obsevents_pos - expevents_pos)**2/expevents_pos) + 
        ((obsevents_neg - expevents_neg)**2/expevents_neg)).sum()

    return hl

def calculate_hltest(y_prob, y):
    # Hoslem-Lemeshow Test
    print('Hoslem-Lemeshow Test')
    y_prob = pd.DataFrame(y_prob, columns=['pred'])
    y = pd.DataFrame(y, columns=['test'])
    y_prob1 = pd.concat([y_prob, y], axis =1)
    print(y_prob1.head())
    y_prob1['decile'] = pd.qcut(y_prob1['pred'], 10)
    # the number of positive value 1 and negative value 0 for each cut
    obsevents_pos = y_prob1['test'].groupby(y_prob1.decile).sum()
    obsevents_neg = y_prob1['pred'].groupby(y_prob1.decile).count() - obsevents_pos
    expevents_pos = y_prob1['pred'].groupby(y_prob1.decile).sum()
    expevents_neg = y_prob1['pred'].groupby(y_prob1.decile).count() - expevents_pos
    hl = ((obsevents_neg - expevents_neg)**2/expevents_neg).sum()+((obsevents_pos - expevents_pos)**2/expevents_pos).sum()
    print('chi-square: {:.2f}'.format(hl))
    ## df = group-2
    pvalue=1-chi2.cdf(hl,8)
    print('p-value: {:.2f}'.format(pvalue))
    obsevents_pos = pd.DataFrame(obsevents_pos)
    obsevents_neg = pd.DataFrame(obsevents_neg)
    expevents_pos = pd.DataFrame(expevents_pos)
    expevents_neg = pd.DataFrame(expevents_neg)
    final = pd.concat([obsevents_pos, obsevents_neg, expevents_pos, expevents_neg], axis =1)
    final.columns=['obs_pos','obs_neg','exp_pos', 'exp_neg']
    
    return final


    

