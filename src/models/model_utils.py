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

def get_model(name):
    if clf == 'lg':
        # logistic regression
        model = LogisticRegression(penalty='l2', random_state=0)
    elif clf == 'rf':
        # random forest
        model = RandomForestClassifier(bootstrap=True, criterion='gini',
                max_depth=7, max_features='sqrt', min_impurity_decrease=0.0,
                min_samples_leaf=20, min_samples_split=100,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                random_state=0, verbose=0)
    else:
        # Decision tree
        model = DecisionTreeClassifier(random_state=0, max_features='sqrt')  # max_feature: None->sqrt recall: 0.82->0.86


def train_model(score_func, clf, k):
	if score_func == 'mutual_info_classif':
        selector = SelectKBest(mutual_info_classif, k=k)  # f_regression k特征值数量
    else:
        selector = SelectKBest(f_classif, k=k)

        y_train, y_test = train_test_split()
        # print(X)
        feature_names = X_pd.columns

        if clf=='lg':
            model = LogisticRegression(penalty='l2', random_state=0)
        elif clf=='rf':
            # random forest
            model = RandomForestClassifier(bootstrap=True, criterion='gini',
                    max_depth=7, max_features='sqrt', min_impurity_decrease=0.0,
                    min_samples_leaf=20, min_samples_split=100,
                    min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                    random_state=0, verbose=0)
        else:
            # Decision tree
            model = DecisionTreeClassifier(random_state=0, max_features='sqrt')  # max_feature: None->sqrt recall: 0.82->0.86


        model.fit(x_train, y_train.ravel())
        y_hat = model.predict(x_test)
        y_score = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)

        return auc

def other():
    # optimal combination
if __name__ == '__main__':
    print('current dir is: ', os.getcwd())
    random.seed(5)

    selector = SelectKBest(f_classif, k=17) # the best for classifiers is 21(17/19/21)

    X, y, X_pd = load_data('../data/out1028.csv', selector=selector)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.6)
    y_train, y_test = y_train.astype(int), y_test.astype(int)
    feature_names = X_pd.columns
    print(feature_names)

    # # roc_lst = []
    # #
    # Logistic回归
    result = sm.Logit(y_train.ravel(), pd.DataFrame(x_train, columns=feature_names))
    result = result.fit()

    y_hat = result.predict(x_test)

    print('result.coef: ', result.summary())
    # lr = sm.Logit(y_train.ravel(), x_train)
    # lr = lr.fit()
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train.ravel())
    # print(shape(x_train), shape(y_train.ravel()))
    ## predict the test data
    y_hat = lr.predict(x_test).astype(int)
    print(x_test[1,:])
    y_score = lr.predict_proba(x_test)[:, 1]
    print(y_hat[1])
    print(y_score[1])
    # [0.32655368 - 0.49375     0.0387266   0.90384356  1.47738186  1.64691479
    #  - 0.03436049  0.30254665  2.0737764   0.15868139  2.34756613 - 0.40769132
    #  0.48397768  1.05671348 - 0.88486264 - 0.48177822 - 1.09588258]
    # 1
    # 0.9828157313454033
    # # print(y_hat)
    # # print(y_score)
    # # print('lr.coef: ', lr.summary())
    #
    # roc_lst.append(roc_auc_score(y_test, y_score))
    # print('Logistic\n', 'accuracy：', accuracy_score(y_test, y_hat),
    #       'precision:', precision_score(y_test, y_hat),
    #       'recall：', recall_score(y_test, y_hat),
    #       'roc_auc: ', roc_auc_score(y_test, y_score),
    #       '\n')


    # # random forest
    # RF = RandomForestClassifier(bootstrap=True, criterion='gini',
    #                             max_depth=8, max_features='sqrt', min_impurity_decrease=0.0,
    #                             min_samples_leaf=20, min_samples_split=100,
    #                             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
    #                             random_state=9, verbose=0)
    # RF.fit(x_train, y_train.ravel())
    # # print(RF.best_estimator_)
    # y_hat = RF.predict(x_test)
    # y_score = RF.predict_proba(x_test)[:, 1]
    # roc_lst.append(roc_auc_score(y_test, y_score))
    # print('随机森林\n', 'accuracy：', accuracy_score(y_test, y_hat),
    #       'precision:', precision_score(y_test, y_hat),
    #       'recall：', recall_score(y_test, y_hat),
    #       'roc_auc: ', roc_auc_score(y_test, y_score),
    #       '\n')
    # print(roc_auc_score(y_test, y_score))

    # # Decision tree
    # tree = DecisionTreeClassifier(random_state=0, max_features='sqrt')  # max_feature: None->sqrt recall: 0.82->0.86
    # tree.fit(x_train, y_train.ravel())
    # #     print(tree)
    # y_hat = tree.predict(x_test)
    # y_score = tree.predict_proba(x_test)[:, 1]
    # roc_lst.append(roc_auc_score(y_test, y_score))
    # print('决策树\n',
    #       'accuracy：', accuracy_score(y_test, y_score),
    #       'precision:', precision_score(y_test, y_score),
    #       'recall：', recall_score(y_test, y_score),
    #       'roc_auc: ', roc_auc_score(y_test, y_score),
    #       '\n')

    # visualize_tree(tree, feature_names)
    # grd = GradientBoostingClassifier(n_estimators=10)
    # grd.fit(x_train, y_train.ravel())
    # y_hat = grd.predict(x_test)
    # y_score = grd.predict_proba(x_test)[:, 1]
    # roc_lst.append(roc_auc_score(y_test, y_score))
    # print('GB\n', 'accuracy：', accuracy_score(y_test, y_hat),
    #       'precision:', precision_score(y_test, y_hat),
    #       'recall：', recall_score(y_test, y_hat),
    #       'roc_auc: ', roc_auc_score(y_test, y_score),
    #       '\n')
    #
    # plot_feature_importance(lr, X=X_pd, text='NAPS')
    # #
    # # auc_df = pd.DataFrame({'model': ['LR', 'RF', 'Tree'],
    # #                        'auc': roc_lst})
    # # plot_data(auc_df, type=4)
    # fpr, tpr, thresholds = roc_curve(y_test, y_score)
    # roc_auc = auc_func(fpr, tpr)
    # plot_auc(lr, fpr, tpr, roc_auc, text='NAPS')