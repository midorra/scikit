# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt

if __name__ == '__main__':
    
    train = pd.read_csv('./data/gbdt/train_modified.csv')
    target='Disbursed' # Disbursed的值就是二元分类的输出，也就是Label
    IDcol = 'ID' # Instance编号或行号
    print train['Disbursed'].value_counts()

    ### build training data and target
    x_columns = [x for x in train.columns if x not in [target, IDcol]]
    x_train = train[x_columns]
    y_train = train['Disbursed']

    ### train model
    gbm0 = GradientBoostingClassifier(random_state=10, verbose = 2) # 首先全部使用默认参数
    gbm0.fit(x_train, y_train)

    ### check model
    y_pred = gbm0.predict(x_train)
    y_predprob = gbm0.predict_proba(x_train)[:,1]
    print "Accuracy : %.4g" % metrics.accuracy_score(y_train.values, y_pred)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob)

    ### param optimization : n_estimators
    param_test1 = {'n_estimators': range(20, 81, 10)}
    gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier( learning_rate = 0.1, \
                                                                    min_samples_split = 300, \
                                                                    min_samples_leaf = 20, \
                                                                    max_depth = 8, \
                                                                    max_features = 'sqrt', \
                                                                    subsample = 0.8, \
                                                                    random_state = 10, \
                                                                    verbose = 1), \
                            param_grid = param_test1, \
                            scoring='roc_auc', \
                            iid=False, \
                            cv=5)
    gsearch1.fit(x_train, y_train)
    print '\n### param optimization : n_estimatots'
    print '\ngrid_scores :'
    print gsearch1.grid_scores_
    print '\nbest_params :'
    print gsearch1.best_params_
    print '\nbest_score  :'
    print gsearch1.best_score_

    ### param optimization : max_depth
    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
    gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier( learning_rate = 0.1, \
                                                                    n_estimators = 60, \
                                                                    min_samples_leaf = 20, \
                                                                    max_features = 'sqrt', \
                                                                    subsample = 0.8, \
                                                                    random_state = 10, \
                                                                    verbose = 1), \
                            param_grid = param_test2, \
                            scoring='roc_auc', \
                            iid=False, \
                            cv=5)
    gsearch2.fit(x_train, y_train)
    print '\n### param optimization : max_depth'
    print '\ngrid_scores :'
    print gsearch2.grid_scores_
    print '\nbest_params :'
    print gsearch2.best_params_
    print '\nbest_score  :'
    print gsearch2.best_score_

