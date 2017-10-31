# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt

if __name__ == '__main__':
    
    train = pd.read_csv('./data/gbdt/train_modified.csv')
    target='Disbursed' # Disbursed的值就是二元分类的输出
    IDcol = 'ID'
    print train['Disbursed'].value_counts()

    x_columns = [x for x in train.columns if x not in [target, IDcol]]
    x_train = train[x_columns]
    y_train = train['Disbursed']

    gbm0 = GradientBoostingClassifier(  n_estimators = 120, \
                                        learning_rate = 0.1, \
                                        random_state=10, \
                                        verbose = 2)
    gbm0.fit(x_train, y_train)
    y_pred = gbm0.predict(x_train)
    y_predprob = gbm0.predict_proba(x_train)[:,1]
    print "Accuracy : %.4g" % metrics.accuracy_score(y_train.values, y_pred)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob)

    param_test1 = {'n_estimators':range(20,81,10)}
    gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier( learning_rate=0.1, \
                                                                    min_samples_split=300, \
                                                                    min_samples_leaf=20, \
                                                                    max_depth=8, \
                                                                    max_features='sqrt', \
                                                                    subsample=0.8, \
                                                                    random_state=10), \
                            param_grid = param_test1, \
                            scoring='roc_auc', \
                            iid=False, \
                            cv=5)
    gsearch1.fit(x_train, y_train)
    print 'grid_scores :'
    print gsearch1.grid_scores_
    print 'best_params :'
    print gsearch1.best_params_
    print 'best_score  :'
    print gsearch1.best_score_
