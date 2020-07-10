import csv
import os
import warnings

import numpy

import src.config as cfg
import src.feature_subsets as subset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, \
    recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from src.multiscorer import MultiScorer
from numpy import average, arange
from tqdm import tqdm
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}.csv'

    label = subset.get_target(features_csv)

    baseline_ft, baseline_ft_name = subset.get_baseline_ft(features_csv)
    time_ft, time_ft_name = subset.get_time_ft(features_csv)
    conn_ft, conn_ft_name = subset.get_conn_ft(features_csv)
    comb_ft, comb_ft_name = subset.get_comb_ft(features_csv)

    time_fwd_ft, time_fwd_ft_name = subset.get_time_fwd_ft(features_csv)
    time_bwd_ft, time_bwd_ft_name = subset.get_time_bwd_ft(features_csv)

    conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    conn_bwd_ft, conn_bwd_ft_name = subset.get_conn_bwd_ft(features_csv)

    comb_fwd_ft, comb_fwd_ft_name = subset.get_comb_fwd_ft(features_csv)
    comb_bwd_ft, comb_bwd_ft_name = subset.get_comb_bwd_ft(features_csv)


    x_train, x_test, y_train, y_test = train_test_split(
        conn_fwd_ft, label, test_size=0.3, random_state=42)

    # param_test1 = {'max_features' : [None, 'sqrt', 'log2'],
    #                }

    #  # 'learning_rate': arange(0.05, 0.8, 0.05)
    # # 'n_estimators':[100,150, 200, 250, 300, 500,750,1000],
    # # 'max_depth': range(5, 16, 1)
    # # 'min_samples_split':[2,4,6,8,10,20,40,60,100],
    # # 'min_samples_leaf':[1,2,3,5,7,9]
    # # 'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]
    # # 'max_features' : [None, 'sqrt', 'log2']
    # gsearch1 = GridSearchCV(
    #     #     estimator=GradientBoostingClassifier(random_state=42, learning_rate=0.6, max_depth=12, min_samples_split=2, min_samples_leaf=1, n_estimators=500, subsample=1),
    #     #     param_grid=param_test1, scoring='accuracy', n_jobs=-1, cv=5)
    #     # gsearch1.fit(x_train, y_train)
    #     # print(gsearch1.cv_results_)
    #     # print(gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_)
    param_test2 = { 'min_samples_split': [2, 4, 6, 8, 10, 20, 40, 60]

                   }
    # 'n_estimators': [100,150, 200, 250, 300, 500,750,1000],
    #                    'max_depth': range(5, 105, 10),
    #                    'min_samples_leaf': [1, 2, 3, 5, 7, 9],
    #                    'min_samples_split': [2, 4, 6, 8, 10, 20, 40, 60],
    # 'bootstrap': [True, False]
    # 'max_features' : [None, 'sqrt', 'log2'],

    gsearch2 = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=300, max_depth=25, max_features=None),
        param_grid=param_test2, scoring='accuracy', n_jobs=-1, cv=5)
    gsearch2.fit(x_train, y_train)
    print(gsearch2.cv_results_)
    print(gsearch2.best_estimator_, gsearch2.best_params_, gsearch2.best_score_)






if __name__ == '__main__':
    main()