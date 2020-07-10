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
from numpy import average
from tqdm import tqdm
import os
import csv
from train_model import get_model_cross_val_results, get_model_general_results, random_forest_classifyer_default,random_forest_classifyer_tuned

def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../data/processed/full_ft_netflow_crw_{10000}_trw_{10}.csv'

    label = subset.get_target(features_csv)

    baseline_ft, basene_ft_name = subset.get_baseline_ft(features_csv)
    time_ft, time_ft_name = subset.get_time_ft(features_csv)
    conn_ft, conn_ft_name = subset.get_conn_ft(features_csv)
    comb_ft, comb_ft_name = subset.get_comb_ft(features_csv)

    time_fwd_ft, time_fwd_ft_name = subset.get_time_fwd_ft(features_csv)
    time_bwd_ft, time_bwd_ft_name = subset.get_time_bwd_ft(features_csv)

    conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    conn_bwd_ft, conn_bwd_ft_name = subset.get_conn_bwd_ft(features_csv)

    comb_fwd_ft, comb_fwd_ft_name = subset.get_comb_fwd_ft(features_csv)
    comb_bwd_ft, comb_bwd_ft_name = subset.get_comb_bwd_ft(features_csv)





    feature_subsets= [ [baseline_ft, 'baseline'],
                       [time_ft,'time'],
                       [conn_ft,'conn'],
                       [comb_ft,'comb'],
                       [time_fwd_ft,'time_fwd'],
                       [time_bwd_ft,'time_bwd'],
                       [conn_fwd_ft,'conn_fwd'],
                       [conn_bwd_ft,'conn_bwd'],
                       [comb_fwd_ft,'comb_fwd'],
                       [comb_bwd_ft,'comb_bwd']]

    metric_list_list=[]
    for ft_set in feature_subsets:
        print(f' ////////////// {ft_set[1]}  ////////////// ')
        print('Starting default model')

        model_info_default, time_info_default = random_forest_classifyer_default(ft_set[0], label,
                                                                                 random_state=42)
        print('Starting tuned model')
        model_info_tuned, time_info_tuned = random_forest_classifyer_tuned(ft_set[0], label, random_state=42)
        print('Calculating default metrics')
        metric_list_list.append(get_model_general_results(model_info_default[0], model_info_default[1], model_info_default[3], f'RF - DF - {ft_set[1]} - RW {connection_rw_size}/{timw_rw_size_min}'))
        print('Calculating tuned metrics')
        metric_list_list.append(
            get_model_general_results(model_info_tuned[0], model_info_tuned[1], model_info_tuned[3],
                                      f'RF - Tuned - {ft_set[1]} - RW {connection_rw_size}/{timw_rw_size_min}'))
        file = open(os.getcwd() + 'metric_list.csv', 'w', newline='', encoding='utf-8')
        print('\n')
        print(metric_list_list)
        print('\n')

    with file:
        writer=csv.writer(file)
        for rf_entry in metric_list_list:
            writer.writerow(rf_entry)



    # for ft_sub in feature_subsets:
    #
    #     name=ft_sub[1]
    #     print(f'/=========== {name} ===========\\ ')
    #     rf_filename=f'/random_forest_metrics/default_param/rf_dp_{name}_2K-2.csv'
    #     gb_filename = f'/gradient_boost_metrics/default_param/gb_dp_{name}_2K-2.csv'
    #
    #     if os.path.exists(tl+rf_filename):
    #         print(f' {name} rf is already done')
    #     else:
    #         rf_feature_metrics_lits_cv=[]
    #         t = tqdm(total=100)
    #         for x in range(0, 100):
    #             model_data, time_data = random_forest_classifyer(ft_sub[0], label)
    #             rf_feature_metrics_lits_cv.append(get_model_general_results(model_data[0], model_data[1])[1])
    #             t.update(1)
    #         t.close()
    #
    #         file = open(tl+rf_filename, 'w', newline='', encoding='utf-8')
    #         with file:
    #             writer=csv.writer(file)
    #             writer.writerow(csv_header)
    #             for rf_entry in rf_feature_metrics_lits_cv:
    #                 writer.writerow(rf_entry)
    #     if os.path.exists(tl+gb_filename):
    #         print(f' {name} gb is already done')
    #
    #     else:
    #         gb_feature_metrics_lits_cv = []
    #         t = tqdm(total=100)
    #         for x in range(0, 100):
    #             model_data, time_data = gradient_boost_classifier(ft_sub[0], label)
    #             gb_feature_metrics_lits_cv.append(get_model_general_results(model_data[0], model_data[1])[1])
    #             t.update(1)
    #         t.close()
    #
    #         file = open(tl+gb_filename, 'w', newline='', encoding='utf-8')
    #         with file:
    #             writer = csv.writer(file)
    #             writer.writerow(csv_header)
    #             for gb_entry in gb_feature_metrics_lits_cv:
    #                 writer.writerow(gb_entry)

    # for ft_sub in feature_subsets:
    #     name = ft_sub[1]
    #     print(f'/=========== {name}-CV ===========\\ ')
    #
    #     rf_filename = f'/random_forest_metrics/default_param_cross_val/rf_dp_cv_{name}.csv'
    #     gb_filename = f'/gradient_boost_metrics/default_param_cross_val/gb_dp_cv_{name}.csv'
    #
    #     if os.path.exists(tl+rf_filename):
    #         print(f' {name} rf is already done')
    #     else:
    #         rf_feature_metrics_lits_cv = []
    #         t = tqdm(total=10)
    #         for x in range(0, 10):
    #             model_data, time_data = random_forest_classifyer(ft_sub[0], label)
    #             rf_feature_metrics_lits_cv.append(get_model_cross_val_results(model_data[2], ft_sub[0], label)[1])
    #             t.update(1)
    #         t.close()
    #
    #         file = open(tl+rf_filename, 'w', newline='', encoding='utf-8')
    #         with file:
    #             writer = csv.writer(file)
    #             writer.writerow(cv_csv_header)
    #             for rf_entry in rf_feature_metrics_lits_cv:
    #                 writer.writerow(rf_entry)
    #     if os.path.exists(tl+gb_filename):
    #         print(f' {name} gb is already done')
    #     else:
    #         gb_feature_metrics_lits_cv = []
    #         t = tqdm(total=10)
    #         for x in range(0, 10):
    #             model_data, time_data = gradient_boost_classifier(ft_sub[0], label)
    #             gb_feature_metrics_lits_cv.append(get_model_cross_val_results(model_data[2], ft_sub[0],label)[1])
    #             t.update(1)
    #         t.close()
    #
    #         file = open(tl+gb_filename, 'w', newline='', encoding='utf-8')
    #         with file:
    #             writer = csv.writer(file)
    #             writer.writerow(cv_csv_header)
    #             for gb_entry in gb_feature_metrics_lits_cv:
    #                 writer.writerow(gb_entry)

if __name__ == '__main__':
    main()
