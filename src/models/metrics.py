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
from train_model import random_forest_classifyer_default,random_forest_classifyer_tuned,gradient_boost_classifier_default,gradient_boost_classifier_tuned

def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../data/processed/full_ft_netflow_crw_{5000}_trw_{5}.csv'

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

    rw_size_set =[[1000,1],
             [2000,2],
             [5000,5],
             [10000,10],
             [15000,15],
             [20000,20]]

    metric_list_list=[]
    # for ft_set in feature_subsets:
    #     print(f' ////////////// {ft_set[1]}  ////////////// ')
    #     print('Starting default model')
    #
    #     model_info_default, time_info_default = random_forest_classifyer_default(ft_set[0], label,
    #                                                                              random_state=42)
    #     print('Starting tuned model')
    #     model_info_tuned, time_info_tuned = random_forest_classifyer_tuned(ft_set[0], label, random_state=42)
    #     print('Calculating default metrics')
    #     metric_list_list.append(get_model_general_results(model_info_default[0], model_info_default[1], model_info_default[3], f'RF - DF - {ft_set[1]} - RW {connection_rw_size}/{timw_rw_size_min}'))
    #     print('Calculating tuned metrics')
    #     metric_list_list.append(
    #         get_model_general_results(model_info_tuned[0], model_info_tuned[1], model_info_tuned[3],
    #                                   f'RF - Tuned - {ft_set[1]} - RW {connection_rw_size}/{timw_rw_size_min}'))
    #     file = open(os.getcwd() + 'metric_list.csv', 'w', newline='', encoding='utf-8')
    #     print('\n')
    #     print(metric_list_list)
    #     print('\n')
    # for ft_set in feature_subsets:
    #     print(f' ////////////// {ft_set[1]}  ////////////// ')
    #     print('Starting default model')
    #
    #     model_info_default, time_info_default = gradient_boost_classifier_default(ft_set[0], label,
    #                                                                              random_state=42)
    #     print('Starting tuned model')
    #     model_info_tuned, time_info_tuned = gradient_boost_classifier_tuned(ft_set[0], label, random_state=42)
    #     print('Calculating default metrics')
    #     metric_list_list.append(get_model_general_results(model_info_default[0], model_info_default[1], model_info_default[3], f'GB - DF - {ft_set[1]} - RW {connection_rw_size}/{timw_rw_size_min}'))
    #     print('Calculating tuned metrics')
    #     metric_list_list.append(
    #         get_model_general_results(model_info_tuned[0], model_info_tuned[1], model_info_tuned[3],
    #                                   f'GB - Tuned - {ft_set[1]} - RW {connection_rw_size}/{timw_rw_size_min}'))
    #     file = open(os.getcwd() + 'metric_list.csv', 'w', newline='', encoding='utf-8')
    #     print('\n')
    #     print(metric_list_list)
    #     print('\n')

    # for rw_size in rw_size_set:
    #     print(f' ////////////// {rw_size}  ////////////// ')
    #
    #     features_csv = f'../../data/processed/full_ft_netflow_crw_{rw_size[0]}_trw_{rw_size[1]}.csv'
    #     label = subset.get_target(features_csv)
    #     conn_ft, conn_ft_name = subset.get_conn_ft(features_csv)
    #     print('Trainning model')
    #     model_info_tuned, time_info_tuned = random_forest_classifyer_tuned(conn_ft, label, random_state=42)
    #     print('Generating metrics')
    #     metric_list_list.append(
    #                 get_model_general_results(model_info_tuned[0], model_info_tuned[1], model_info_tuned[3],
    #                                           f'RF - Tuned - CONN - RW {rw_size[0]}/{rw_size[1]}'))
    # for rw_size in rw_size_set:
    #     print(f' ////////////// {rw_size}  ////////////// ')
    #
    #     features_csv = f'../../data/processed/full_ft_netflow_crw_{rw_size[0]}_trw_{rw_size[1]}.csv'
    #     label = subset.get_target(features_csv)
    #     conn_ft, conn_ft_name = subset.get_conn_ft(features_csv)
    #     print('Trainning model')
    #     model_info_tuned, time_info_tuned = gradient_boost_classifier_tuned(conn_ft, label, random_state=42)
    #     print('Generating metrics')
    #     metric_list_list.append(
    #                 get_model_general_results(model_info_tuned[0], model_info_tuned[1], model_info_tuned[3],
    #                                           f'GB - Tuned - CONN - RW {rw_size[0]}/{rw_size[1]}'))
    #
    #
    # file = open(f'{os.getcwd()}/metric_list.csv', 'a', newline='', encoding='utf-8')
    # with file:
    #     writer=csv.writer(file)
    #     for rf_entry in metric_list_list:
    #         writer.writerow(rf_entry)
    print(f'GB Tuned || {datetime.now()}')
    model_info, time_info_tuned = gradient_boost_classifier_tuned(conn_fwd_ft, label, random_state=42)
    gb_tuned = model_info[2]
    print(f'GB Def || {datetime.now()}')
    model_info, time_info_tuned = gradient_boost_classifier_default(conn_fwd_ft, label, random_state=42)
    gb_def = model_info[2]
    print(f'RF Tuned || {datetime.now()}')
    model_info, time_info_tuned = random_forest_classifyer_tuned(conn_fwd_ft, label, random_state=42)
    rf_tuned = model_info[2]
    print(f'RF Def || {datetime.now()}')
    model_info, time_info_tuned = random_forest_classifyer_default(conn_fwd_ft, label, random_state=42)
    rf_def = model_info[2]


    time_gb_tuned=[]
    time_gb_def = []
    time_rf_tuned = []
    time_rf_def = []
    for x in range(0, 10):
        print(f'Iteration: {x} || {datetime.now()}')
        time_start=datetime.now()
        gb_tuned.predict(conn_fwd_ft)
        end_time=datetime.now()-time_start
        time_gb_tuned.append(end_time.total_seconds())

        time_start = datetime.now()
        gb_def.predict(conn_fwd_ft)
        end_time = datetime.now() - time_start
        time_gb_def.append(end_time.total_seconds())

        time_start = datetime.now()
        rf_tuned.predict(conn_fwd_ft)
        end_time = datetime.now() - time_start
        time_rf_tuned.append(end_time.total_seconds())

        time_start = datetime.now()
        rf_def.predict(conn_fwd_ft)
        end_time = datetime.now() - time_start
        time_rf_def.append(end_time.total_seconds())

    print(f' GB tuned: {sum(time_gb_tuned)/10} || {time_gb_tuned}')
    print(f' GB def: {sum(time_gb_def)/10} || {time_gb_def}')
    print(f' RF tuned: {sum(time_rf_tuned)/10} || {time_rf_tuned}')
    print(f' RF def: {sum(time_rf_def)/10} || {time_rf_def}')


    gb_tune_pred= gb_tuned.predict(conn_fwd_ft)
    gb_def_pred = gb_def.predict(conn_fwd_ft)
    rf_tune_pred = rf_tuned.predict(conn_fwd_ft)
    rf_def_pred = rf_def.predict(conn_fwd_ft)



    print('GB Tuned Results')
    model_general_results(gb_tune_pred, label)
    print('GB Tuned Results CV')
    get_model_cross_val_results(gb_tuned, conn_fwd_ft, label)
    print('GB Def Results')
    model_general_results(gb_def_pred, label)
    print('RF Tuned Results')
    model_general_results(rf_tune_pred, label)
    print('RF Tuned Results CV')
    get_model_cross_val_results(rf_tuned, conn_fwd_ft, label)
    print('RF Def Results')
    model_general_results(rf_def_pred, label)
    print('RF Def Results CV')
    get_model_cross_val_results(rf_def, conn_fwd_ft, label)



    import src.visualization as vis


    # vis.plot_cf_matrix(gb_tuned, conn_ft, label, 'GB Tuned')
    # vis.plot_cf_matrix(gb_def, conn_ft, label, 'GB def')
    # vis.plot_cf_matrix(rf_tuned, conn_ft, label, 'RF Tuned')
    # vis.plot_cf_matrix(rf_def, conn_ft, label, 'RF Def')










def model_general_results(predictions, y_test):
    print("=== Confusion Matrix ===")
    cf_matrix = confusion_matrix(y_test, predictions)
    print(cf_matrix)
    print('\n')
    print("=== Classification Report ===")
    class_report = classification_report(y_test, predictions)
    print(class_report)
    print('\n')
    print("=== ROC AUC Scores ===")
    print(roc_auc_score(y_test, predictions))
    print('\n')

def model_cross_val_results(model, ft_data, label):
    """
    Code adapted from https://github.com/StKyr/multiscorer/
    """
    scorer = MultiScorer({  # Create a MultiScorer instance
        'Accuracy': (accuracy_score, {}),
        'Precision': (precision_score, {'average': None}),
        'Recall': (recall_score, {'average': None}),
        'F-1': (f1_score, {'average': None}),
        'Area Under ROC Curve': (roc_auc_score, {})
    })

    cross_val_score(model, ft_data, label, scoring=scorer, cv=10)
    results = scorer.get_results()
    for metric in results.keys():  # Iterate and use the results
        if metric in ["Precision", 'Recall', 'F-1']:
            metric_vpn = []
            metric_non_vpn = []
            for sub_array in results[metric]:
                metric_non_vpn.append(sub_array[0])
                metric_vpn.append(sub_array[1])
            print(f'{metric}-Non VPN: {average(metric_non_vpn)}')
            print(f'{metric}-VPN: {average(metric_vpn)}')


        else:
            print(f'{metric}: {average(results[metric])}')

def get_model_general_results(predictions, y_test, prediction_probabilities, ft_set):
    # f1, f1_vpn = f1_score(y_test, predictions, average=None)
    prec, prec_vpn = precision_score(y_test, predictions, average=None)
    recc, recc_vpn = recall_score(y_test, predictions, average=None)
    auc = roc_auc_score(y_test, prediction_probabilities[:, 1])
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    acc = accuracy_score(y_test, predictions)
    model_metrics_list = [ft_set, acc, prec, prec_vpn, recc, recc_vpn, auc, tn, fp, fn, tp]

    return model_metrics_list

def get_model_cross_val_results(model, ft_data, label):
    """
        Code adapted from https://github.com/StKyr/multiscorer/
        """
    scorer = MultiScorer({  # Create a MultiScorer instance
        'precision': (precision_score, {'average': None}),
        'recall': (recall_score, {'average': None}),
        'f1': (f1_score, {'average': None}),
        'accuracy': (accuracy_score, {}),
        'auc': (roc_auc_score, {})
    })

    model_cross_val_metrics_list = []
    cross_val_score(model, ft_data, label, scoring=scorer, cv=10)
    results = scorer.get_results()
    for metric in results.keys():  # Iterate and use the results
        if metric in ["precision", 'recall', 'f1']:
            metric_vpn = []
            metric_non_vpn = []
            for sub_array in results[metric]:
                metric_non_vpn.append(sub_array[0])
                metric_vpn.append(sub_array[1])

            model_cross_val_metrics_list.append(average(metric_non_vpn))
            model_cross_val_metrics_list.append(average(metric_vpn))

        else:

            model_cross_val_metrics_list.append(average(results[metric]))

    model_cross_val_metrics_string = ""
    for ele in model_cross_val_metrics_list:
        model_cross_val_metrics_string += f', {str(ele)}'
    model_cross_val_metrics_string = model_cross_val_metrics_string[2:]

    return model_cross_val_metrics_string, model_cross_val_metrics_list


if __name__ == '__main__':
    main()
