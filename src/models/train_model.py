import csv
import os
import warnings

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

def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}_2.csv'

    label = subset.get_target(features_csv)

    baseline_ft, baselone_ft_name = subset.get_baseline_ft(features_csv)
    time_ft, time_ft_name = subset.get_time_ft(features_csv)
    conn_ft, conn_ft_name = subset.get_conn_ft(features_csv)
    comb_ft, comb_ft_name = subset.get_comb_ft(features_csv)

    time_fwd_ft, time_fwd_ft_name = subset.get_time_fwd_ft(features_csv)
    time_bwd_ft, time_bwd_ft_name = subset.get_time_bwd_ft(features_csv)

    conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    conn_bwd_ft, conn_bwd_ft_name = subset.get_conn_bwd_ft(features_csv)

    comb_fwd_ft, comb_fwd_ft_name = subset.get_comb_fwd_ft(features_csv)
    comb_bwd_ft, comb_bwd_ft_name = subset.get_comb_bwd_ft(features_csv)

    # model_data, time_data = random_forest_classifyer(time_ft, label, random_state=42)
    # start_time=datetime.now()
    # model_general_results(model_data[0], model_data[1])
    # print(datetime.now()-start_time)
    # print('////////////////////////////////')
    # start_time = datetime.now()
    # print(get_model_general_results(model_data[0], model_data[1])[0])
    # print(datetime.now() - start_time)
    # print('////////////////////////////////')
    # start_time = datetime.now()
    # model_cross_val_results(model_data[2],time_ft, label)
    # print(datetime.now() - start_time)



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

    csv_header=['p','p_VPN', 'r', 'r_VPN', 'f1', 'f1_VPN', 'acc', 'auc', 'tp', 'tn', 'fp', 'fn']
    tl = 'C:/Users/boerg/PycharmProjects/vpn_network_traffic_analyser/src/models'
    cv_csv_header=['p_cv','p_VPN_cv', 'r_cv', 'r_VPN_cv', 'f1_cv', 'f1_VPN_cv', 'acc_cv', 'auc_cv']


    for ft_sub in feature_subsets:

        name=ft_sub[1]
        print(f'/=========== {name} ===========\\ ')
        rf_filename=f'/random_forest_metrics/default_param/rf_dp_{name}.csv'
        gb_filename = f'/gradient_boost_metrics/default_param/gb_dp_{name}.csv'

        if os.path.exists(tl+rf_filename):
            print(f' {name} rf is already done')
        else:
            rf_feature_metrics_lits_cv=[]
            t = tqdm(total=100)
            for x in range(0, 100):
                model_data, time_data = random_forest_classifyer(ft_sub[0], label)
                rf_feature_metrics_lits_cv.append(get_model_general_results(model_data[0], model_data[1])[1])
                t.update(1)
            t.close()

            file = open(tl+rf_filename, 'w', newline='', encoding='utf-8')
            with file:
                writer=csv.writer(file)
                writer.writerow(csv_header)
                for rf_entry in rf_feature_metrics_lits_cv:
                    writer.writerow(rf_entry)
        if os.path.exists(tl+gb_filename):
            print(f' {name} gb is already done')

        else:
            gb_feature_metrics_lits_cv = []
            t = tqdm(total=100)
            for x in range(0, 100):
                model_data, time_data = gradient_boost_classifier(ft_sub[0], label)
                gb_feature_metrics_lits_cv.append(get_model_general_results(model_data[0], model_data[1])[1])
                t.update(1)
            t.close()

            file = open(tl+gb_filename, 'w', newline='', encoding='utf-8')
            with file:
                writer = csv.writer(file)
                writer.writerow(csv_header)
                for gb_entry in gb_feature_metrics_lits_cv:
                    writer.writerow(gb_entry)

    for ft_sub in feature_subsets:
        name = ft_sub[1]
        print(f'/=========== {name}-CV ===========\\ ')

        rf_filename = f'/random_forest_metrics/default_param_cross_val/rf_dp_cv_{name}.csv'
        gb_filename = f'/gradient_boost_metrics/default_param_cross_val/gb_dp_cv_{name}.csv'

        if os.path.exists(tl+rf_filename):
            print(f' {name} gb is already done')
        else:
            rf_feature_metrics_lits_cv = []
            t = tqdm(total=10)
            for x in range(0, 10):
                model_data, time_data = random_forest_classifyer(ft_sub[0], label)
                rf_feature_metrics_lits_cv.append(get_model_cross_val_results(model_data[2], ft_sub[0], label)[1])
                t.update(1)
            t.close()

            file = open(tl+rf_filename, 'w', newline='', encoding='utf-8')
            with file:
                writer = csv.writer(file)
                writer.writerow(cv_csv_header)
                for rf_entry in rf_feature_metrics_lits_cv:
                    writer.writerow(rf_entry)
        if os.path.exists(tl+gb_filename):
            pass
        else:
            gb_feature_metrics_lits_cv = []
            t = tqdm(total=10)
            for x in range(0, 10):
                model_data, time_data = gradient_boost_classifier(ft_sub[0], label)
                gb_feature_metrics_lits_cv.append(get_model_cross_val_results(model_data[2], ft_sub[0],label)[1])
                t.update(1)
            t.close()

            file = open(tl+gb_filename, 'w', newline='', encoding='utf-8')
            with file:
                writer = csv.writer(file)
                writer.writerow(cv_csv_header)
                for gb_entry in gb_feature_metrics_lits_cv:
                    writer.writerow(gb_entry)
















def random_forest_classifyer(feature_data, label, test_size=0.3, random_state=None):  # todo: add hyperparameter
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    random_forest_model = RandomForestClassifier(n_jobs=-1, random_state=random_state, n_estimators=100)
    start_time = datetime.now()
    random_forest_model.fit(x_train, y_train)
    train_time = datetime.now() - start_time

    start_time = datetime.now()
    predictions = random_forest_model.predict(x_test)
    predict_time = datetime.now() - start_time
    return [predictions, y_test, random_forest_model], [predict_time, train_time]


def gradient_boost_classifier(feature_data, label, test_size=0.3, random_state=None):  # todo: add hyperparameter
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    gradient_boost_model = GradientBoostingClassifier(random_state=random_state)
    start_time = datetime.now()
    gradient_boost_model.fit(x_train, y_train)
    train_time = datetime.now() - start_time

    start_time = datetime.now()
    predictions = gradient_boost_model.predict(x_test)
    predict_time = datetime.now() - start_time

    return [predictions, y_test, gradient_boost_model], [predict_time, train_time]

def svm_classifier(feature_data, label, test_size=0.3, random_state=None):  # todo: add hyperparameter
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)


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



def get_model_general_results(predictions, y_test):
    f1, f1_vpn = f1_score(y_test, predictions, average=None)
    prec, prec_vpn = precision_score(y_test, predictions, average=None)
    recc, recc_vpn = recall_score(y_test, predictions, average=None)
    auc = roc_auc_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    acc = accuracy_score(y_test, predictions)
    model_metrics_list = [prec, prec_vpn, recc, recc_vpn, f1, f1_vpn, acc, auc, tp, fn, fp, tp]
    model_metrics_string = ""
    for ele in model_metrics_list:
        model_metrics_string += f', {str(ele)}'
    model_metrics_string = model_metrics_string[2:]
    return model_metrics_string, model_metrics_list


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


# todo: return data from cv evaluation scores (mean)


if __name__ == '__main__':
    main()
