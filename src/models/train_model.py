import csv
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import src.config as cfg
import src.feature_subsets as subset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, \
    recall_score, accuracy_score, plot_confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from src.multiscorer import MultiScorer
from numpy import average
from tqdm import tqdm
from sklearn.svm import SVC
import scikitplot as skplt
import matplotlib.pyplot as plt

def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}.csv'

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

    # model_data, time_data = svm_classifier(baseline_ft, label)
    # print(time_data)
    # print('////////////////////////////////')
    #
    #
    # start_time=datetime.now()
    # model_general_results(model_data[0], model_data[1])
    # print(datetime.now()-start_time)
    #
    # print('////////////////////////////////')
    # start_time = datetime.now()
    # print(get_model_general_results(model_data[0], model_data[1])[0])
    # print(datetime.now() - start_time)
    # print('////////////////////////////////')
    # # start_time = datetime.now()
    # # model_cross_val_results(model_data[2],time_ft, label)
    # # print(datetime.now() - start_time)




    model_info, time_info =random_forest_classifyer_tuned(baseline_ft, label, random_state=42)
    model_general_results(model_info[0], model_info[1])
    print(time_info)
    # adapted from https://towardsdatascience.com/visualizing-decision-trees-with-python-scikit-learn-graphviz-matplotlib-1c50b4aa68dc
    # fn = conn_ft_name
    # cn = ['Non-VPN','VPN']
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=800)
    # from sklearn import tree
    # from sklearn.tree import export_graphviz
    # tree.plot_tree(model_info[2].estimators_[0],
    #                feature_names=fn,
    #                class_names=cn,
    #                filled=True)
    # fig.savefig('rf_tree_depth_unl.png')
    # plt.show()

    # model_info, time_info =gradient_boost_classifier(conn_fwd_ft, label, random_state=42)
    #
    # model_general_results(model_info[0], model_info[1])
    # print(time_info)

    # print(f' /////////////// 1000 - 1  ///////////////')
    #
    # features_csv = f'../../data/processed/full_ft_netflow_crw_{1000}_trw_{1}.csv'
    #
    # label = subset.get_target(features_csv)
    # conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    # model_info, time_info = gradient_boost_classifier(conn_fwd_ft, label, random_state=42)
    #
    # model_general_results(model_info[0], model_info[1])
    # print(time_info)
    #
    # print(f' /////////////////////////////////////////')
    # print(f' /////////////// 2000 - 2  ///////////////')
    #
    # features_csv = f'../../data/processed/full_ft_netflow_crw_{2000}_trw_{2}.csv'
    #
    # label = subset.get_target(features_csv)
    # conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    # model_info, time_info = gradient_boost_classifier(conn_fwd_ft, label, random_state=42)
    #
    # model_general_results(model_info[0], model_info[1])
    # print(time_info)
    print(f' /////////////////////////////////////////')
    print(f' /////////////// 5000 - 5  ///////////////')

    features_csv = f'../../data/processed/full_ft_netflow_crw_{5000}_trw_{5}.csv'

    label = subset.get_target(features_csv)
    conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    model_info, time_info = gradient_boost_classifier_tuned(conn_fwd_ft, label, random_state=42)



    model_general_results(model_info[0], model_info[1])
    print(time_info)
    # print(f' /////////////////////////////////////////')
    # print(f' /////////////// 10000 - 10  ///////////////')
    #
    # features_csv = f'../../data/processed/full_ft_netflow_crw_{10000}_trw_{10}.csv'
    #
    # label = subset.get_target(features_csv)
    # conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    # model_info, time_info = gradient_boost_classifier(conn_fwd_ft, label, random_state=42)
    #
    # model_general_results(model_info[0], model_info[1])
    # print(time_info)
    #
    # print(f' /////////////////////////////////////////')
    # print(f' /////////////// 15000 - 15  ///////////////')
    #
    # features_csv = f'../../data/processed/full_ft_netflow_crw_{15000}_trw_{15}.csv'
    #
    # label = subset.get_target(features_csv)
    # conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    # model_info, time_info = gradient_boost_classifier(conn_fwd_ft, label, random_state=42)
    #
    # model_general_results(model_info[0], model_info[1])
    # print(time_info)
    #
    # print(f' /////////////////////////////////////////')
    # print(f' /////////////// 20000 - 20  ///////////////')
    #
    # features_csv = f'../../data/processed/full_ft_netflow_crw_{20000}_trw_{20}.csv'
    #
    # label = subset.get_target(features_csv)
    # conn_fwd_ft, conn_fwd_ft_name = subset.get_conn_fwd_ft(features_csv)
    # model_info, time_info = gradient_boost_classifier(conn_fwd_ft, label, random_state=42)
    #
    # model_general_results(model_info[0], model_info[1])
    # print(time_info)



















def random_forest_classifyer_tuned(feature_data, label, test_size=0.3, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    random_forest_model = RandomForestClassifier(n_jobs=-1, random_state=random_state, n_estimators=300, max_depth=25,  max_features=None)
    start_time = datetime.now()
    random_forest_model.fit(x_train, y_train)
    train_time = datetime.now() - start_time

    start_time = datetime.now()
    predictions = random_forest_model.predict(x_test)
    prediction_probabilities=random_forest_model.predict_proba(x_test)
    predict_time = datetime.now() - start_time





    return [predictions, y_test, random_forest_model, prediction_probabilities, x_test], [predict_time, train_time]


def random_forest_classifyer_default(feature_data, label, test_size=0.3, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    random_forest_model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    start_time = datetime.now()
    random_forest_model.fit(x_train, y_train)
    train_time = datetime.now() - start_time

    start_time = datetime.now()
    predictions = random_forest_model.predict(x_test)
    prediction_probabilities = random_forest_model.predict_proba(x_test)
    predict_time = datetime.now() - start_time

    return [predictions, y_test, random_forest_model, prediction_probabilities, x_test], [predict_time, train_time]

def gradient_boost_classifier_tuned(feature_data, label, test_size=0.3, random_state=None):  # todo: add hyperparameter
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    gradient_boost_model = GradientBoostingClassifier(random_state=random_state, learning_rate=0.6, max_depth=12, min_samples_split=2, min_samples_leaf=1, n_estimators=500, subsample=1 )
    start_time = datetime.now()
    gradient_boost_model.fit(x_train, y_train)
    train_time = datetime.now() - start_time

    start_time = datetime.now()
    predictions = gradient_boost_model.predict(x_test)
    prediction_probabilities=gradient_boost_model.predict_proba(x_test)
    predict_time = datetime.now() - start_time



    return [predictions, y_test, gradient_boost_model, prediction_probabilities, x_test], [predict_time, train_time]


def gradient_boost_classifier_default(feature_data, label, test_size=0.3, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    gradient_boost_model = GradientBoostingClassifier(random_state=random_state)
    start_time = datetime.now()
    gradient_boost_model.fit(x_train, y_train)
    train_time = datetime.now() - start_time

    start_time = datetime.now()
    predictions = gradient_boost_model.predict(x_test)
    prediction_probabilities = gradient_boost_model.predict_proba(x_test)
    predict_time = datetime.now() - start_time

    return [predictions, y_test, gradient_boost_model, prediction_probabilities, x_test], [predict_time, train_time]

def svm_classifier(feature_data, label, test_size=0.3, random_state=None):  # todo: add hyperparameter
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    svm_model=SVC(random_state=random_state, kernel='sigmoid')
    start_time = datetime.now()
    svm_model.fit(x_train, y_train)
    train_time = datetime.now() - start_time

    start_time = datetime.now()
    predictions = svm_model.predict(x_test)
    predict_time = datetime.now() - start_time

    return [predictions, y_test, svm_model], [predict_time, train_time]


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
    auc = roc_auc_score(y_test, prediction_probabilities[:,1])
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    acc = accuracy_score(y_test, predictions)
    model_metrics_list = [ft_set,acc,prec, prec_vpn, recc, recc_vpn, auc, tp, fn, fp, tn]

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
