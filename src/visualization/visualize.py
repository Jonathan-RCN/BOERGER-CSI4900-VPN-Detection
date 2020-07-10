import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import scikitplot as skplt


from src.models.train_model import random_forest_classifyer_tuned,gradient_boost_classifier_tuned ,gradient_boost_classifier_default, random_forest_classifyer_default
import src.config as cfg
import src.feature_subsets as subset
import os

def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}.csv'
    label = subset.get_target(features_csv)

    random_state=42

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

    feature_subsets = [[baseline_ft, 'Baseline'],
                       [time_ft, 'Time'],
                       [conn_ft, 'Connection'],
                       [comb_ft, 'Combined'],
                       [time_fwd_ft, 'Time Forward'],
                       [time_bwd_ft, 'Time Backwards'],
                       [conn_fwd_ft, 'Connection Forward'],
                       [conn_bwd_ft, 'Connection Backward'],
                       [comb_fwd_ft, 'Combined Forward'],
                       [comb_bwd_ft, 'Combined Backwards']]

    # for ft_set in feature_subsets:
    #     print(f'  ///////////  {ft_set[1]} ///////////')
    #     print('Starting default model')
    #     model_info_default, time_info_default = random_forest_classifyer_default(ft_set[0], label, random_state=random_state)
    #     print('Starting tuned model')
    #     model_info_tuned, time_info_tuned = random_forest_classifyer_tuned(ft_set[0], label, random_state=random_state)
    #     print('Starting confusion matrix')
    #     plot_cf_matrix(model_info_default[2], model_info_default[4], model_info_default[1],
    #                    f'Confusion Matrix:: Random Forest - {ft_set[1]} Features - Default Param - Random State: {random_state} - RW Size: {connection_rw_size}-{timw_rw_size_min}')
    #     plot_cf_matrix(model_info_tuned[2], model_info_tuned[4], model_info_tuned[1],
    #                    f'Confusion Matrix:: Random Forest - {ft_set[1]} Features - Tuned Param - Random State: {random_state} - RW Size: {connection_rw_size}-{timw_rw_size_min}')
    #     print('Starting ROC curve')
    #     plot_ROC_curve(model_info_default[1],model_info_default[3],
    #                    f'ROC Curve:: Random Forest - {ft_set[1]} Features - Default Param - Random State: {random_state} - RW Size: {connection_rw_size}-{timw_rw_size_min}')
    #     plot_ROC_curve(model_info_tuned[1], model_info_tuned[3],
    #                    f'ROC Curve:: Random Forest - {ft_set[1]} Features - Tuned Param - Random State: {random_state} - RW Size: {connection_rw_size}-{timw_rw_size_min}')
    #
    # for ft_set in feature_subsets:
    #     print(f'  ///////////  {ft_set[1]} ///////////')
    #     print('Starting default model')
    #     model_info_default, time_info_default = gradient_boost_classifier_default(ft_set[0], label,
    #                                                                              random_state=random_state)
    #     print('Starting tuned model')
    #     model_info_tuned, time_info_tuned = gradient_boost_classifier_tuned(ft_set[0], label, random_state=random_state)
    #     print('Starting confusion matrix')
    #     plot_cf_matrix(model_info_default[2], model_info_default[4], model_info_default[1],
    #                    f'Confusion Matrix:: Gradient Boost - {ft_set[1]} Features - Default Param - Random State: {random_state} - RW Size: {connection_rw_size}-{timw_rw_size_min}')
    #     plot_cf_matrix(model_info_tuned[2], model_info_tuned[4], model_info_tuned[1],
    #                    f'Confusion Matrix:: Gradient Boost - {ft_set[1]} Features - Tuned Param - Random State: {random_state} - RW Size: {connection_rw_size}-{timw_rw_size_min}')
    #     print('Starting ROC curve')
    #     plot_ROC_curve(model_info_default[1], model_info_default[3],
    #                    f'ROC Curve:: Gradient Boost - {ft_set[1]} Features - Default Param - Random State: {random_state} - RW Size: {connection_rw_size}-{timw_rw_size_min}')
    #     plot_ROC_curve(model_info_tuned[1], model_info_tuned[3],
    #                    f'ROC Curve:: Gradient Boost - {ft_set[1]} Features - Tuned Param - Random State: {random_state} - RW Size: {connection_rw_size}-{timw_rw_size_min}')


    model_info_default, time_info_default = random_forest_classifyer_default(baseline_ft, label,
                                                                             random_state=random_state)

    feature_importance_v2(model_info_default[2], baselone_ft_name, 'Random Forest - Baseline - Default Parameters - Feature Importance')

    model_info_default, time_info_default = random_forest_classifyer_default(conn_ft, label,
                                                                             random_state=random_state)
    feature_importance_v2(model_info_default[2], conn_ft_name,
                       'Random Forest - Conn - Default Parameters - Feature Importance')
    model_info_default, time_info_default = random_forest_classifyer_default(comb_fwd_ft, label,
                                                                             random_state=random_state)
    feature_importance_v2(model_info_default[2], comb_fwd_ft_name,
                       'Random Forest - Comb Fwd - Default Parameters - Feature Importance')



def plot_cf_matrix(model, x_test,y_test,title ):
    disp = plot_confusion_matrix(model, x_test, y_test,display_labels=['Non-VPN', 'VPN'], cmap=plt.cm.Blues,
                                 normalize=None)
    disp.ax_.set_title(title[:title.find("- Random")])
    title = title.replace('::', ' --')
    title = title.replace(':', '')
    plt.savefig(f'{os.getcwd()}/cf_matrix/{title}.png')
    plt.show()


def plot_cf_matrix_normalized(model, x_test, y_test, title):
    disp = plot_confusion_matrix(model, x_test, y_test, display_labels=['Non-VPN', 'VPN'], cmap=plt.cm.Blues,
                                 normalize='True')
    disp.ax_.set_title(f'{title[:title.find("- Random")]}-Normaized')
    title = title.replace('::', '--')
    title = title.replace(':', '')

    plt.savefig(f'{os.getcwd()}/cf_matrix/{title}-Normaized.png')
    plt.show()


def plot_ROC_curve(y_test,prediction_probabilities, title):
    skplt.metrics.plot_roc(y_test, prediction_probabilities, title=title[:title.find("- Random")],plot_micro=False, plot_macro=False)
    title = title.replace('::', '--')
    title = title.replace(':', '')
    plt.savefig(f'{os.getcwd()}/roc_curve/{title}.png')
    plt.show()


def random_forest_single_tree(ft_name, rf_model, target_filename):
    fn = ft_name
    cn = ['Non-VPN', 'VPN']
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=800)
    from sklearn import tree
    from sklearn.tree import export_graphviz
    tree.plot_tree(rf_model[2].estimators_[1],
                   feature_names=fn,
                   class_names=cn,
                   filled=True)
    fig.savefig(f'{target_filename}.png')
    plt.show()

def feature_importance(model, ft_names, title, x_test):
    # adapted from https://chrisalbon.com/machine_learning/trees_and_forests/feature_importance/
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [ft_names for i in indices]
    plt.figure()
    plt.title(title)
    plt.bar(range(x_test.shape[1]), importances[indices])
    plt.xticks(range(x_test.shape[1]), names, rotation=45)

def feature_importance_v2(model, ft_names, title):

    df_feature_importance = pd.DataFrame(model.feature_importances_, index=ft_names,
                                         columns=['feature importance']).sort_values('feature importance',
                                                                                     ascending=False)
    df_feature_importance.plot(kind='bar', figsize=(8,4.5), title=title)
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.show()


if __name__ == '__main__':
    main()