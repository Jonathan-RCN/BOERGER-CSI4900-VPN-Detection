from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import src.config as cfg
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm



def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}_2.csv'
    features_df = pd.read_csv(features_csv)

    label = np.array(features_df['VPN'])

    baseline_drop_table = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'VPN']
    comb_drop_table = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'VPN','Tot Pkts', 'TotLen']
    time_drop_table = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
    conn_drop_table = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
    time_features = features_df.loc[:, 'Time Flow count': 'Time-Rev Pkt Len Tot'].columns
    conn_features = features_df.loc[:, 'Con Flow count': 'Conn-Rev Pkt Len Tot'].columns

    for tft in time_features:
        baseline_drop_table.append(tft)
        conn_drop_table.append(tft)
    for ctf in conn_features:
        baseline_drop_table.append(ctf)
        time_drop_table.append(ctf)

    rf_baseline_ft = features_df.drop(baseline_drop_table, axis=1)
    rf_time_ft = features_df.drop(time_drop_table, axis=1)
    rf_conn_ft = features_df.drop(conn_drop_table, axis=1)
    rf_comb_ft = features_df.drop(comb_drop_table,axis=1)

    baseline_ft_name = list(rf_baseline_ft.columns)
    time_ft_name=list(rf_time_ft.columns)
    conn_ft_name=list(rf_conn_ft.columns)
    comb_ft_name=list(rf_comb_ft.columns)

    rf_baseline_ft_array=np.array(rf_baseline_ft)
    rf_time_ft_array=np.array(rf_time_ft)
    rf_conn_ft_array=np.array(rf_conn_ft)
    rf_comb_ft_array=np.array(rf_comb_ft)

    baseline_predictions, baseline_y_test, baseline_model=random_forest_classifyer(rf_baseline_ft_array,label)
    time_predictions, time_y_test, time_model = random_forest_classifyer(rf_time_ft_array, label)
    conn_predictions, conn_y_test, conn_model = random_forest_classifyer(rf_conn_ft_array, label)
    comb_predictions, comb_y_test, comb_model = random_forest_classifyer(rf_comb_ft_array, label)


    print('///////////////// Baseline /////////////////')
    print('\n')
    evaluate_rf_results(baseline_predictions,baseline_y_test,baseline_model, rf_baseline_ft_array, label)
    feature_importance(baseline_model, baseline_ft_name)
    print('///////////////// Time /////////////////')
    print('\n')
    evaluate_rf_results(time_predictions,time_y_test, time_model, rf_time_ft_array, label)
    print('///////////////// Connection /////////////////')
    print('\n')
    evaluate_rf_results(conn_predictions,conn_y_test,conn_model,rf_conn_ft_array,label)
    print('///////////////// Combined /////////////////')
    print('\n')
    evaluate_rf_results(comb_predictions, comb_y_test, comb_model, rf_comb_ft_array, label)
    feature_importance(comb_model,comb_ft_name)






def random_forest_classifyer(feature_data, label, test_size=0.3, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    random_forest = RandomForestClassifier(n_jobs=2, random_state=random_state)
    start_time=datetime.now()
    random_forest.fit(x_train, y_train)
    print(datetime.now()-start_time)

    start_time = datetime.now()
    predictions = random_forest.predict(x_test)
    print(datetime.now() - start_time)
    return predictions, y_test, random_forest


def evaluate_rf_results(predictions, y_test, model, feature_data, label):
    start_time = datetime.now()
    rfc_cv_score = cross_val_score(model, feature_data, label, cv=10, scoring='roc_auc')
    print(datetime.now() - start_time)
    start_time = datetime.now()
    rfc_cv_score_acc = cross_val_score(model, feature_data, label, cv=10, scoring='accuracy')
    print(datetime.now() - start_time)
    print("=== Confusion Matrix ===")
    start_time = datetime.now()
    cf_matrix = confusion_matrix(y_test, predictions)
    print(datetime.now() - start_time)
    print(cf_matrix)
    print('\n')
    print("=== Classification Report ===")
    start_time = datetime.now()
    class_report = classification_report(y_test, predictions)
    print(datetime.now() - start_time)
    print(class_report)
    print('\n')
    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
    print('\n')
    print("=== All ACC Scores ===")
    print(rfc_cv_score_acc)
    print('\n')
    print("=== Mean ACC Score ===")
    print("Mean ACC Score - Random Forest: ", rfc_cv_score_acc.mean())


    return cf_matrix, class_report, rfc_cv_score, rfc_cv_score.mean()


def feature_importance(model, feature_names):
    fi = pd.DataFrame({'feature': feature_names,
                       'importance': model.feature_importances_}). \
        sort_values('importance', ascending=False)
    print(fi)
    return fi

if __name__ == '__main__':
    main()
# print(features_connection)

# connection_train_ft, connection_test_ft, connection_train_labels, conection_test_labels=train_test_split(features_connection, label, test_size=0.3, random_state=42)
#
# print(f'Training Features Shape: {connection_train_ft.shape}' )
# print(f'Training Labels Shape:{connection_test_ft.shape}' )
# print(f'Testing Features Shape:{connection_train_labels.shape}')
# print(f'Testing Labels Shape:{conection_test_labels.shape}')
#
# accuracy_list=[]
#
# random_forest= RandomForestClassifier(n_jobs=2)
# # t = tqdm(total=1000)
# # for xx in range(0,1000):
#
# random_forest.fit(connection_train_ft,connection_train_labels)
#
# predictions=random_forest.predict(connection_test_ft)
#
#
#
# matches=0
#
# # print(int(predictions[20]))
# # print(int(conection_test_labels[20]))
#
#
# for x in range(0, predictions.shape[0]):
#     if int(predictions[x])==int(conection_test_labels[x]):
#         matches+=1
#
# # print(matches/predictions.shape[0])
# accuracy_list.append(matches/predictions.shape[0])
# # t.update(1)
#
# print('///////////////////////////////////')
# print(max(accuracy_list))
# print(min(accuracy_list))
# print(sum(accuracy_list)/len(accuracy_list))
#
# rfc_cv_score = cross_val_score(random_forest, features_connection, label, cv=10, scoring='roc_auc')
# print("=== Confusion Matrix ===")
# print(confusion_matrix(conection_test_labels, predictions))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(conection_test_labels, predictions))
# print('\n')
# print("=== All AUC Scores ===")
# print(rfc_cv_score)
# print('\n')
# print("=== Mean AUC Score ===")
# print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
#
# fi=pd.DataFrame({'feature': features_connection_names,
#                    'importance': random_forest.feature_importances_}).\
#                     sort_values('importance', ascending = False)
#
# print(fi)
# print("/////////////")
# print(fi.head())