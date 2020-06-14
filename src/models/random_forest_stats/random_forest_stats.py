import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
import src.config as cfg
import numpy as np
import pandas as pd
import sklearn
import sys
from tqdm import tqdm
import csv

def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}_2.csv'
    features_df = pd.read_csv(features_csv)

    label = np.array(features_df['VPN'])

    time_features = features_df.loc[:, 'Time Flow count': 'Time-Rev Pkt Len Tot'].columns
    conn_features = features_df.loc[:, 'Con Flow count': 'Conn-Rev Pkt Len Tot'].columns

    fwd_time_features=features_df.loc[:, 'Time Flow count': 'Time Pkt Len Tot'].columns
    bwd_time_features=features_df.loc[:, 'Time-Rev Flow count': 'Time-Rev Pkt Len Tot'].columns
    fwd_conn_features=features_df.loc[:, 'Con Flow count': 'Conn Pkt Len Tot'].columns
    bwd_conn_features=features_df.loc[:, 'Conn-Rev Flow count': 'Conn-Rev Pkt Len Tot'].columns


    baseline_drop_table = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'VPN']
    comb_drop_table = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'VPN','Tot Pkts', 'TotLen']
    time_drop_table = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
    conn_drop_table = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']

    time_fwd_dt=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
    time_bwd_dt = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
    conn_fwd_dt = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
    conn_bwd_dt = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
    comb_fwd_dt = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
    comb_bwd_dt = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']


    for tft in time_features:
        baseline_drop_table.append(tft)
        conn_drop_table.append(tft)
        conn_bwd_dt.append(tft)
        conn_fwd_dt.append(tft)

    for ctf in conn_features:
        baseline_drop_table.append(ctf)
        time_drop_table.append(ctf)
        time_fwd_dt.append(ctf)
        time_bwd_dt.append(ctf)

    for time_bwd_ft in bwd_time_features:
        time_fwd_dt.append(time_bwd_ft)
        comb_fwd_dt.append(time_bwd_ft)

    for time_fwd_ft in fwd_time_features:
        time_bwd_dt.append(time_fwd_ft)
        comb_bwd_dt.append(time_fwd_ft)

    for conn_bwd_ft in bwd_conn_features:
        conn_fwd_dt.append(conn_bwd_ft)
        comb_fwd_dt.append(conn_bwd_ft)

    for conn_fwd_ft in fwd_conn_features:
        conn_bwd_dt.append(conn_fwd_ft)
        comb_bwd_dt.append(conn_fwd_ft)


    rf_baseline_ft = features_df.drop(baseline_drop_table, axis=1)
    rf_time_ft = features_df.drop(time_drop_table, axis=1)
    rf_conn_ft = features_df.drop(conn_drop_table, axis=1)
    rf_comb_ft = features_df.drop(comb_drop_table,axis=1)

    rf_time_fwd_ft=features_df.drop(time_fwd_dt,axis=1)
    rf_time_bwd_ft = features_df.drop(time_bwd_dt, axis=1)
    rf_conn_fwd_ft = features_df.drop(conn_fwd_dt, axis=1)
    rf_conn_bwd_ft = features_df.drop(conn_bwd_dt, axis=1)
    rf_comb_fwd_ft = features_df.drop(comb_fwd_dt, axis=1)
    rf_comb_bwd_ft = features_df.drop(comb_bwd_dt, axis=1)

    baseline_ft_name = list(rf_baseline_ft.columns)
    time_ft_name=list(rf_time_ft.columns)
    conn_ft_name=list(rf_conn_ft.columns)
    comb_ft_name=list(rf_comb_ft.columns)

    rf_baseline_ft=np.array(rf_baseline_ft)
    rf_time_ft=np.array(rf_time_ft)
    rf_conn_ft=np.array(rf_conn_ft)
    rf_comb_ft=np.array(rf_comb_ft)

    rf_time_fwd_ft = np.array(rf_time_fwd_ft)
    rf_time_bwd_ft = np.array(rf_time_bwd_ft)
    rf_conn_fwd_ft = np.array(rf_conn_fwd_ft)
    rf_conn_bwd_ft = np.array(rf_conn_bwd_ft)
    rf_comb_fwd_ft = np.array(rf_comb_fwd_ft)
    rf_comb_bwd_ft = np.array(rf_comb_bwd_ft)

    model_sets=[[rf_time_ft, "rf_time.csv", "time" ],
                [rf_conn_ft, "rf_connection.csv", "conn"],
                [rf_comb_ft, "rf_combined.csv", "comb"],
                [rf_time_fwd_ft, "rf_time_forward.csv", "time fwd"],
                [rf_time_bwd_ft, "rf_time_backwards.csv", "time bwd"],
                [rf_conn_fwd_ft, "rf_connection_forwards.csv", "conn fwd"],
                [rf_conn_bwd_ft, "rf_connection_backwards.csv", "conn bwd"],
                [rf_comb_fwd_ft, "rf_combined_forward.csv", "comb fwd"],
                [rf_comb_bwd_ft, "rf_combined_backwards.csv", "comb bwd"],
                ]

    # random_forest_classifyer_stats(rf_baseline_ft,label,'rf_baseline.csv','baseline')


    for model_set in model_sets:
        data=pd.read_csv(model_set[1])
        if data.shape[0]==100:
            continue
        else:
            random_forest_classifyer_stats(model_set[0], label, model_set[1], model_set[2])













def random_forest_classifyer_stats(feature_data, label, target_csv, target_col, test_size=0.3, random_state=None):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    metric_list=[]
    cv_acc_list=[]
    cv_auc_list=[]
    data = pd.read_csv('general_stats.csv')
    t = tqdm(total=100)
    for x in range (0,100):
        x_train, x_test, y_train, y_test = train_test_split(
            feature_data, label, test_size=test_size, random_state=random_state)
        with warnings.catch_warnings():
            random_forest = RandomForestClassifier(n_jobs=2, random_state=random_state)
            warnings.filterwarnings("ignore")

        random_forest.fit(x_train, y_train)

        predictions = random_forest.predict(x_test)
        rfc_cv_score = cross_val_score(random_forest, feature_data, label, cv=10, scoring='roc_auc',n_jobs=2)
        cv_auc_score=rfc_cv_score.mean()
        rfc_cv_score_acc = cross_val_score(random_forest, feature_data, label, cv=10, scoring='accuracy',n_jobs=2)
        cv_acc_score=rfc_cv_score_acc.mean()

        acc=accuracy_score(y_test,predictions)
        # f1_test=f1_score(y_test,predictions, average=None)
        f1,f1_vpn=f1_score(y_test,predictions, average=None)

        prec,prec_vpn = precision_score(y_test, predictions, average=None)
        recc,recc_vpn = recall_score(y_test, predictions, average=None)
        auc=roc_auc_score(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()
        to_append=[fp,fn,tp,prec,prec_vpn,recc,recc_vpn,f1,f1_vpn, acc, cv_acc_score, auc, cv_auc_score]
        to_append_string=str(tn)

        for ele in to_append:
            to_append_string+=f', {str(ele)}'
        # metric_list.append(to_append_string)
        # cv_acc_list.append(cv_acc_score)
        # cv_auc_list.append(cv_auc_score)
        t.update(1)

        file = open(target_csv, 'a', newline='', encoding='utf-8')
        with file:
            writer=csv.writer(file)
            writer.writerow(to_append_string.split(', '))


    data=pd.read_csv('general_stats.csv')
    data_2=pd.read_csv(target_csv)
    data[f'{target_col} acc']=data_2['cv accuracy']
    data[f'{target_col} auc'] = data_2['cv auc']
    data.to_csv('general_stats.csv', index=False, encoding='utf-8')






if __name__ == '__main__':
    main()



