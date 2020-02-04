import csv

import pandas as pd
import os
from datetime import datetime


def main():
    netflow_list = []
    vpn_netflow_list = []
    complete_netflow_list = []
    print(os.getcwd())
    net_traffic_csv_dir = '../../data/raw/net_traffic_csv/'
    filtered_netflow_csv_dir = '../../data/interim/'

    for file in os.listdir(net_traffic_csv_dir):
        if file.startswith("vpn"):
            vpn_netflow_list.append(file)
        else:
            netflow_list.append(file)
        complete_netflow_list.append(file)

    columns_to_keep = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Protocol', 'Tot Fwd Pkts',
                       'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Byts/s', 'Flow Pkts/s',
                       'Flow IAT Mean',
                       'Fwd Header Len', 'Bwd Header Len', 'Pkt Size Avg']
    count=0

    for file in complete_netflow_list:
        file_name = file
        dest_filename = file_name.replace('.csv', '_filtered.csv')





        if not os.path.exists(filtered_netflow_csv_dir + dest_filename):

            print(f' Started processing file: {file} || {count} of {len(complete_netflow_list)} || {datetime.now()} ')
            create_filtered_csv(dest_filename, filtered_netflow_csv_dir)
            src_data = pd.read_csv(net_traffic_csv_dir + file_name)
            filtered_data = pd.read_csv(filtered_netflow_csv_dir + dest_filename)
            transfer_netflow_data(filtered_data, src_data, columns_to_keep, file_name,
                                  filtered_netflow_csv_dir + dest_filename)
            print(
                "=======================================================================================================")

        count+=1


def create_filtered_csv(dest_file_name, dest_dir):
    file = open(dest_dir + dest_file_name, 'w', newline='', encoding='utf-8')
    with file:
        header = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Protocol', 'Tot Fwd Pkts',
                  'Tot Bwd Pkts', 'Tot Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'TotLen Pkts',
                  'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Fwd Header Len', 'Bwd Header Len', 'Pkt Size Avg',
                  'Flow Type', 'VPN']
        writer = csv.writer(file)
        writer.writerow(header)


def transfer_netflow_data(filtered_data, src_data, columns_to_keep, file_name, dest_dir_filename):
    dest_col_list = []
    for col in filtered_data.columns:
        dest_col_list.append(col)


    y = 0
    for x in range(0, len(dest_col_list)):
        print(f' Started processing : {dest_col_list[x]}-{file_name} || {x} of {len(dest_col_list)} || {datetime.now()} ')

        if y < len(columns_to_keep):
            if dest_col_list[x] == columns_to_keep[y]:
                for jj in range(0, src_data.shape[0] - 1):
                    if jj % 5000 == 0:
                        print(f'{jj} of {src_data.shape[0] - 1}')
                    filtered_data.loc[jj, dest_col_list[x]] = src_data.loc[jj, columns_to_keep[y]]
                y += 1

            elif dest_col_list[x] == "Tot Pkts":
                for jj in range(0, src_data.shape[0] - 1):
                    if jj % 5000 == 0:
                        print(f'{jj} of {src_data.shape[0] - 1}')
                    filtered_data.loc[jj, dest_col_list[x]] = filtered_data.loc[jj, 'Tot Fwd Pkts'] + \
                                                              filtered_data.loc[jj, 'Tot Bwd Pkts']
            elif dest_col_list[x] == "TotLen Pkts":
                for jj in range(0, src_data.shape[0] - 1):
                    if jj % 5000 == 0:
                        print(f'{jj} of {src_data.shape[0] - 1}')
                    filtered_data.loc[jj, dest_col_list[x]] = filtered_data.loc[jj, 'TotLen Fwd Pkts'] + \
                                                              filtered_data.loc[jj, 'TotLen Bwd Pkts']
        elif dest_col_list[x] == "Flow Type":
            for jj in range(0, src_data.shape[0] - 1):
                if jj % 5000 == 0:
                    print(f'{jj} of {src_data.shape[0] - 1}')
                filtered_data.loc[jj, dest_col_list[x]] = file_name.replace(".csv", "")

        elif dest_col_list[x] == "VPN":
            for jj in range(0, src_data.shape[0] - 1):
                if jj % 5000 == 0:
                    print(f'{jj} of {src_data.shape[0] - 1}')
                if file_name.startswith("VPN"):
                    filtered_data.loc[jj, dest_col_list[x]] = 1
                else:
                    filtered_data.loc[jj, dest_col_list[x]] = 0
        filtered_data.to_csv(dest_dir_filename)




if __name__ == '__main__':
    main()

# todo questions:
# todo -> Do I keep src/dst port 0
# todo -> Do I keep FWD/BWK packet number = 0
