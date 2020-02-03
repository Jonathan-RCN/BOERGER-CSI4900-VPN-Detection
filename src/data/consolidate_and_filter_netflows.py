import csv

import pandas as pd
import os


def main():
    netflow_list = []
    vpn_netflow_list = []
    print(os.getcwd())
    net_traffic_csv_dir = '../../data/raw/net_traffic_csv/'
    for file in os.listdir(net_traffic_csv_dir):
        if file.startswith("vpn"):
            vpn_netflow_list.append(file)
        else:
            netflow_list.append(file)

    file_name = netflow_list[4]
    print(file_name)
    src_data = pd.read_csv(net_traffic_csv_dir + file_name)
    coloums_to_keep = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Protocol', 'Tot Fwd Pkts',
                       'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Byts/s', 'Flow Pkts/s',
                       'Flow IAT Mean',
                       'Fwd Header Len', 'Bwd Header Len', 'Pkt Size Avg']

    filtered_netflow_csv_dir = '../../data/interim/'
    dest_filename = file_name.replace('.csv', '_filtered.csv')
    create_filtered_csv(dest_filename, filtered_netflow_csv_dir)
    filtered_data = pd.read_csv(filtered_netflow_csv_dir + dest_filename)
    dest_col_list = []
    for col in filtered_data.columns:
        dest_col_list.append(col)
    print(dest_col_list)
    y = 0
    for x in range(0, len(dest_col_list)):
        if y < len(coloums_to_keep):
            if dest_col_list[x] == coloums_to_keep[y]:
                for jj in range(0, src_data.shape[0] - 1):
                    filtered_data.loc[jj, dest_col_list[x]] = src_data.loc[jj, coloums_to_keep[y]]
                y += 1
            elif dest_col_list[x] == "Tot Pkts":
                for jj in range(0, src_data.shape[0] - 1):
                    filtered_data.loc[jj, dest_col_list[x]] = filtered_data.loc[jj, 'Tot Fwd Pkts'] + \
                                                              filtered_data.loc[jj, 'Tot Bwd Pkts']
            elif dest_col_list[x] == "TotLen Pkts":
                for jj in range(0, src_data.shape[0] - 1):
                    filtered_data.loc[jj, dest_col_list[x]] = filtered_data.loc[jj, 'TotLen Fwd Pkts'] + \
                                                              filtered_data.loc[jj, 'TotLen Bwd Pkts']
        elif dest_col_list[x] == "Flow Type":
            for jj in range(0, src_data.shape[0] - 1):
                filtered_data.loc[jj, dest_col_list[x]] = file_name.replace(".csv", "")
        elif dest_col_list[x] == "VPN":
            for jj in range(0, src_data.shape[0] - 1):
                if file_name.startswith("VPN"):
                    filtered_data.loc[jj, dest_col_list[x]] = 1
                else:
                    filtered_data.loc[jj, dest_col_list[x]] = 0

        filtered_data.to_csv(filtered_netflow_csv_dir + dest_filename)



def create_filtered_csv(dest_file_name, dest_dir):
    file = open(dest_dir + dest_file_name, 'w', newline='', encoding='utf-8')
    with file:
        header = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Protocol', 'Tot Fwd Pkts',
                  'Tot Bwd Pkts', 'Tot Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'TotLen Pkts',
                  'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Fwd Header Len', 'Bwd Header Len', 'Pkt Size Avg',
                  'Flow Type', 'VPN']
        writer = csv.writer(file)
        writer.writerow(header)


if __name__ == '__main__':
    main()

# todo questions:
# todo -> Do I keep src/dst port 0
# todo -> Do I keep FWD/BWK packet number = 0
