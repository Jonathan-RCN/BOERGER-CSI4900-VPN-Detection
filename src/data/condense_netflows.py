import csv

import pandas as pd
import numpy as np
import os
from datetime import datetime


def main():

    complete_netflow_list = []
    net_traffic_csv_dir = '../../data/raw/net_traffic_csv/'
    filtered_netflow_csv_dir = '../../data/interim/filtered/'
    cummlative_dir='../../data/processed/'
    cummulative_netflow_csv="cummulative_netflows.csv"



    for file in os.listdir(net_traffic_csv_dir):
        complete_netflow_list.append(file)
    filtered_netflows=[]
    for file in os.listdir(filtered_netflow_csv_dir):
        filtered_netflows.append(filtered_netflow_csv_dir+file)

    create_filtered_csv(cummulative_netflow_csv, cummlative_dir)
    produceOneCSV(filtered_netflows, cummlative_dir+cummulative_netflow_csv)



    columns_to_keep = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Tot Fwd Pkts',
                       'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts']
    count=0
    for file in complete_netflow_list:
        file_name = file
        dest_filename = file_name.replace('.csv', '_filtered.csv')





        if not os.path.exists(filtered_netflow_csv_dir + dest_filename):

            print(f' Started processing file: {file} || {count} of {len(complete_netflow_list)} || {datetime.now()} ')
            start_time=datetime.now()
            header =create_filtered_csv(dest_filename, filtered_netflow_csv_dir)
            src_data = pd.read_csv(net_traffic_csv_dir + file_name)
            transfer_netflow_data(src_data, columns_to_keep, file_name,
                                  filtered_netflow_csv_dir + dest_filename, header)
            end_time=datetime.now()
            print(f'Finished processing file: {file} || {count} of {len(complete_netflow_list)} || Total processing {end_time-start_time} ')
            print(
                "=======================================================================================================")

        count+=1

    # sorting by time     create_filtered_csv(cummulative_netflow_csv, cummlative_dir)
    produceOneCSV(filtered_netflows, cummlative_dir + cummulative_netflow_csv)
    cummulative_nf=pd.read_csv(cummlative_dir + cummulative_netflow_csv)
    cummulative_nf=cummulative_nf.sort_values(by='Timestamp')
    cummulative_nf.to_csv(cummlative_dir + cummulative_netflow_csv, index=False)
def create_filtered_csv(dest_file_name, dest_dir):
    file = open(dest_dir + dest_file_name, 'w', newline='', encoding='utf-8')
    with file:
        header = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Tot Pkts', 'TotLen','VPN']
        writer = csv.writer(file)
        writer.writerow(header)
    return header


def transfer_netflow_data(src_data, columns_to_keep, file_name, dest_dir_filename, header):
    filtered_data = pd.read_csv(dest_dir_filename)
    time_check=datetime.now()
    start_time=datetime.now()
    for x in range(0, src_data.shape[0]):
        if (x % 1000)==0:
            print(f'{x} of {src_data.shape[0]} || {datetime.now()}|| Time since last TC: {datetime.now()-time_check}|| Time since start: {datetime.now()-start_time}'  )
            time_check=datetime.now()
        to_append = []

        for y in range(0, len(header)):
            if header[y] == "VPN":
                if file_name.startswith("vpn"):
                    to_append.append(np.uint8(1))
                else:
                    to_append.append(np.uint8(0))

            elif header[y] == "Timestamp":
                    time_data=src_data.loc[x, columns_to_keep[y]]
                    time_data_2=pd.Timestamp(time_data)
                    to_append.append(time_data_2)
            elif header[y] == 'Src Port' or header[y] == 'Dst Port':
                port_data = src_data.loc[x, columns_to_keep[y]]
                to_append.append(np.uint16(port_data))
            elif header[y] =='Tot Pkts':
                pck_count=src_data.loc[x, "Tot Fwd Pkts"] + src_data.loc[x, "Tot Bwd Pkts"]
                to_append.append(pck_count)
            elif header[y]=='TotLen':
                tot_len = src_data.loc[x, "TotLen Fwd Pkts"] + src_data.loc[x, "TotLen Bwd Pkts"]
                to_append.append(tot_len)

            else:
                to_append.append(src_data.loc[x, columns_to_keep[y]])

        filtered_data.loc[x]=to_append


    filtered_data.to_csv(dest_dir_filename, index=False)


def produceOneCSV(list_of_files, file_out):
   # Consolidate all CSV files into one object
   result_obj = pd.concat([pd.read_csv(file) for file in list_of_files])
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")






if __name__ == '__main__':
    main()

# todo: clean it up
