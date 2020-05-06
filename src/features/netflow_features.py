import csv

import pandas as pd
import numpy as np
from datetime import datetime
SRC_IP=1
SRC_PORT=2
DEST_IP=3
DEST_PORT=4
TIMESTAMP=5
PACKET_COUNT=6
PACKET_LENGTH=7
"""
Issues: 

The netflows may have different src/dest ports so allthough the IP addr match the ports do not. I don't think
its a problem but will need to confirm. However, if it is, then will use list to keep track of ports

Also issue with dealing with the 0-th flow in terms of rolling windoow fade out

Outstanding:
-Method for efficiencies:
    Every 5000 check to see if the last TTL was is past the rw and if so then delete entry in dictionary 
    Else validate the ttl values 
Method to check if the csv is already completled, and if not continue from previous point 
Standard deviation 
Current estimated full completion time: 5hrs (13min per 10k) 
"""

class rw_netflow:
    """
    Rolling window netflow -> class which contians the cummulative info for all the netflows in the same RW
    """
    def __init__(self, src_ip, src_port, dest_ip, dest_port, packet_count_list, packet_length_list, timestamp_list, ttl_list):
        """Course init."""
        self.src_ip = src_ip
        self.src_port = src_port
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.connection_packet_count_list=[packet_count_list]
        self.connection_packet_length_list=[packet_length_list]
        self.connection_timestamp_list=[pd.Timestamp(timestamp_list)]
        self.connection_ttl_list=[ttl_list]
        self.connection_flow_count=1
        self.time_packet_count_list = [packet_count_list]
        self.time_packet_length_list = [packet_length_list]
        self.time_timestamp_list = [pd.Timestamp(timestamp_list)]
        self.time_ttl_list = [ttl_list]
        self.time_flow_count = 1
    def add_subsequent_flow_data(self, new_timestamp, new_packet_length,new_packet_count, new_ttl_index):
        self.connection_packet_length_list.append(new_packet_length)
        self.connection_packet_count_list.append(new_packet_count)
        self.connection_ttl_list.append(new_ttl_index)
        self.connection_timestamp_list.append(pd.Timestamp(new_timestamp))
        self.connection_flow_count+=1
        self.time_packet_length_list.append(new_packet_length)
        self.time_packet_count_list.append(new_packet_count)
        self.time_ttl_list.append(new_ttl_index)
        self.time_timestamp_list.append(pd.Timestamp(new_timestamp))
        self.time_flow_count += 1
    def validate_connection_flow_ttl(self, rw_size, current_index):
        if len(self.connection_ttl_list)>1:
            while self.connection_ttl_list!=[]:
                if current_index-self.connection_ttl_list[0]>=rw_size:
                    self.connection_ttl_list.pop(0)
                    self.connection_timestamp_list.pop(0)
                    self.connection_packet_count_list.pop(0)
                    self.connection_packet_length_list.pop(0)
                    self.connection_flow_count= self.connection_flow_count - 1
                else:
                    break
    def validate_time_flow_ttl(self,rw_time, curent_timestamp):
        # validate for time based rolling
        if len(self.time_ttl_list) > 1:

            while self.time_ttl_list!=[]:
                if (curent_timestamp-self.time_timestamp_list[0]).total_seconds()>=rw_time*60:
                    self.time_ttl_list.pop(0)
                    self.time_timestamp_list.pop(0)
                    self.time_packet_count_list.pop(0)
                    self.time_packet_length_list.pop(0)
                    self.time_flow_count-=1

                else:
                    break

    def calculate_flow_connection_based_features(self, current_index, rw_size):
        self.validate_connection_flow_ttl(rw_size, current_index)
        time_based_feat_list=min_max_mean_time_delay(self.connection_timestamp_list)
        packet_len_feat_list=min_max_mean_total_feature(self.connection_packet_length_list)
        packet_num_feat_list=min_max_mean_total_feature(self.connection_packet_count_list)
        return [self.connection_flow_count], time_based_feat_list, packet_len_feat_list, packet_num_feat_list
    def caulculate_flow_time_based_features(self, curent_timestamp, rw_time):
        self.validate_time_flow_ttl(rw_time, curent_timestamp)
        time_based_feat_list=min_max_mean_time_delay(self.time_timestamp_list)
        packet_len_feat_list=min_max_mean_total_feature(self.time_packet_length_list)
        packet_num_feat_list=min_max_mean_total_feature(self.time_packet_count_list)
        return [self.time_flow_count], time_based_feat_list, packet_len_feat_list, packet_num_feat_list

    def print_connection_netflow(self):
        print(f' Src IP/Port: {self.src_ip}/{self.src_port}')
        print(f' Dest IP/Port: {self.dest_ip}/{self.dest_port}')
        print(f' Flow count in the RW: {self.connection_flow_count}')
        print(f' The data TTL values: {self.connection_ttl_list}')
        print(f' The timestamp list: {self.connection_timestamp_list}')
        print(f' The packet count list: {self.connection_packet_count_list}')
        print(f' The packet length list: {self.connection_packet_length_list}')
    def print_time_netflow(self):
        print(f' Src IP/Port: {self.src_ip}/{self.src_port}')
        print(f' Dest IP/Port: {self.dest_ip}/{self.dest_port}')
        print(f' Flow count in the RW: {self.time_flow_count}')
        print(f' The data TTL values: {self.time_ttl_list}')
        print(f' The timestamp list: {self.time_timestamp_list}')
        print(f' The packet count list: {self.time_packet_count_list}')
        print(f' The packet length list: {self.time_packet_length_list}')
    def print_complete_netflow(self):
        print(f' Src IP/Port: {self.src_ip}/{self.src_port}')
        print(f' Dest IP/Port: {self.dest_ip}/{self.dest_port}')
        print(f' Flow count in the RW: [Con]>> {self.connection_flow_count} '
              f'|| [Time]>> {self.time_flow_count}')
        print(f' The data TTL values: [Con]>> {self.connection_ttl_list} '
              f'|| [Time]>> {self.time_ttl_list}')
        print(f' The timestamp list: [Con]>> {self.connection_timestamp_list} '
              f'|| [Time]>> {self.time_timestamp_list}')
        print(f' The packet count list: [Con]>> {self.connection_packet_count_list} '
              f'|| [Time]>> {self.time_packet_count_list}')
        print(f' The packet length list: [Con]>> {self.connection_packet_length_list} '
              f'|| [Time]>> {self.time_packet_length_list}')

def main():
    connection_rw_size=10000
    timw_rw_size_min=10
    condensed_netflow_csv_file='../../data/processed/cummulative_netflows.csv'
    full_feature_netflow_csv_file=f'../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}.csv'
    extracted_feature_csv_creator(full_feature_netflow_csv_file)


    netflow_feature_extraction(condensed_netflow_csv_file,full_feature_netflow_csv_file,connection_rw_size,timw_rw_size_min)
    #test_bed(netflow_csv_file)

    """
    need to check out pandans rolling windows to see if they have any inbuilt methods to assit
    """

def netflow_feature_extraction(netflow_csv_file, target_csv_file, connection_rw_size, time_rw_size):
    start1=datetime.now()
    netflow_dictionary = {}
    netflow_data = pd.read_csv(netflow_csv_file)
    netflow_data = netflow_data.astype(
        {'Src Port': np.uint16, 'Dst Port': np.uint16, 'Timestamp': np.datetime64, 'VPN': np.uint8})
    for netflow_index in range(1, netflow_data.shape[0]):
        if netflow_index==10000:
            start2=datetime.now()
        if netflow_index==20000:
            start3=datetime.now()
        if netflow_index == 30000:
            start4 = datetime.now()
        if netflow_index == 40000:
            start5 = datetime.now()


        if netflow_index % 1000 ==0:
            print(f'{netflow_index} of {netflow_data.shape[0]}')
        src_dest_ip_tag = f'{netflow_data.iloc[netflow_index, SRC_IP]}-{netflow_data.iloc[netflow_index, DEST_IP]}'
        dest_src_ip_tag = f'{netflow_data.iloc[netflow_index, DEST_IP]}-{netflow_data.iloc[netflow_index, SRC_IP]}'

        if src_dest_ip_tag in netflow_dictionary:
            netflow_dictionary[src_dest_ip_tag].add_subsequent_flow_data(netflow_data.iloc[netflow_index, TIMESTAMP],
                                                                         netflow_data.iloc[netflow_index, PACKET_LENGTH],
                                                                         netflow_data.iloc[
                                                                             netflow_index, PACKET_COUNT],
                                                                         netflow_index)
        else:
            netflow_dictionary[src_dest_ip_tag] = rw_netflow(netflow_data.iloc[netflow_index, SRC_IP],
                                                             netflow_data.iloc[netflow_index, SRC_PORT],
                                                             netflow_data.iloc[netflow_index, DEST_IP],
                                                             netflow_data.iloc[netflow_index, DEST_PORT],
                                                             netflow_data.iloc[netflow_index, PACKET_COUNT],
                                                             netflow_data.iloc[netflow_index, PACKET_LENGTH],
                                                             netflow_data.iloc[netflow_index, TIMESTAMP],
                                                             netflow_index)

        con_flow_count, con_timedelta_ft_list, con_pkt_len_ft_list, con_pkt_num_ft_list = \
            netflow_dictionary[src_dest_ip_tag].calculate_flow_connection_based_features(netflow_index,
                                                                                         connection_rw_size)
        time_flow_count, time_timedelta_ft_list, time_pkt_len_ft_list, time_pkt_num_ft_list = \
            netflow_dictionary[src_dest_ip_tag].caulculate_flow_time_based_features(
                netflow_data.iloc[netflow_index, TIMESTAMP], time_rw_size)

        if dest_src_ip_tag in netflow_dictionary:

            rev_con_flow_count, rev_con_timedelta_ft_list, rev_con_pkt_len_ft_list, rev_con_pkt_num_ft_list = \
                netflow_dictionary[dest_src_ip_tag].calculate_flow_connection_based_features(netflow_index,
                                                                                             connection_rw_size)
            rev_time_flow_count, rev_time_timedelta_ft_list, rev_time_pkt_len_ft_list, rev_time_pkt_num_ft_list = \
                netflow_dictionary[dest_src_ip_tag].caulculate_flow_time_based_features(
                    netflow_data.iloc[netflow_index, TIMESTAMP], time_rw_size)
        else:
            rev_con_flow_count = [0]
            rev_con_timedelta_ft_list = [0, 0, 0]
            rev_con_pkt_len_ft_list = [0, 0, 0, 0]
            rev_con_pkt_num_ft_list = [0, 0, 0, 0]
            rev_time_flow_count = [0]
            rev_time_timedelta_ft_list = [0, 0, 0]
            rev_time_pkt_len_ft_list = [0, 0, 0, 0]
            rev_time_pkt_num_ft_list = [0, 0, 0, 0]
        engineered_features=[con_flow_count, con_timedelta_ft_list,  con_pkt_num_ft_list,con_pkt_len_ft_list,
                         rev_con_flow_count, rev_con_timedelta_ft_list, rev_con_pkt_num_ft_list, rev_con_pkt_len_ft_list,
                         time_flow_count, time_timedelta_ft_list,  time_pkt_num_ft_list, time_pkt_len_ft_list,
                         rev_time_flow_count, rev_time_timedelta_ft_list,  rev_time_pkt_num_ft_list,rev_time_pkt_len_ft_list]

        base_features=[netflow_data.iloc[netflow_index, 0],
            netflow_data.iloc[netflow_index, SRC_IP],
            netflow_data.iloc[netflow_index, SRC_PORT],
            netflow_data.iloc[netflow_index, DEST_IP],
            netflow_data.iloc[netflow_index, DEST_PORT],
            netflow_data.iloc[netflow_index, TIMESTAMP],
            netflow_data.iloc[netflow_index, PACKET_COUNT],
            netflow_data.iloc[netflow_index, PACKET_LENGTH]]
        add_features_to_csv(target_csv_file, base_features, engineered_features,netflow_data.iloc[netflow_index, 8] )
    endtime=datetime.now()
    print(f' Totaltime: {endtime-start1}')
    print(f' First 10000: {start2-start1}')
    print(f'  10k- 20k: {start3 - start2}')
    print(f' 20k -30k: {start4 - start3}')
    print(f' 30k -40k: {start5 - start4}')
    print(f' 40k -50k: {endtime - start5}')
    #print(f' 10k -30k: {endtime - start2}')

def add_features_to_csv(csv_filename, base_features, engineered_features, vpn_status):
    to_append=''
    for elements in base_features:
        to_append= f'{to_append}, {elements}'
    for list in engineered_features:
        for list_element in list:
            to_append = f'{to_append}, {list_element}'
    to_append = f'{to_append}, {vpn_status}'
    file = open(csv_filename, 'a', newline='', encoding='utf-8')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append[1:].split(','))

def min_max_mean_total_feature(feature_list):
    if feature_list == []:
        feature_list=[0]
    feature_min=min(feature_list)
    feature_max=max(feature_list)
    feature_total=sum(feature_list)
    feature_mean=feature_total/len(feature_list)

    return [feature_min, feature_max, feature_mean, feature_total]


def min_max_mean_time_delay(timestamp_list):
    if timestamp_list==[]:
        timestamp_list=[0]
    timeflow_differential = []
    for time_flow in range(1, len(timestamp_list)):
        time_delta = timestamp_list[time_flow] - timestamp_list[time_flow - 1]
        timeflow_differential.append(abs(time_delta.total_seconds()))
    if timeflow_differential==[]:
        timeflow_differential=[0]
    min_time_delta = min(timeflow_differential)
    max_time_delta = max(timeflow_differential)
    mean_time_delta = sum(timeflow_differential) / len(timeflow_differential)
    return [min_time_delta, max_time_delta, mean_time_delta]

def extracted_feature_csv_creator(csv_file_name):

    file = open(csv_file_name, 'w', newline='', encoding='utf-8')
    with file:
        header = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Tot Pkts', 'TotLen',
                  'Con Flow count','Conn Timedelta Min', 'Conn Timedelta Max', 'Conn Timedelta Mean',
                  'Conn Pkt Num Min', 'Conn Pkt Num Max', 'Conn Pkt Num Mean', 'Conn Pkt Num Tot',
                  'Conn Pkt Len Min', 'Conn Pkt Len Min', 'Conn Pkt Len Mean', 'Conn Pkt Len Tot',
                  'Conn-Rev Flow count', 'Conn-Rev Timedelta Min', 'Conn-Rev Timedelta Max', 'Conn-Rev Timedelta Mean',
                  'Conn-Rev Pkt Num Min', 'Conn-Rev Pkt Num Max', 'Conn-Rev Pkt Num Mean', 'Conn-Rev Pkt Num Tot',
                  'Conn-Rev Pkt Len Min', 'Conn-Rev Pkt Len Min', 'Conn-Rev Pkt Len Mean', 'Conn-Rev Pkt Len Tot',
                  'Time Flow count', 'Time Timedelta Min', 'Time Timedelta Max', 'Time Timedelta Mean',
                  'Time Pkt Num Min', 'Time Pkt Num Max', 'Time Pkt Num Mean', 'Time Pkt Num Tot',
                  'Time Pkt Len Min', 'Time Pkt Len Min', 'Time Pkt Len Mean', 'Time Pkt Len Tot',
                  'Time-Rev Flow count',
                  'Time-Rev Timedelta Min', 'Time-Rev Timedelta Max', 'Time-Rev Timedelta Mean',
                  'Time-Rev Pkt Num Min', 'Time-Rev Pkt Num Max', 'Time-Rev Pkt Num Mean', 'Time-Rev Pkt Num Tot',
                  'Time-Rev Pkt Len Min', 'Time-Rev Pkt Len Min', 'Time-Rev Pkt Len Mean', 'Time-Rev Pkt Len Tot', 'VPN'
                  ]
        writer = csv.writer(file)
        writer.writerow(header)







if __name__ == '__main__':
    main()

