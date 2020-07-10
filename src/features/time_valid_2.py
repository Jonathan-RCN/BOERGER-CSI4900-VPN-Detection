"""
Title: Feature Engineering Module

Project: CSI4900 Honours Project

Created: Feb 2020
Last modified: 06 May 2020

Author: Jonathan Boerger
Status: In Progress

Description: This module take the pre-processed data and uses it to extract expanded (engineered features) to be used
in the classifier.

"""

import csv
import random

import pandas as pd
import numpy as np
from datetime import datetime
import src.config as cfg
from tqdm import tqdm

RAW_DATA_DICT={}
NETFLOW_DICT = {}
EXPANDED_FT_DIC={}


"""
Issues: 

The netflows may have different src/dest ports so all though the IP addr match the ports do not. I don't think
its a problem but will need to confirm. However, if it is, then will use list to keep track of ports

Also issue with dealing with the 0-th flow in terms of rolling window fade out

Outstanding:
-Method for efficiencies:
    Every 5000 flows check to see if the last TTL was is past the rw and if so then delete entry in dictionary
    When validating TTL values check to see if the last value in the list exceeds the TTL, thus one check done if possible instead
    of iterating across entire list (eliminating the need to validate at every 5000 check) 
    Else validate the ttl values 
Method to check if the csv is already completled, and if not continue from previous point 
Standard deviation 
Current estimated full completion time: 5hrs (13min per 10k) 
"""
"""
Todo (in addition to above):
-Adapt netflow_feature_extraction method (as a new method)for more real time use (does not itearate through file but 
each time new netflow is processed then the method is used to check the dictionary and generate the enhanced features
such that it can be passed to the ML model for classification

-Apadpt the csv creator method to validate if a file already exist and if it contains the required number of lines.


- Search for general efficiencies
    |-->https://towardsdatascience.com/how-to-make-your-pandas-loop-71-803-times-faster-805030df4f06
        |--> iterrow
        |--> research Cython space WRT apply

Efficiency check (Friday)
-Time to complete the whole csv list
-Time to retreive enhance features for csv (maybe also consider a dict for this infromation) 
-Time to calculate from nil (retoactively) enhanced features for a random netflow 
-Time to calculate and generate features for a progressive netflow (i.e. when a new netflow is processed, given already
progressed flows) 
-Compare times for different RW sizes 


"""


class RW_NETFLOW:
    """
    Rolling window netflow -> class which contians the cummulative info for all the netflows in the same RW
    """

    def __init__(self, src_ip, src_port, dest_ip, dest_port, packet_count_list, packet_length_list, timestamp_list,
                 ttl_list):
        """
        This initalizer methods creates the rw_netflow object upon encountering a new netflow (determined by
        src and dest IP pair). As such the methods takes the basic netflow characteristics to initialize the
        characteristics list used to track the netflow.

        Connection based RW and time based RW characteristics list are maintained separately since although they may
        have the same starting point, the continue relevance of the given flow varies between the two methods.

        :param src_ip: The source IP address for the given netflow
        :param src_port: The source port for the given netflow
        :param dest_ip: The destination IP address for the given netflow
        :param dest_port: The destination port for the given netflow
        :param packet_count_list: The number of packets in the given netflow
        :param packet_length_list: The total length of the packets in bytes of the netflow
        :param timestamp_list: The given timestamp of the netflow
        :param ttl_list: The file index which corresponds to the netflow
        """
        self.src_ip = src_ip
        self.src_port = src_port
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.connection_packet_count_list = [packet_count_list]
        self.connection_packet_length_list = [packet_length_list]
        self.connection_timestamp_list = [pd.Timestamp(timestamp_list)]
        self.connection_ttl_list = [ttl_list]
        self.connection_flow_count = 1
        self.time_packet_count_list = [packet_count_list]
        self.time_packet_length_list = [packet_length_list]
        self.time_timestamp_list = [pd.Timestamp(timestamp_list)]
        self.time_ttl_list = [ttl_list]
        self.time_flow_count = 1

    def add_subsequent_flow_data(self, new_timestamp, new_packet_length, new_packet_count, new_ttl_index):
        """
        This methods used when a rw_netflow object already exist for the given src dest IP pair.
        The methods adds the netflow characteristics for the given netflow to the object's characteristics list.

        The values are added both to connection and time based characteristics lists.

        The number of active flow is also increased to reflected having added a flow.

        :param new_timestamp: Timestamp of the netflow in question
        :param new_packet_length: Total packet length in bytes of the netflow in question
        :param new_packet_count: Count of packets in the netflow in question
        :param new_ttl_index:  The file index for the netflow in question
        :return: Updated rw_netflow object
        """
        self.connection_packet_length_list.append(new_packet_length)
        self.connection_packet_count_list.append(new_packet_count)
        self.connection_ttl_list.append(new_ttl_index)
        self.connection_timestamp_list.append(pd.Timestamp(new_timestamp))
        self.connection_flow_count += 1
        self.time_packet_length_list.append(new_packet_length)
        self.time_packet_count_list.append(new_packet_count)
        self.time_ttl_list.append(new_ttl_index)
        self.time_timestamp_list.append(pd.Timestamp(new_timestamp))
        self.time_flow_count += 1

    def validate_connection_flow_ttl(self, rw_size, current_index):
        """
        This methods determines if the flow features in the characteristics list are still relevant
        given the connection based rolling window size.

        The relevance is assed using the TTL (time to live) list values. Since these values represent the file index
        of when the characteristic set is added to the list, if the difference between the current index and the TTL
        value exceeds the RW size it indicates that the given netflow feature set is no longer relevant.

        Then netflow feature set includes, packet count, packets total size, time stamp.

        When the feature set is deemed to no longer be relevant, all the values of associated with the set is removed
        from the characteristics list and the flow count is decremented.

        :param rw_size: The number of flow to be considered in the rolling window.
        :param current_index: The current netflow index
        :return: A rw_netflow object that has been validated as having only relevant netflow feature sets
        """
        # only consider netflows_objects that actually have values
        if len(self.connection_ttl_list) > 1:

            # if the last value in the TLL list is outside the RW, the whole list is outside the RW
            if current_index - self.connection_ttl_list[len(self.connection_ttl_list)-1]>= rw_size:
                #print('All conn flow char data is no longer relevant')
                self.connection_ttl_list=[]
                self.connection_timestamp_list=[]
                self.connection_packet_count_list=[]
                self.connection_packet_length_list=[]
                self.connection_flow_count =0
                return
            # todo: test to see if this is more efficient
            break_value=0
            for ttl in self.connection_ttl_list:
                if current_index - ttl >= rw_size:
                    break_value+=1
                else:
                    break
            self.connection_ttl_list=self.connection_ttl_list[break_value:]
            self.connection_timestamp_list=self.connection_timestamp_list[break_value:]
            self.connection_packet_count_list=self.connection_packet_count_list[break_value:]
            self.connection_packet_length_list=self.connection_packet_length_list[break_value:]
            self.connection_flow_count-=break_value

            # stop once the characteristic lists are empty
            # while self.connection_ttl_list != []:
            #     # if a netflow feature set is determined to no longer be relevant
            #     if current_index - self.connection_ttl_list[0] >= rw_size:
            #         # removing (popping) all values associated with the netflow feature set
            #         self.connection_ttl_list.pop(0)
            #         self.connection_timestamp_list.pop(0)
            #         self.connection_packet_count_list.pop(0)
            #         self.connection_packet_length_list.pop(0)
            #         self.connection_flow_count = self.connection_flow_count - 1
            #
            #     # since the characteristic list is maintained as oldest netflow first, as soon as a valid netflow is
            #     # encountered, all remaining netflows sets are also valid.
            #     else:
            #         break
        else:
            self.connection_flow_count=0

    def validate_time_flow_ttl(self, rw_time, curent_timestamp):
        """
        This methods determines if the flow features in the characteristics list are still relevant
        given the time based rolling window size.

        The relevance is assed using the timestamp of the current netflow. This is then compared to the timestamp
        values of the netflow feature sets. If the difference between the two exceed the rolling window size,
        it is deemed to no longer be relevant.

        Then netflow feature set includes, packet count, packets total size, time stamp.

        When the feature set is deemed to no longer be relevant, all the values of associated with the set is removed
        from the characteristics list and the flow count is decremented.

        :param rw_time: The length of rolling window in minutes.
        :param curent_timestamp: The current netflow timestamp
        :return: A rw_netflow object that has been validated as having only relevant netflow feature sets

        """
        # only consider netflows_objects that actually have values
        if len(self.time_ttl_list) >= 1:

            # if the last value in the TLL list is outside the RW, the whole list is outside the RW
            if(curent_timestamp - self.time_timestamp_list[len(self.time_ttl_list)-1]).total_seconds() >= rw_time * 60:
                #print('All time flow char data is no longer relevant')
                self.time_ttl_list=[]
                self.time_timestamp_list=[]
                self.time_packet_count_list=[]
                self.time_packet_length_list=[]
                self.time_flow_count=0
                return

            # todo: if above is better than adapt here
            # stop once the characteristic lists are empty
            while self.time_ttl_list != []:
                # if a netflow feature set is determined to no longer be relevant
                if (curent_timestamp - self.time_timestamp_list[0]).total_seconds() >= rw_time * 60:
                    # removing (popping) all values associated with the netflow feature set
                    self.time_ttl_list.pop(0)
                    self.time_timestamp_list.pop(0)
                    self.time_packet_count_list.pop(0)
                    self.time_packet_length_list.pop(0)
                    self.time_flow_count -= 1

                # since the characteristic list is maintained as oldest netflow first, as soon as a valid netflow is
                # encountered, all remaining netflows sets are also valid.
                else:
                    break
        else:
            self.time_flow_count=0

    def calculate_flow_connection_based_features(self, current_index, rw_size):
        """
        This methods calculates the expanded netflow feature set given the values in the rw_netflow characteristics
        list for connection based RW.

        Calculates and returns:
            time based features (min, max, mean time between flows)
            packet count based features (min, max, mean, sum of packet counts for the flows)
            packet length based features (min, max, mean, sum of packet length for the flows)




        :param rw_size: The number of flow to be considered in the rolling window.
        :param current_index: The current netflow index
        :return: Tuple of the enhanced connection RW features
        """
        # ensures that only valid netflow features sets are used to calculate expanded feature set
        self.validate_connection_flow_ttl(rw_size, current_index)
        time_based_feat_list = min_max_mean_time_delay(self.connection_timestamp_list)
        packet_len_feat_list = min_max_mean_total_feature(self.connection_packet_length_list)
        packet_num_feat_list = min_max_mean_total_feature(self.connection_packet_count_list)
        return [self.connection_flow_count], time_based_feat_list, packet_len_feat_list, packet_num_feat_list

    def caulculate_flow_time_based_features(self, curent_timestamp, rw_time):
        """
        This methods calculates the expanded netflow feature set given the values in the rw_netflow characteristics
        list for time based RW.

        Calculates and returns:
            time based features (min, max, mean time between flows)
            packet count based features (min, max, mean, sum of packet counts for the flows)
            packet length based features (min, max, mean, sum of packet length for the flows)

        :param rw_time: The length of rolling window in minutes.
        :param curent_timestamp: The current netflow timestamp
        :return: Tuple of the enhanced time RW features
        """
        self.validate_time_flow_ttl(rw_time, curent_timestamp)
        time_based_feat_list = min_max_mean_time_delay(self.time_timestamp_list)
        packet_len_feat_list = min_max_mean_total_feature(self.time_packet_length_list)
        packet_num_feat_list = min_max_mean_total_feature(self.time_packet_count_list)
        return [self.time_flow_count], time_based_feat_list, packet_len_feat_list, packet_num_feat_list

    def print_connection_netflow(self):
        """
        This methods prints the connection based characteristics of the rw_netflow object.
        """
        print(f' Src IP/Port: {self.src_ip}/{self.src_port}')
        print(f' Dest IP/Port: {self.dest_ip}/{self.dest_port}')
        print(f' Flow count in the RW: {self.connection_flow_count}')
        print(f' The data TTL values: {self.connection_ttl_list}')
        print(f' The timestamp list: {self.connection_timestamp_list}')
        print(f' The packet count list: {self.connection_packet_count_list}')
        print(f' The packet length list: {self.connection_packet_length_list}')

    def print_time_netflow(self):
        """
        This methods prints the time based characteristics of the rw_netflow object.
        """
        print(f' Src IP/Port: {self.src_ip}/{self.src_port}')
        print(f' Dest IP/Port: {self.dest_ip}/{self.dest_port}')
        print(f' Flow count in the RW: {self.time_flow_count}')
        print(f' The data TTL values: {self.time_ttl_list}')
        print(f' The timestamp list: {self.time_timestamp_list}')
        print(f' The packet count list: {self.time_packet_count_list}')
        print(f' The packet length list: {self.time_packet_length_list}')

    def print_complete_netflow(self):
        """
        This methods prints all characteristics of the rw_netflow object.
        """
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
    connection_rw_size = 5000
    timw_rw_size_min = 0

    full_feature_netflow_csv_file = f'../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}.csv'
    #extracted_feature_csv_creator(full_feature_netflow_csv_file)

    # netflow_feature_extraction(cfg.CONSOLIDATED_NETFLOW_DATA, full_feature_netflow_csv_file, connection_rw_size,
    #                            timw_rw_size_min)
    # # test_bed(netflow_csv_file)
    csv_base_netflow_features_to_dict()

    netflow_feature_extraction(cfg.TOTAL_NETFLOW_COUNT, full_feature_netflow_csv_file, connection_rw_size,
                                                       timw_rw_size_min)

    import sys
    print (sys.getsizeof(RAW_DATA_DICT)/1000000)
    print(sys.getsizeof(NETFLOW_DICT)/1000000)
    extracted_feature_csv_creator('test.csv')
    start_time=datetime.now()
    eng_ft_dict_to_csv('test.csv',cfg.TOTAL_NETFLOW_COUNT)
    print(datetime.now()-start_time)
    # validate_time_single_flow_ft_extraction(100, 10000)
    # start_time=datetime.now()
    # for x in range(0,215000):
    #     get_single_flow_conn_features_from_raw(x,10000)
    #     if x % 10000 == 0:
    #         print(f'{x} || {datetime.now()-start_time}')


    #get_single_flow_conn_features_from_raw(211000,10000)

    #get_single_flow_conn_features_from_raw(cfg.CONSOLIDATED_NETFLOW_DATA, 500, 10000)
    # netflow_data = pd.read_csv(cfg.CONSOLIDATED_NETFLOW_DATA)
    # netflow_data = netflow_data.astype(
    #     {'Src Port': np.uint16, 'Dst Port': np.uint16, 'Timestamp': np.datetime64, 'VPN': np.uint8})
    # validate_time_single_flow_ft_extraction(100,10000)
    # netflow_data = pd.read_csv(cfg.CONSOLIDATED_NETFLOW_DATA)
    # netflow_data = netflow_data.astype(
    #     {'Src Port': np.uint16, 'Dst Port': np.uint16, 'Timestamp': np.datetime64, 'VPN': np.uint8})
    # for row in netflow_data[10:20].itertuples():
    #     print(row)
    #     print(row[0])
    #     print(row[cfg.INDEX_TUPL_NUM])


    """
    need to check out pandans rolling windows to see if they have any inbuilt methods to assit
    """


def netflow_feature_extraction(loop_upper_limit, connection_rw_size, time_rw_size):
    """
    This method calculates and extracts expanded features from basic netflow features sets.

    The methods takes a set of netflows (ordered chronologically) and process them to calculate the expanded (engineered)
    features.

    Feature are calculated in forward direction (src ip addr -> dest ip addr) as well as reverse direction
    (dest ip addr -> src ip addr).

    These features are then (along with base features) written to a global dictionary. Enhanced features are calculated
    given a specific connection rolling window size and time based rolling window size.

    :param loop_upper_limit: The number of netflows to be processed
    :param connection_rw_size: Size of connection based RW
    :param time_rw_size: Size of time based RW
    :return: dictionary containing list of base and enginerred features (fwd/bwd // conn/time) for each flow
    """
    start1 = datetime.now()
    flow_loop_time_list=[]


    with tqdm(total=loop_upper_limit) as pbar:
        for netflow_index in range(0, loop_upper_limit):
            loop_start_time=datetime.now()

            # creating the src-dst ip tags used to identify the netflows in the dictionary

            src_dest_ip_tag = RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_SRC_DEST_IP_TAG]
            # creating the reverse (dest-src) ip tags
            dest_src_ip_tag = f'{RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM]}-' \
                              f'{RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM]}'

            # If there already exist a rw_netflow object for the given src-dest tag in the RW dictionary
            if src_dest_ip_tag in NETFLOW_DICT:
                # adding the current netflow base features to the rw_netflow object
                NETFLOW_DICT[src_dest_ip_tag].add_subsequent_flow_data(RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                                                     RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM],
                                                     RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                                                     netflow_index)
            # Otherwise creating a rw_netflow object (with the netflow base features) and adding it to the RW dictionary
            else:
                NETFLOW_DICT[src_dest_ip_tag] = RW_NETFLOW(RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM],
                                                         RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_PORT_COL_NUM],
                                                         RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM],
                                                         RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_PORT_COL_NUM],
                                                         RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                                                         RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM],
                                                         RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                                                         netflow_index)

            # calculating the connection based enhanced features
            con_flow_count, con_timedelta_ft_list, con_pkt_len_ft_list, con_pkt_num_ft_list = \
                NETFLOW_DICT[src_dest_ip_tag].calculate_flow_connection_based_features(netflow_index,
                                                                                       connection_rw_size)

            # calculating the time based enhanced features
            # time_flow_count, time_timedelta_ft_list, time_pkt_len_ft_list, time_pkt_num_ft_list = \
            #     NETFLOW_DICT[src_dest_ip_tag].caulculate_flow_time_based_features(
            #         RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM], time_rw_size)



            # Determining if the reverse flow exist in the dictionary

            # Of note, reverse flow are in simply another set of flows in the dictionary (which happen to match the
            # dest-src ip tag of the given flow). Therefore, there is no need to track reverse flow base characteristics
            # (since the are already being accounted for) and thus only the enhanced feature values are extracted from
            # the reverse flow.
            # if dest_src_ip_tag in NETFLOW_DICT:
            #     # if yes, calculating the reverse flow enhanced feature sets (connection and time)
            #
            #     rev_con_flow_count, rev_con_timedelta_ft_list, rev_con_pkt_len_ft_list, rev_con_pkt_num_ft_list = \
            #         NETFLOW_DICT[dest_src_ip_tag].calculate_flow_connection_based_features(netflow_index,
            #                                                                                connection_rw_size)
            #
            #
            #     rev_time_flow_count, rev_time_timedelta_ft_list, rev_time_pkt_len_ft_list, rev_time_pkt_num_ft_list = \
            #         NETFLOW_DICT[dest_src_ip_tag].caulculate_flow_time_based_features(
            #             RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM], time_rw_size)
            #
            #
            # else:
            #     # Otherwise, if there is no reverse flow, setting all the reverse flow features to 0
            #     rev_con_flow_count = [0]
            #     rev_con_timedelta_ft_list = [0, 0, 0]
            #     rev_con_pkt_len_ft_list = [0, 0, 0, 0]
            #     rev_con_pkt_num_ft_list = [0, 0, 0, 0]
            #     rev_time_flow_count = [0]
            #     rev_time_timedelta_ft_list = [0, 0, 0]
            #     rev_time_pkt_len_ft_list = [0, 0, 0, 0]
            #     rev_time_pkt_num_ft_list = [0, 0, 0, 0]

            # 2D list containing all the enhanced features calculated
            engineered_features = [con_flow_count, con_timedelta_ft_list, con_pkt_num_ft_list, con_pkt_len_ft_list]
            # engineered_features = [con_flow_count, con_timedelta_ft_list, con_pkt_num_ft_list, con_pkt_len_ft_list,
            #                        rev_con_flow_count, rev_con_timedelta_ft_list, rev_con_pkt_num_ft_list,
            #                        rev_con_pkt_len_ft_list,
            #                        time_flow_count, time_timedelta_ft_list, time_pkt_num_ft_list, time_pkt_len_ft_list,
            #                        rev_time_flow_count, rev_time_timedelta_ft_list, rev_time_pkt_num_ft_list,
            #                        rev_time_pkt_len_ft_list]
            # list containing the base features
            base_features = [RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][0],
                             RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM],
                             RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_PORT_COL_NUM],
                             RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM],
                             RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_PORT_COL_NUM],
                             RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                             RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                             RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM]]


            # creating a dictionary entry for the base and enginerred features
            EXPANDED_FT_DIC[netflow_index]={}
            EXPANDED_FT_DIC[netflow_index]['base_features']=base_features
            EXPANDED_FT_DIC[netflow_index]['engineered_features']=engineered_features
            EXPANDED_FT_DIC[netflow_index]['VPN']=RAW_DATA_DICT[netflow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.VPN_COL_NUM]
            flow_loop_time_list.append(datetime.now()-loop_start_time)
            pbar.update(1)




    endtime = datetime.now()
    print(f' Totaltime: {endtime - start1}')
    print(min_max_mean_time_delay(flow_loop_time_list))



def get_single_flow_conn_features_from_raw(target_flow_index, conn_rw_size):
    """
    This methods calculates the engineered features for a singular netflow given a connection based rolling window.

    :param target_flow_index: The index for the netflow which the features will be calculated
    :param conn_rw_size: The size fo the connection based RW
    :return: The FWD and BWD conection based engineered features

    """

    # since connection RW size is correlated with the flow index, the RW base can be determined by subtracting the
    # RW size from the target flow index
    base_limit = target_flow_index - conn_rw_size
    # The base cannot be any smaller than 1
    if base_limit < 1: base_limit = 1

    # the src-dest ip tag of the targeted netflow
    target_src_dest_ip_tag = RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_SRC_DEST_IP_TAG]
    # the dest-src ip tag of the targeted netflow
    target_dest_src_ip_tag = f'{RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM]}-' \
                             f'{RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM]}'

    # initializing the RW_NETFLOW object for the forward direction (src->dest)
    target_flow = RW_NETFLOW(RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_PORT_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_PORT_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                             target_flow_index)
    # initializing the RW_NETFLOW object for the backward direction (dest->src)
    # it is initialized to zero since at this moment there has not been a reverse flow
    reverse_target_flow = RW_NETFLOW(RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM],
                                     RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_PORT_COL_NUM],
                                     RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM],
                                     RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_PORT_COL_NUM],
                                     0,
                                     0,
                                     0,
                                     # substracting large number such that these initiating values don't actaully count in the result
                                     target_flow_index-9999999)


    for flow_index in range(base_limit,target_flow_index):
        # checking to see if the current flow matches the target flow
        if RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_SRC_DEST_IP_TAG]==target_src_dest_ip_tag:
            # if yes, adding the flow data to the target flow RW_NETFLOW object
            target_flow.add_subsequent_flow_data(RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                                                 RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM],
                                                 RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                                                 flow_index)
        # checking to see if the current flow matches the reverse target flow
        elif RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_SRC_DEST_IP_TAG]==target_dest_src_ip_tag:
            # if yes, adding the flow data to the reverse target flow RW_NETFLOW object
            reverse_target_flow.add_subsequent_flow_data(
                RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM],
                RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                flow_index)

    # Once all flow in the RW have been processed, calculate both forward and reverse connection base features

    con_flow_count, con_timedelta_ft_list, con_pkt_len_ft_list, con_pkt_num_ft_list = \
        target_flow.calculate_flow_connection_based_features(target_flow_index, conn_rw_size)
    rev_con_flow_count, rev_con_timedelta_ft_list, rev_con_pkt_len_ft_list, rev_con_pkt_num_ft_list = \
        reverse_target_flow.calculate_flow_connection_based_features(target_flow_index, conn_rw_size)


def get_single_flow_time_features_from_raw(target_flow_index, time_rw_size):
    # todo: test method
    """
    This methods calculates the engineered features for a singular netflow given a time based rolling window.

    :param target_flow_index: The index for the netflow which the features will be calculated
    :param time_rw_size: The size fo the time based RW
    :return: The FWD and BWD time based engineered features
    """
    # the src-dest ip tag of the targeted netflow
    target_src_dest_ip_tag = RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_SRC_DEST_IP_TAG]
    # the dest-src ip tag of the targeted netflow
    target_dest_src_ip_tag = f'{RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM]}-' \
                             f'{RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM]}'
    # the timestamp of the target netflow
    base_time=RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM]

    # initializing the RW_NETFLOW object for the forward direction (src->dest)
    target_flow = RW_NETFLOW(RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_PORT_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_PORT_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM],
                             RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                             target_flow_index)
    # initializing the RW_NETFLOW object for the backward direction (dest->src)
    # it is initialized to zero since at this moment there has not been a reverse flow
    reverse_target_flow = RW_NETFLOW(RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_IP_COL_NUM],
                                     RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.DEST_PORT_COL_NUM],
                                     RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_IP_COL_NUM],
                                     RAW_DATA_DICT[target_flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.SRC_PORT_COL_NUM],
                                     0,
                                     0,
                                     0,
                                     # substracting large number such that these initiating values don't actaully count in the result
                                     target_flow_index - 9999999)

    # since the timestamp of a flow is not correlated to the flow index, need to iterate through the flows backwards
    # until a flow exceeds the time based RW.

    # flow index pointer
    flow_index=target_flow_index-1

    # checking to see if the current flow is still within the RW
    while (base_time-RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM]).total_seconds() <= time_rw_size*60:
        #if yes, checking to see if the current flow matches the target flow
        if RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_SRC_DEST_IP_TAG]==target_src_dest_ip_tag:
            # if yes, adding the flow data to the target flow RW_NETFLOW object
            target_flow.add_subsequent_flow_data(RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                                                 RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM],
                                                 RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                                                 flow_index)
        # checking to see if the current flow matches the reverse target flow
        elif RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_SRC_DEST_IP_TAG]==target_dest_src_ip_tag:
            # if yes, adding the flow data to the reverse target flow RW_NETFLOW object
            reverse_target_flow.add_subsequent_flow_data(
                RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.TIMESTAMP_COL_NUM],
                RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_LENGTH_COL_NUM],
                RAW_DATA_DICT[flow_index][cfg.DICT_FLOW_DATA_VALUES][cfg.PACKET_COUNT_COL_NUM],
                flow_index)
        flow_index-=1
    # Once all flow in the RW have been processed, calculate both forward and reverse time base features
    time_flow_count, time_timedelta_ft_list, time_pkt_len_ft_list, time_pkt_num_ft_list = \
        target_flow.caulculate_flow_time_based_features( base_time, time_rw_size)
    rev_time_flow_count, rev_time_timedelta_ft_list, rev_time_pkt_len_ft_list, rev_time_pkt_num_ft_list = \
        reverse_target_flow.caulculate_flow_time_based_features(base_time, time_rw_size)



def add_features_to_csv(csv_filename, base_features, engineered_features, vpn_status):
    """
    This methods takes a list of base features and a list of list of enhanced features and combines adds them to a csv
    file.

    :param csv_filename: The file which the features are to be added too
    :param base_features: List of base netflow features
    :param engineered_features: 2D list of engineered features
    :param vpn_status: Feature indicating if netflow is from a VPN or not
    :return: updated CSV file containing new features
    """
    # string where all features will be appended
    to_append = ''
    # adding the base features to the string
    for elements in base_features:
        to_append = f'{to_append}, {elements}'

    for list in engineered_features:
        # for each list of enhanced features, extracting the features and adding them to the string
        for list_element in list:
            to_append = f'{to_append}, {list_element}'
    to_append = f'{to_append}, {vpn_status}'

    # adding the string of features to csv file
    file = open(csv_filename, 'a', newline='', encoding='utf-8')
    with file:
        writer = csv.writer(file)
        # to_append[1:] is used since to append starts with a ','
        writer.writerow(to_append[1:].split(','))

def eng_ft_dict_to_csv(csv_filename,upper_bound):
    """
    This methods translate the dictionary containing the engineered features to a csv file.

    :param csv_filename: The file name where the eng ft will be saved
    :param upper_bound: The size of the dictionary
    :return: A csv file containing all the information from the dictionary
    """

    file = open(csv_filename, 'a', newline='', encoding='utf-8')
    with file:
        writer = csv.writer(file)

        # for each flow
        for dict_index in range(0, upper_bound):
            # getting base features from dict
            base_ft=EXPANDED_FT_DIC[dict_index]['base_features']
            # getting engineered features from dict
            eng_ft=EXPANDED_FT_DIC[dict_index]['engineered_features']
            # string where all features will be appended
            to_append = ''
            # adding the base features to the string
            for elements in base_ft:
                to_append = f'{to_append}, {elements}'
            for list in eng_ft:
                # for each list of enhanced features, extracting the features and adding them to the string
                for list_element in list:
                    to_append = f'{to_append}, {list_element}'
            # adding VPN info
            to_append = f'{to_append}, {EXPANDED_FT_DIC[dict_index]["VPN"]}'
            # adding the string of features to csv file
            # to_append[1:] is used since to append starts with a ','
            writer.writerow(to_append[1:].split(','))


def min_max_mean_total_feature(feature_list):
    """
    This method calculates the min, max, mean and sum value of a characteristic list.

    Used in conjunction with packet count and packet length base netflow features

    :param feature_list: List of netflow characteristics values
    :return: A list containing min, max, mean and sum value of characteristic list
    """
    if feature_list == []:
        return [0, 0, 0, 0]
    feature_min = min(feature_list)
    feature_max = max(feature_list)
    feature_total = sum(feature_list)
    feature_mean = feature_total / len(feature_list)

    return [feature_min, feature_max, feature_mean, feature_total]

def min_max_mean_time_delay(timestamp_list):
    # called 4 times for each flow // potential for some efficiency improvements
    """
        This method calculates the min, max and mean time deltas between multiple timestamps.

        :param timestamp_list: List of netflow timestamp values
        :return: A list containing min, max, and mean time deltas
        """
    if timestamp_list == []:
        return [0, 0, 0]

    # calculating the time deltas between timestamps in seconds
    timeflow_differential = []
    # todo: potential to eliminate for loop with map function of list comprehension
    for time_flow in range(1, len(timestamp_list)):
        time_delta = timestamp_list[time_flow] - timestamp_list[time_flow - 1]
        timeflow_differential.append(abs(time_delta.total_seconds()))
    if timeflow_differential == []:
        timeflow_differential = [0]
    # calculating min max mean values
    min_time_delta = min(timeflow_differential)
    max_time_delta = max(timeflow_differential)
    mean_time_delta = sum(timeflow_differential) / len(timeflow_differential)
    return [min_time_delta, max_time_delta, mean_time_delta]

def extracted_feature_csv_creator(csv_file_name):
    """
    This method creates a csv file to hold the expanded features.

    :param csv_file_name: The intended name for the CSV file
    :return: A csv file with the speficied header
    """
    file = open(csv_file_name, 'w', newline='', encoding='utf-8')
    with file:
        writer = csv.writer(file)
        writer.writerow(cfg.ENG_FT_HEADER)

def validate_time_single_flow_ft_extraction(number_of_repititions, rw_size):
    start_time=datetime.now()
    time_list=[]
    for x in range (0, number_of_repititions):
        iner_start=datetime.now()
        target_index= random.randint(0,215000)
        get_single_flow_conn_features_from_raw(target_index,rw_size)
        time_list.append(datetime.now()-iner_start)
    end_time=datetime.now()
    print(end_time-start_time)
    print(min_max_mean_time_delay(time_list))

def csv_base_netflow_features_to_dict():
    """
    This method transforms the base netflow data contained in a csv file into a dictionary.
    This method is only suitable to transform the cummulative netflow csv file

    The created dictionary is a dictionary of dictionary. The key of the outer dictionary is the flow index.
    The inner dictionary contains two values a) the srd-dest ip tag and b) a tuple containing all the base netflow
    features.

    The double dictionary was used since there is no unique identifier within the base netflow features for a given
    netflow.

    :return: A dictionary containing all the information from the CSV file
    """

    # reading the CSV file
    netflow_data = pd.read_csv(cfg.CONSOLIDATED_NETFLOW_DATA)
    netflow_data = netflow_data.astype(
        {'Src Port': np.uint16, 'Dst Port': np.uint16, 'Timestamp': np.datetime64, 'VPN': np.uint8})
    # for each flow in the csv file adding it to the dictionary
    for row in netflow_data.itertuples():
        # getting the src-dest ip tag
        src_dest_ip_tag=f'{row[cfg.SRC_IP_TUPL_NUM]}-{row[cfg.DEST_IP_TUPL_NUM]}'

        # creating the inner dict
        RAW_DATA_DICT[row[cfg.INDEX_TUPL_NUM]]={}
        # adding the src-dest ip tag to the inner dict
        RAW_DATA_DICT[row[cfg.INDEX_TUPL_NUM]][cfg.DICT_FLOW_SRC_DEST_IP_TAG]=src_dest_ip_tag
        # adding the base features to the inner dict
        RAW_DATA_DICT[row[cfg.INDEX_TUPL_NUM]][cfg.DICT_FLOW_DATA_VALUES] = row[1:len(row)]

def data_struct_efficiency_check():
    """
    This method was used to test various data structure access methods to determine which one was the most efficient.

    The dictionary was determined to be the most efficient
    """

    netflow_data=pd.read_csv(cfg.CONSOLIDATED_NETFLOW_DATA)
    netflow_data = netflow_data.astype(
        {'Src Port': np.uint16, 'Dst Port': np.uint16, 'Timestamp': np.datetime64, 'VPN': np.uint8})

    # time required for accessing data at random indexes (dict)
    start_time=datetime.now()
    for x in range(0,100000):
        xx=RAW_DATA_DICT[random.randint(0,215000)]['data_values'][2]

    print(datetime.now()-start_time)

    # time required for accessing data at random indexes (pandas data frame)
    start_time = datetime.now()
    for x in range(0, 100000):
        xx=netflow_data.iloc[random.randint(0,215000), cfg.SRC_PORT_COL_NUM]
    print(datetime.now() - start_time)

    # time required for iterating through a data struct (dict)
    start_time = datetime.now()
    for x in range(0,100000):
        xx = RAW_DATA_DICT[x]['data_values'][2]
    print(datetime.now() - start_time)

    # time required for iterating through a data struct (pandas data frame -iloc)
    start_time = datetime.now()
    for x in range(0, 100000):
        xx = netflow_data.iloc[x, cfg.SRC_PORT_COL_NUM]
    print(datetime.now() - start_time)

    # time required for iterating through a data struct (pandas data frame -itertuples)
    start_time = datetime.now()
    for row in netflow_data[0:100000].itertuples():
        xx=row[cfg.SRC_PORT_TUPL_NUM]
    print(datetime.now() - start_time)










if __name__ == '__main__':
    main()
