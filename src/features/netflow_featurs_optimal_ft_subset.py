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
import pandas as pd
import numpy as np
from datetime import datetime
import src.config as cfg

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
        self.dest_ip = dest_ip
        self.connection_packet_count_list = [packet_count_list]
        self.connection_packet_length_list = [packet_length_list]
        self.connection_timestamp_list = [pd.Timestamp(timestamp_list)]
        self.connection_ttl_list = [ttl_list]
        self.connection_flow_count = 1
        # self.time_packet_count_list = [packet_count_list]
        # self.time_packet_length_list = [packet_length_list]
        # self.time_timestamp_list = [pd.Timestamp(timestamp_list)]
        # self.time_ttl_list = [ttl_list]
        # self.time_flow_count = 1

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
        # self.time_packet_length_list.append(new_packet_length)
        # self.time_packet_count_list.append(new_packet_count)
        # self.time_ttl_list.append(new_ttl_index)
        # self.time_timestamp_list.append(pd.Timestamp(new_timestamp))
        # self.time_flow_count += 1

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
            # stop once the characteristic lists are empty
            while self.connection_ttl_list != []:
                # if a netflow feature set is determined to no longer be relevant
                if current_index - self.connection_ttl_list[0] >= rw_size:
                    # removing (popping) all values associated with the netflow feature set
                    self.connection_ttl_list.pop(0)
                    self.connection_timestamp_list.pop(0)
                    self.connection_packet_count_list.pop(0)
                    self.connection_packet_length_list.pop(0)
                    self.connection_flow_count = self.connection_flow_count - 1

                # since the characteristic list is maintained as oldest netflow first, as soon as a valid netflow is
                # encountered, all remaining netflows sets are also valid.
                else:
                    break

    # def validate_time_flow_ttl(self, rw_time, curent_timestamp):
    #     """
    #     This methods determines if the flow features in the characteristics list are still relevant
    #     given the time based rolling window size.
    #
    #     The relevance is assed using the timestamp of the current netflow. This is then compared to the timestamp
    #     values of the netflow feature sets. If the difference between the two exceed the rolling window size,
    #     it is deemed to no longer be relevant.
    #
    #     Then netflow feature set includes, packet count, packets total size, time stamp.
    #
    #     When the feature set is deemed to no longer be relevant, all the values of associated with the set is removed
    #     from the characteristics list and the flow count is decremented.
    #
    #     :param rw_time: The length of rolling window in minutes.
    #     :param curent_timestamp: The current netflow timestamp
    #     :return: A rw_netflow object that has been validated as having only relevant netflow feature sets
    #
    #     """
    #     # only consider netflows_objects that actually have values
    #     if len(self.time_ttl_list) > 1:
    #
    #         # stop once the characteristic lists are empty
    #         while self.time_ttl_list != []:
    #             # if a netflow feature set is determined to no longer be relevant
    #             if (curent_timestamp - self.time_timestamp_list[0]).total_seconds() >= rw_time * 60:
    #                 # removing (popping) all values associated with the netflow feature set
    #                 self.time_ttl_list.pop(0)
    #                 self.time_timestamp_list.pop(0)
    #                 self.time_packet_count_list.pop(0)
    #                 self.time_packet_length_list.pop(0)
    #                 self.time_flow_count -= 1
    #
    #             # since the characteristic list is maintained as oldest netflow first, as soon as a valid netflow is
    #             # encountered, all remaining netflows sets are also valid.
    #             else:
    #                 break

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

    # def caulculate_flow_time_based_features(self, curent_timestamp, rw_time):
    #     """
    #     This methods calculates the expanded netflow feature set given the values in the rw_netflow characteristics
    #     list for time based RW.
    #
    #     Calculates and returns:
    #         time based features (min, max, mean time between flows)
    #         packet count based features (min, max, mean, sum of packet counts for the flows)
    #         packet length based features (min, max, mean, sum of packet length for the flows)
    #
    #     :param rw_time: The length of rolling window in minutes.
    #     :param curent_timestamp: The current netflow timestamp
    #     :return: Tuple of the enhanced time RW features
    #     """
    #     self.validate_time_flow_ttl(rw_time, curent_timestamp)
    #     time_based_feat_list = min_max_mean_time_delay(self.time_timestamp_list)
    #     packet_len_feat_list = min_max_mean_total_feature(self.time_packet_length_list)
    #     packet_num_feat_list = min_max_mean_total_feature(self.time_packet_count_list)
    #     return [self.time_flow_count], time_based_feat_list, packet_len_feat_list, packet_num_feat_list

    def print_connection_netflow(self):
        """
        This methods prints the connection based characteristics of the rw_netflow object.
        """
        print('\n')
        print('RW_NETFLOW Connection Attributes')
        # print('\n')
        print(f' Src IP: {self.src_ip}')
        print(f' Dest IP: {self.dest_ip}')
        print(f' Flow count in the RW: {self.connection_flow_count}')
        print(f' The data TTL values: {self.connection_ttl_list}')
        print(f' The timestamp list: {self.connection_timestamp_list}')
        print(f' The packet count list: {self.connection_packet_count_list}')
        print(f' The packet length list: {self.connection_packet_length_list}')
        print('\n')

    # def print_time_netflow(self):
    #     """
    #     This methods prints the time based characteristics of the rw_netflow object.
    #     """
    #     print(f' Src IP: {self.src_ip}')
    #     print(f' Dest IP: {self.dest_ip}')
    #     print(f' Flow count in the RW: {self.time_flow_count}')
    #     print(f' The data TTL values: {self.time_ttl_list}')
    #     print(f' The timestamp list: {self.time_timestamp_list}')
    #     print(f' The packet count list: {self.time_packet_count_list}')
    #     print(f' The packet length list: {self.time_packet_length_list}')
    #
    # def print_complete_netflow(self):
    #     """
    #     This methods prints all characteristics of the rw_netflow object.
    #     """
    #     print(f' Src IP: {self.src_ip}')
    #     print(f' Dest IP: {self.dest_ip}')
    #     print(f' Flow count in the RW: [Con]>> {self.connection_flow_count} '
    #           f'|| [Time]>> {self.time_flow_count}')
    #     print(f' The data TTL values: [Con]>> {self.connection_ttl_list} '
    #           f'|| [Time]>> {self.time_ttl_list}')
    #     print(f' The timestamp list: [Con]>> {self.connection_timestamp_list} '
    #           f'|| [Time]>> {self.time_timestamp_list}')
    #     print(f' The packet count list: [Con]>> {self.connection_packet_count_list} '
    #           f'|| [Time]>> {self.time_packet_count_list}')
    #     print(f' The packet length list: [Con]>> {self.connection_packet_length_list} '
    #           f'|| [Time]>> {self.time_packet_length_list}')


def main():
    start_time=datetime.now()
    connection_rw_size = 2000
    timw_rw_size_min = 0

    full_feature_netflow_csv_file = f'../../data/processed/stream_line_test_2.csv'
    extracted_feature_csv_creator(full_feature_netflow_csv_file)

    netflow_feature_extraction(cfg.CONSOLIDATED_NETFLOW_DATA, full_feature_netflow_csv_file, connection_rw_size,
                               timw_rw_size_min)
    # test_bed(netflow_csv_file)
    end_time=datetime.now()
    print(end_time-start_time)




def netflow_feature_extraction(netflow_csv_file, target_csv_file, connection_rw_size, time_rw_size):
    """
    This method calculates and extracts expanded features from basic netflow features sets.

    The methods takes a set of netflows (ordered chronologically) and process them to calculate the expanded (engineered)
    features.

    Feature are calculated in forward direction (src ip addr -> dest ip addr) as well as reverse direction
    (dest ip addr -> src ip addr).

    These features are then (along with base features) written to a new file. Enhanced features are calculated
    given a specific connection rolling window size and time based rolling window size.

    :param netflow_csv_file: File which contains base netflow feature set
    :param target_csv_file: File to which enhanced features set will be written too
    :param connection_rw_size: Size of connection based RW
    :param time_rw_size: Size of time based RW
    :return:
    """
    start1 = datetime.now()

    # the list of active (i.e. relevant netflows) is maintained through the use of a dictionary.
    netflow_dictionary = {}
    netflow_data = pd.read_csv('../'+netflow_csv_file)
    # given the large size of the feature sets, when reading the base netflow list, specifiying the data types to
    # increase memory efficiency.
    netflow_data = netflow_data.astype(
        {'Src Port': np.uint16, 'Dst Port': np.uint16, 'Timestamp': np.datetime64, 'VPN': np.uint8})
    for netflow_index in range(1, netflow_data.shape[0]):
        if netflow_index == 10000:
            start2 = datetime.now()
        if netflow_index == 20000:
            start3 = datetime.now()
        if netflow_index == 30000:
            start4 = datetime.now()
        if netflow_index == 40000:
            start5 = datetime.now()

        if netflow_index % 1000 == 0:
            print(f'{netflow_index} of {netflow_data.shape[0]}')

        # creating the src-dst ip tags used to identify the netflows in the dictionary
        src_dest_ip_tag = f'{netflow_data.iloc[netflow_index, cfg.SRC_IP_COL_NUM]}-{netflow_data.iloc[netflow_index, cfg.DEST_IP_COL_NUM]}'
        # creating the reverse (dest-src) ip tags
        # dest_src_ip_tag = f'{netflow_data.iloc[netflow_index, cfg.DEST_IP_COL_NUM]}-{netflow_data.iloc[netflow_index, cfg.SRC_IP_COL_NUM]}'

        # If there already exist a rw_netflow object for the given src-dest tag in the RW dictionary
        if src_dest_ip_tag in netflow_dictionary:
            # adding the current netflow base features to the rw_netflow object
            netflow_dictionary[src_dest_ip_tag].add_subsequent_flow_data(
                netflow_data.iloc[netflow_index, cfg.TIMESTAMP_COL_NUM],
                netflow_data.iloc[
                    netflow_index, cfg.PACKET_LENGTH_COL_NUM],
                netflow_data.iloc[
                    netflow_index, cfg.PACKET_COUNT_COL_NUM],
                netflow_index)
        # Otherwise creating a rw_netflow object (with the netflow base features) and adding it to the RW dictionary
        else:
            netflow_dictionary[src_dest_ip_tag] = RW_NETFLOW(netflow_data.iloc[netflow_index, cfg.SRC_IP_COL_NUM],
                                                             netflow_data.iloc[netflow_index, cfg.SRC_PORT_COL_NUM],
                                                             netflow_data.iloc[netflow_index, cfg.DEST_IP_COL_NUM],
                                                             netflow_data.iloc[netflow_index, cfg.DEST_PORT_COL_NUM],
                                                             netflow_data.iloc[netflow_index, cfg.PACKET_COUNT_COL_NUM],
                                                             netflow_data.iloc[netflow_index, cfg.PACKET_LENGTH_COL_NUM],
                                                             netflow_data.iloc[
                                                                 netflow_index, cfg.TIMESTAMP_COL_NUM],
                                                             netflow_index)

        # calculating the connection based enhanced features
        con_flow_count, con_timedelta_ft_list, con_pkt_len_ft_list, con_pkt_num_ft_list = \
            netflow_dictionary[src_dest_ip_tag].calculate_flow_connection_based_features(netflow_index,
                                                                                         connection_rw_size)


        # calculating the time based enhanced features
        # time_flow_count, time_timedelta_ft_list, time_pkt_len_ft_list, time_pkt_num_ft_list = \
        #     netflow_dictionary[src_dest_ip_tag].caulculate_flow_time_based_features(
        #         netflow_data.iloc[netflow_index, cfg.TIMESTAMP_COL_NUM], time_rw_size)

        # Determining if the reverse flow exist in the dictionary

        # Of note, reverse flow are in simply another set of flows in the dictionary (which happen to match the
        # dest-src ip tag of the given flow). Therefore, there is no need to track reverse flow base characteristics
        # (since the are already being accounted for) and thus only the enhanced feature values are extracted from
        # the reverse flow.
        # if dest_src_ip_tag in netflow_dictionary:
        #     # if yes, calculating the reverse flow enhanced feature sets (connection and time)
        #     rev_con_flow_count, rev_con_timedelta_ft_list, rev_con_pkt_len_ft_list, rev_con_pkt_num_ft_list = \
        #         netflow_dictionary[dest_src_ip_tag].calculate_flow_connection_based_features(netflow_index,
        #                                                                                      connection_rw_size)
        #     rev_time_flow_count, rev_time_timedelta_ft_list, rev_time_pkt_len_ft_list, rev_time_pkt_num_ft_list = \
        #         netflow_dictionary[dest_src_ip_tag].caulculate_flow_time_based_features(
        #             netflow_data.iloc[netflow_index, cfg.TIMESTAMP_COL_NUM], time_rw_size)
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

        #
        # list containing the base features
        base_features = [netflow_data.iloc[netflow_index, 0],
                         netflow_data.iloc[netflow_index, cfg.SRC_IP_COL_NUM],
                         netflow_data.iloc[netflow_index, cfg.SRC_PORT_COL_NUM],
                         netflow_data.iloc[netflow_index, cfg.DEST_IP_COL_NUM],
                         netflow_data.iloc[netflow_index, cfg.DEST_PORT_COL_NUM],
                         netflow_data.iloc[netflow_index, cfg.TIMESTAMP_COL_NUM],
                         netflow_data.iloc[netflow_index, cfg.PACKET_COUNT_COL_NUM],
                         netflow_data.iloc[netflow_index, cfg.PACKET_LENGTH_COL_NUM]]

        # adding the base features, engineered features and VPN identifier to the enhanced features file.
        add_features_to_csv(target_csv_file, base_features, engineered_features,
                            netflow_data.iloc[netflow_index, cfg.VPN_COL_NUM])

    endtime = datetime.now()
    # print(f' Totaltime: {endtime - start1}')
    # print(f' First 10000: {start2 - start1}')
    # print(f'  10k- 20k: {start3 - start2}')
    # print(f' 20k -30k: {start4 - start3}')
    # print(f' 30k -40k: {start5 - start4}')
    # print(f' 40k -50k: {endtime - start5}')
    # print(f' 10k -30k: {endtime - start2}')



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


def min_max_mean_total_feature(feature_list):
    """
    This method calculates the min, max, mean and sum value of a characteristic list.

    Used in conjunction with packet count and packet length base netflow features

    :param feature_list: List of netflow characteristics values
    :return: A list containing min, max, mean and sum value of characteristic list
    """
    if feature_list == []:
        feature_list = [0]
    feature_min = min(feature_list)
    feature_max = max(feature_list)
    feature_total = sum(feature_list)
    feature_mean = feature_total / len(feature_list)

    return [feature_min, feature_max, feature_mean, feature_total]


def min_max_mean_time_delay(timestamp_list):
    """
        This method calculates the min, max and mean time deltas between multiple timestamps.

        :param timestamp_list: List of netflow timestamp values
        :return: A list containing min, max, and mean time deltas
        """
    if timestamp_list == []:
        timestamp_list = [0]

    # calculating the time deltas between timestamps in seconds
    timeflow_differential = []
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


if __name__ == '__main__':
    main()
