"""
Title: Data Prepossessing

Project: CSI4900 Honours Project

Created: Jan 2020
Last modified: 06 May 2020

Author: Jonathan Boerger
Status: Complete

Description: This module transforms raw data into a consolidated CSV file for further manipulation

"""

import csv
import pandas as pd
import numpy as np
import os
from datetime import datetime


def main():
    process_raw_data_set()


def process_raw_data_set():
    """
    This method transform the raw data into a consolidated file which can be used for feature engineering.

    The method takes the raw netflow files (netflow representation of pcap files), filters them such that only relevant
    data is keep and combines all relevant data into a singular csv file

    :return: A singular CSV file containing all the filtered data which represent the base features of the data set.
    """
    main_start_time = datetime.now()

    raw_netflow_div = '../../data/raw/net_traffic_csv/'
    filtered_netflow_csv_dir = '../../data/interim/filtered/'
    cummlative_dir = '../../data/processed/'
    cummulative_netflow_csv = "cummulative_netflows.csv"

    raw_netflow_list = []
    for file in os.listdir(raw_netflow_div):
        raw_netflow_list.append(file)

    filtered_netflows_list = []
    for file in os.listdir(filtered_netflow_csv_dir):
        filtered_netflows_list.append(filtered_netflow_csv_dir + file)

    """ The following elements of the raw netflow are to be transferred to the filtered data files. 
    Although the generated netflows have significantly more elements, these elements were selected since
    they are what an ISP would have access to in a netflow. 
    """
    header = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']

    for raw_file in raw_netflow_list:
        filtered_filename = raw_file.replace('.csv', '_filtered.csv')

        # if there is no filtered version of the raw neflow
        if filtered_netflow_csv_dir + filtered_filename not in filtered_netflows_list:

            # create the csv file and transfer over the relevant data
            create_filtered_csv(filtered_filename, filtered_netflow_csv_dir)
            filter_raw_netflow_data(raw_netflow_div + raw_file, filtered_netflow_csv_dir, filtered_filename, header)

            filtered_netflows_list.append(filtered_netflow_csv_dir + filtered_filename)
        else:
            # otherwise if the file exist, verify that there is an equal number of rows between filtered and raw data
            raw_data = pd.read_csv(raw_netflow_div + raw_file)
            filtered_data = pd.read_csv(filtered_netflow_csv_dir + filtered_filename)

            # in the event the two files do no have the same amount of data, simply recreate the filtered file
            if raw_data.shape[0] != filtered_data.shape[0]:
                # print(f'Part of {raw_file} is missing')
                create_filtered_csv(filtered_filename, filtered_netflow_csv_dir)
                filter_raw_netflow_data(raw_netflow_div + raw_file, filtered_netflow_csv_dir, filtered_filename, header)

    print(datetime.now() - main_start_time)
    # create and populate the single consolidated csv file
    create_filtered_csv(cummulative_netflow_csv, cummlative_dir)
    merge_filtered_netflows(filtered_netflows_list, cummlative_dir + cummulative_netflow_csv)
    print(datetime.now() - main_start_time)


def create_filtered_csv(dest_file_name, dest_dir):
    """
    Creates a csv file for the filtered data

    :param dest_file_name: Directory where the file will be located
    :param dest_dir: The name of the file to be created
    :return: The csv file with containing only the header
    """
    file = open(dest_dir + dest_file_name, 'w', newline='', encoding='utf-8')
    with file:
        header = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
        writer = csv.writer(file)
        writer.writerow(header)


def filter_raw_netflow_data(src_file, tgt_dir, tgt_filename, header_list):
    """
    This methods takes a raw netflow and filters out superfluous information to created a filtered version of the
    raw data set.

    This method is a significantly more efficient iteration over teh previous version (transfer_netflow_data).
    Efficiency improvements were implemented through the elimination of loops and the introduction of
    numpy vectorization.
    These improvements were adapted from:
    https://towardsdatascience.com/how-to-make-your-pandas-loop-71-803-times-faster-805030df4f06


    :param src_file: Raw data filename
    :param tgt_dir: Directory of the filtered data
    :param tgt_filename: Filtered data filename
    :param header_list: List of header elements to be copied over from the raw file to the filtered file
    :return: A csv file containing filtered data
    """
    src_data = pd.read_csv(src_file, encoding='utf-8')
    target_data = pd.read_csv(tgt_dir + tgt_filename, encoding='utf-8')

    for header in header_list:

        # Since raw data is split between forward and backward packet count and total length for the flows.
        # Using numpy vectors to sum the forward and backwards values to create a singular packet count/ packet byte
        # count (length) value to be inputted into the filtered data.
        if header == 'Tot Pkts':
            target_data[header] = np.add(src_data['Tot Fwd Pkts'].values, src_data['Tot Bwd Pkts'].values)
        elif header == 'TotLen':
            target_data[header] = np.add(src_data['TotLen Fwd Pkts'].values, src_data['TotLen Bwd Pkts'].values)

        # Adding a column to identify if the flow is from a VPN or not
        elif header == 'VPN':
            if tgt_filename.startswith("vpn"):
                target_data['VPN'] = np.ones(src_data.shape[0], dtype=np.uint8)
            else:
                target_data['VPN'] = np.zeros(src_data.shape[0], dtype=np.uint8)

        # Otherwise simply copying the data over
        else:
            target_data[header] = src_data[header]

    target_data.to_csv(tgt_dir + tgt_filename, index=False, encoding="utf-8")


def merge_filtered_netflows(list_of_files, consolidated_csv_filename):
    """
    This method takes a list of csv files (all the same format) and combines them into a consolidated csv file.
    Furthermost, this consolidated file is then sorted chronologically.

    :param list_of_files: List of filtered netflow CSV files
    :param consolidated_csv_filename:  Filename of the consolidated CSV file
    :return: Consolidated CSV file which contains all the data from each individual CSV file.
    """
    # Consolidate all CSV files into one object
    consolidated_netflows = pd.concat([pd.read_csv(file) for file in list_of_files])

    # Sorting consolidated csv in chronological order based on timestamp
    consolidated_netflows = consolidated_netflows.sort_values(by='Timestamp')
    consolidated_netflows.to_csv(consolidated_csv_filename, index=False, encoding="utf-8")


def transfer_netflow_data(src_data, columns_to_keep, file_name, dest_dir_filename, header):
    """
    Depricated due to inneficiency, functionaly replaced by filter_raw_netflow

    :param src_data:
    :param columns_to_keep:
    :param file_name:
    :param dest_dir_filename:
    :param header:
    :return:
    """
    filtered_data = pd.read_csv(dest_dir_filename)
    time_check = datetime.now()
    start_time = datetime.now()
    cummulative_append = []
    for x in range(0, src_data.shape[0]):
        if (x % 1000) == 0:
            print(
                f'{x} of {src_data.shape[0]} || {datetime.now()}|| Time since last TC: {datetime.now() - time_check}|| Time since start: {datetime.now() - start_time}')
            time_check = datetime.now()
        row_to_append = []

        for y in range(0, len(header)):
            if header[y] == "VPN":
                if file_name.startswith("vpn"):
                    row_to_append.append(np.uint8(1))
                else:
                    row_to_append.append(np.uint8(0))

            elif header[y] == "Timestamp":
                time_data = src_data.loc[x, columns_to_keep[y]]
                time_data_2 = pd.Timestamp(time_data)
                row_to_append.append(time_data_2)
            elif header[y] == 'Src Port' or header[y] == 'Dst Port':
                port_data = src_data.loc[x, columns_to_keep[y]]
                row_to_append.append(np.uint16(port_data))
            elif header[y] == 'Tot Pkts':
                pck_count = src_data.loc[x, "Tot Fwd Pkts"] + src_data.loc[x, "Tot Bwd Pkts"]
                row_to_append.append(pck_count)
            elif header[y] == 'TotLen':
                tot_len = src_data.loc[x, "TotLen Fwd Pkts"] + src_data.loc[x, "TotLen Bwd Pkts"]
                row_to_append.append(tot_len)

            else:
                row_to_append.append(src_data.loc[x, columns_to_keep[y]])

        filtered_data.loc[x] = row_to_append
        cummulative_append.append(row_to_append)

    filtered_data.to_csv(dest_dir_filename, index=False)


if __name__ == '__main__':
    main()
