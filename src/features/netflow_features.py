import pandas as pd
import numpy as np
from datetime import datetime

def main():
    """
    Major issues is efficiency _
    :return:
    """
    flow=1000
    window_size=10000
    netflow_csv_file='../../data/processed/cummulative_netflows.csv'
    # print(forward_netflow_count_connection(flow,window_size,netflow_csv_file))
    # print(backward_netflow_count_connection(flow, window_size, netflow_csv_file))
    # print(min_max_mean_total_forwad_netflow_size_connection(flow,window_size,netflow_csv_file))
    # print(min_max_mean_total_backward_netflow_size_connection(flow,window_size,netflow_csv_file))
    # print(min_max_mean_total_forwad_netflow_packet_number_connection(flow,window_size,netflow_csv_file))
    # print(min_max_mean_total_backward_netflow_packet_number_connection(flow, window_size, netflow_csv_file))
    # print(min_max_mean_forward_netflow_time_delay_connections(flow,window_size,netflow_csv_file))
    # start_time=datetime.now()
    # for flow in range (11000,11100):
    #     connection_based_feature_extraction(flow,window_size, netflow_csv_file)
    #     print("//////////////////////////////////////////////////////")
    # end_time=datetime.now()
    # print(end_time-start_time)
    test_bed()
    test_rolling_dict={}
    data=pd.read_csv(netflow_csv_file)
    start_time=datetime.now()
    for flow in range (0,data.shape[0]):
        src_dst=f'{data.iloc[flow,1]}-{data.iloc[flow,3]}'
        dst_src=f'{data.iloc[flow,3]}-{data.iloc[flow,2]}'
        if src_dst in test_rolling_dict:
            test_rolling_dict[src_dst]["index"].append(flow)
            #print(f'{src_dst} already in list')
        else:
            test_rolling_dict[src_dst]={"index":[flow]}
            #print(f'{src_dst} is being added to the list')
    end_time = datetime.now()
    print(end_time - start_time)
    count=0
    for ip, val in test_rolling_dict.items():
        print(f'{ip} || {val}')
        count+=1
    print(count)

def connection_based_feature_extraction(base_flow, window_size, netflow_csv_file):
    """
    Major issue is efficientcy -> currently takes ~2min to process 100 entries (Given ~215k flows this means 72hrs)
    Currently the algo is O(n^2). Ideally linear. Implement dynamic programming concepts

    Ideas is to keep a dictionary with all processed src/dest ip addr pairs and relevant data points.
    Have a keep alive timer, everytime a flow matches the crit, reset the keep alive values.
    It is probable that each data point will also need a keep alive metric as well

    List 1: addr pair in the window
    List 2: attributes for the addr pair for the window size

    New flow, does the addr pair match one in list 1 (i.e in the last 10k entries has there been this pair
        ->No: add addr pair to list one & calculate the attr for the flow and add them to list 2
        ->Yes: pull the current values of list 2 for addr pair, sheed data points no longer in window, add data
                for the current flow and calculate the mean min max total

    Complications: need a way for addr pair and individual data points to time-out when they exceed the window
        for addr pair it needs to be re-setable, for data points it would simply decrement for each subsequent flow processed

    addr pair: [{sourceip, destip}, TTL]

    data point (ex len list) [[val,TTL], [val, TTL+], [val, TTL++]]
    queque might be a good data structure here
    regardless of what happens in list one, all data points in list two need to be decremented by 1 for each subsequent iteration
        also needs to happen for list 1

    list 1 needs to point to list 2

    need to check out pandans rolling windows to see if they have any inbuilt methods to assit

    Maybe for the TTL simply use the flow index, and if the difference between the current flow index and the TTL value in question
    is larger than the window size then the value has exceded TTL
    And it will always be the next in the queue that would potentially timeout (at least for list 2), so only need to check end

    When it comes to time base rolling window, if the timediff between the current timestamp and the datapoints timestamp
    then its past TTL


    """
    netflows = pd.read_csv(netflow_csv_file)
    netflows = netflows.astype(
        {'Src Port': np.uint16, 'Dst Port': np.uint16, 'Timestamp': np.datetime64, 'VPN': np.uint8})
    src_ip= netflows.iloc[base_flow,1]
    src_port=netflows.iloc[base_flow,2]
    dest_ip=netflows.iloc[base_flow, 3]
    dest_port=netflows.iloc[base_flow,4]
    forward_flow_count=0
    backward_flow_count=0
    forward_length_list=[]
    backward_length_list=[]
    forward_packet_count_list=[]
    backward_packet_count_list=[]
    forward_standard_deviation=0
    backward_standard_deviation=0
    forward_timestamps=[]
    backward_timestamps=[]


    # calculating to make the window size end point are not less than zero
    window_end_point=base_flow-window_size
    if window_end_point<0:
        window_end_point=0
    for flow in range(base_flow, window_end_point, -1):
        # forward direction
        if netflows.iloc[flow, 1] == src_ip and netflows.iloc[flow, 3] == dest_ip:
            forward_flow_count+=1
            forward_length_list.append(netflows.iloc[flow, 8] + netflows.iloc[flow, 9])
            forward_packet_count_list.append(netflows.iloc[flow, 6] + netflows.iloc[flow, 7])
            forward_timestamps.append(netflows.iloc[flow,5])

        # backward direction
        elif netflows.iloc[flow, 1] == dest_ip and netflows.iloc[flow, 3] == src_ip :
            backward_flow_count+=1
            backward_length_list.append(netflows.iloc[flow, 8] + netflows.iloc[flow, 9])
            backward_packet_count_list.append(netflows.iloc[flow, 6] + netflows.iloc[flow, 7])
            backward_timestamps.append(netflows.iloc[flow,5])
    fwd_size_min, fwd_size_max, fwd_size_mean, fwd_size_tot = min_max_mean_total_feature(forward_length_list)
    fwd_pkt_min, fwd_pkt_max, fwd_pkt_mean, fwd_pkt_tot=min_max_mean_total_feature(forward_packet_count_list)
    fwd_time_min,fwd_time_max, fwd_time_mean=min_max_mean_time_delay(forward_timestamps)
    bwd_size_min, bwd_size_max, bwd_size_mean, bwd_size_tot = min_max_mean_total_feature(forward_length_list)
    bwd_pkt_min, bwd_pkt_max, bwd_pkt_mean, bwd_pkt_tot = min_max_mean_total_feature(forward_packet_count_list)
    bwd_time_min, bwd_time_max, bwd_time_mean = min_max_mean_time_delay(forward_timestamps)
    print(forward_flow_count)
    print(min_max_mean_total_feature(forward_length_list))
    print(min_max_mean_total_feature(forward_packet_count_list))
    print(min_max_mean_time_delay(forward_timestamps))
    print('===============================================')
    print(backward_flow_count)
    print(min_max_mean_total_feature(backward_length_list))
    print(min_max_mean_total_feature(backward_packet_count_list))
    print(min_max_mean_time_delay(backward_timestamps))

def test_bed():
    test_list_1=[["1.1.1.1", "2.2.2.2"],["3.3.3.3","4.4.4.4"],["5.5.5.5","6.6.6.6"]]
    test_dict_2={"1.1.1.1-2.2.2.2":{
        "flow_TTL": 10000,
        "count": 45,
        "size": [13434, 34423],
        "packet": [334, 663],
        "time_stamp": ['2015-01-04 10:02',"2015-01-04 10:03"],
        "data_ttl":[44, 55]
    }}
    test_dict_2["1.1.1.1-2.2.2.2"]["time_stamp"].append(["2015-01-04 10:04", 66])
    # print(type(test1))
    # if test1[0][1]<50:
    #     test1.pop(0)
    # print(test1)
    print(test_dict_2["1.1.1.1-2.2.2.2"]["flow_TTL"])
    test_dict_2["1.1.1.1-2.2.2.2"]["flow_TTL"]=14000
    print(test_dict_2["1.1.1.1-2.2.2.2"]["flow_TTL"])
    print(test_dict_2["1.1.1.1-2.2.2.2"])
    print("1.1.1.1-2.2.2.2" in test_dict_2)
    print("1.1.1.1-2.2.2.3" in test_dict_2)



def min_max_mean_total_feature(feature_list):
    if feature_list == []:
        feature_list=[0]
    feature_min=min(feature_list)
    feature_max=max(feature_list)
    feature_total=sum(feature_list)
    feature_mean=feature_total/len(feature_list)

    return feature_min, feature_max, feature_mean, feature_total


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
    return min_time_delta, max_time_delta, mean_time_delta







if __name__ == '__main__':
    main()

