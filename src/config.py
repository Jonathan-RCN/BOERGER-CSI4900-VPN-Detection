CONNECTION_RW_SIZE=10000
TIME_RW_SIZE=10

SRC_IP_COL = 1
SRC_PORT_COL = 2
DEST_IP_COL = 3
DEST_PORT_COL = 4
TIMESTAMP_COL = 5
PACKET_COUNT_COL = 6
PACKET_LENGTH_COL = 7
VPN_COL=8

CONSOLIDATED_NETFLOW_DATA='../data/processed/cummulative_netflows.csv'
RAW_NETFLOW_DIR='../../data/raw/net_traffic_csv/'
FILTERED_NETFLOW_DIR = '../../data/interim/filtered/'


""" The following elements of the raw netflow are to be transferred to the filtered data files. 
    Although the generated netflows have significantly more elements, these elements were selected since
    they are what an ISP would have access to in a netflow. 
    """

HEADER=['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']