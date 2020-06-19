CONNECTION_RW_SIZE = 10000
TIME_RW_SIZE = 10

SRC_IP_COL_NUM = 1
SRC_PORT_COL_NUM = 2
DEST_IP_COL_NUM = 3
DEST_PORT_COL_NUM = 4
TIMESTAMP_COL_NUM = 5
PACKET_COUNT_COL_NUM = 6
PACKET_LENGTH_COL_NUM = 7
VPN_COL_NUM = 8

INDEX_TUPL_NUM = 0
SRC_IP_TUPL_NUM = 2
SRC_PORT_TUPL_NUM = 3
DEST_IP_TUPL_NUM = 4
DEST_PORT_TUPL_NUM = 5
TIMESTAMP_TUPL_NUM = 6
PACKET_COUNT_TUPL_NUM = 7
PACKET_LENGTH_TUPL_NUM = 8
VPN_TUPL_NUM = 9

SRC_IP = 'Src IP'
SRC_PORT = 'Src Port'
DEST_IP = 'Dst IP'
DEST_PORT = 'Dst Port'
TIMESTAMP = 'Timestamp'
PACKET_COUNT = 'Tot Pkts'
PACKET_LENGTH = 'TotLen'
VPN_COL = 'VPN'

DICT_FLOW_SRC_DEST_IP_TAG = "src_dest_ip_tag"
DICT_FLOW_DATA_VALUES = 'data_values'

CONSOLIDATED_NETFLOW_DATA = '../../data/processed/cummulative_netflows.csv'
RAW_NETFLOW_DIR = '../../data/raw/net_traffic_csv/'
FILTERED_NETFLOW_DIR = '../../data/interim/filtered/'

""" The following elements of the raw netflow are to be transferred to the filtered data files. 
    Although the generated netflows have significantly more elements, these elements were selected since
    they are what an ISP would have access to in a netflow. 
    """

HEADER = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN']
MODIFIED_HEADER = ['Flow_ID', 'Src_IP', 'Src_Port', 'Dst_IP', 'Dst_Port', 'Timestamp', 'Tot_Pkts', 'TotLen', 'VPN']

ENG_FT_HEADER = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Tot Pkts', 'TotLen',
                 'Con Flow count', 'Conn Timedelta Min', 'Conn Timedelta Max', 'Conn Timedelta Mean',
                 'Conn Pkt Num Min', 'Conn Pkt Num Max', 'Conn Pkt Num Mean', 'Conn Pkt Num Tot',
                 'Conn Pkt Len Min', 'Conn Pkt Len Max', 'Conn Pkt Len Mean', 'Conn Pkt Len Tot',
                 'Conn-Rev Flow count', 'Conn-Rev Timedelta Min', 'Conn-Rev Timedelta Max', 'Conn-Rev Timedelta Mean',
                 'Conn-Rev Pkt Num Min', 'Conn-Rev Pkt Num Max', 'Conn-Rev Pkt Num Mean', 'Conn-Rev Pkt Num Tot',
                 'Conn-Rev Pkt Len Min', 'Conn-Rev Pkt Len Max', 'Conn-Rev Pkt Len Mean', 'Conn-Rev Pkt Len Tot',
                 'Time Flow count', 'Time Timedelta Min', 'Time Timedelta Max', 'Time Timedelta Mean',
                 'Time Pkt Num Min', 'Time Pkt Num Max', 'Time Pkt Num Mean', 'Time Pkt Num Tot',
                 'Time Pkt Len Min', 'Time Pkt Len Max', 'Time Pkt Len Mean', 'Time Pkt Len Tot',
                 'Time-Rev Flow count',
                 'Time-Rev Timedelta Min', 'Time-Rev Timedelta Max', 'Time-Rev Timedelta Mean',
                 'Time-Rev Pkt Num Min', 'Time-Rev Pkt Num Max', 'Time-Rev Pkt Num Mean', 'Time-Rev Pkt Num Tot',
                 'Time-Rev Pkt Len Min', 'Time-Rev Pkt Len Max', 'Time-Rev Pkt Len Mean', 'Time-Rev Pkt Len Tot', 'VPN'
                 ]

TOTAL_NETFLOW_COUNT = 215590

# The following are the isolation arrays used to isolate particular features subsets. These are drop tables, therefore they
# will be removed to leave place for the desire features subset.

# BASELINE :: Original features // no ft eng
# CONN :: Connection based features
# TIME :: Time based features
# COMB :: Both time and connection features
# FWD :: Only features calculated in the forward direction
# BWD :: Only features calculated in the reverse direction


BASELINE_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'VPN', 'Time Flow count', 'Time Timedelta Min',
                  'Time Timedelta Max', 'Time Timedelta Mean', 'Time Pkt Num Min', 'Time Pkt Num Max',
                  'Time Pkt Num Mean', 'Time Pkt Num Tot', 'Time Pkt Len Min', 'Time Pkt Len Max', 'Time Pkt Len Mean',
                  'Time Pkt Len Tot', 'Time-Rev Flow count', 'Time-Rev Timedelta Min', 'Time-Rev Timedelta Max',
                  'Time-Rev Timedelta Mean', 'Time-Rev Pkt Num Min', 'Time-Rev Pkt Num Max', 'Time-Rev Pkt Num Mean',
                  'Time-Rev Pkt Num Tot', 'Time-Rev Pkt Len Min', 'Time-Rev Pkt Len Max', 'Time-Rev Pkt Len Mean',
                  'Time-Rev Pkt Len Tot', 'Con Flow count', 'Conn Timedelta Min', 'Conn Timedelta Max',
                  'Conn Timedelta Mean', 'Conn Pkt Num Min', 'Conn Pkt Num Max', 'Conn Pkt Num Mean',
                  'Conn Pkt Num Tot', 'Conn Pkt Len Min', 'Conn Pkt Len Max', 'Conn Pkt Len Mean', 'Conn Pkt Len Tot',
                  'Conn-Rev Flow count', 'Conn-Rev Timedelta Min', 'Conn-Rev Timedelta Max', 'Conn-Rev Timedelta Mean',
                  'Conn-Rev Pkt Num Min', 'Conn-Rev Pkt Num Max', 'Conn-Rev Pkt Num Mean', 'Conn-Rev Pkt Num Tot',
                  'Conn-Rev Pkt Len Min', 'Conn-Rev Pkt Len Max', 'Conn-Rev Pkt Len Mean', 'Conn-Rev Pkt Len Tot']

CONN_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN', 'Time Flow count',
              'Time Timedelta Min', 'Time Timedelta Max', 'Time Timedelta Mean', 'Time Pkt Num Min', 'Time Pkt Num Max',
              'Time Pkt Num Mean', 'Time Pkt Num Tot', 'Time Pkt Len Min', 'Time Pkt Len Max', 'Time Pkt Len Mean',
              'Time Pkt Len Tot', 'Time-Rev Flow count', 'Time-Rev Timedelta Min', 'Time-Rev Timedelta Max',
              'Time-Rev Timedelta Mean', 'Time-Rev Pkt Num Min', 'Time-Rev Pkt Num Max', 'Time-Rev Pkt Num Mean',
              'Time-Rev Pkt Num Tot', 'Time-Rev Pkt Len Min', 'Time-Rev Pkt Len Max', 'Time-Rev Pkt Len Mean',
              'Time-Rev Pkt Len Tot']

CONN_FWD_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN', 'Time Flow count',
                  'Time Timedelta Min', 'Time Timedelta Max', 'Time Timedelta Mean', 'Time Pkt Num Min',
                  'Time Pkt Num Max', 'Time Pkt Num Mean', 'Time Pkt Num Tot', 'Time Pkt Len Min', 'Time Pkt Len Max',
                  'Time Pkt Len Mean', 'Time Pkt Len Tot', 'Time-Rev Flow count', 'Time-Rev Timedelta Min',
                  'Time-Rev Timedelta Max', 'Time-Rev Timedelta Mean', 'Time-Rev Pkt Num Min', 'Time-Rev Pkt Num Max',
                  'Time-Rev Pkt Num Mean', 'Time-Rev Pkt Num Tot', 'Time-Rev Pkt Len Min', 'Time-Rev Pkt Len Max',
                  'Time-Rev Pkt Len Mean', 'Time-Rev Pkt Len Tot', 'Conn-Rev Flow count', 'Conn-Rev Timedelta Min',
                  'Conn-Rev Timedelta Max', 'Conn-Rev Timedelta Mean', 'Conn-Rev Pkt Num Min', 'Conn-Rev Pkt Num Max',
                  'Conn-Rev Pkt Num Mean', 'Conn-Rev Pkt Num Tot', 'Conn-Rev Pkt Len Min', 'Conn-Rev Pkt Len Max',
                  'Conn-Rev Pkt Len Mean', 'Conn-Rev Pkt Len Tot']

CONN_BWD_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN', 'Time Flow count',
                  'Time Timedelta Min', 'Time Timedelta Max', 'Time Timedelta Mean', 'Time Pkt Num Min',
                  'Time Pkt Num Max', 'Time Pkt Num Mean', 'Time Pkt Num Tot', 'Time Pkt Len Min', 'Time Pkt Len Max',
                  'Time Pkt Len Mean', 'Time Pkt Len Tot', 'Time-Rev Flow count', 'Time-Rev Timedelta Min',
                  'Time-Rev Timedelta Max', 'Time-Rev Timedelta Mean', 'Time-Rev Pkt Num Min', 'Time-Rev Pkt Num Max',
                  'Time-Rev Pkt Num Mean', 'Time-Rev Pkt Num Tot', 'Time-Rev Pkt Len Min', 'Time-Rev Pkt Len Max',
                  'Time-Rev Pkt Len Mean', 'Time-Rev Pkt Len Tot', 'Con Flow count', 'Conn Timedelta Min',
                  'Conn Timedelta Max', 'Conn Timedelta Mean', 'Conn Pkt Num Min', 'Conn Pkt Num Max',
                  'Conn Pkt Num Mean', 'Conn Pkt Num Tot', 'Conn Pkt Len Min', 'Conn Pkt Len Max', 'Conn Pkt Len Mean',
                  'Conn Pkt Len Tot']

TIME_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN', 'Con Flow count',
              'Conn Timedelta Min', 'Conn Timedelta Max', 'Conn Timedelta Mean', 'Conn Pkt Num Min', 'Conn Pkt Num Max',
              'Conn Pkt Num Mean', 'Conn Pkt Num Tot', 'Conn Pkt Len Min', 'Conn Pkt Len Max', 'Conn Pkt Len Mean',
              'Conn Pkt Len Tot', 'Conn-Rev Flow count', 'Conn-Rev Timedelta Min', 'Conn-Rev Timedelta Max',
              'Conn-Rev Timedelta Mean', 'Conn-Rev Pkt Num Min', 'Conn-Rev Pkt Num Max', 'Conn-Rev Pkt Num Mean',
              'Conn-Rev Pkt Num Tot', 'Conn-Rev Pkt Len Min', 'Conn-Rev Pkt Len Max', 'Conn-Rev Pkt Len Mean',
              'Conn-Rev Pkt Len Tot']

TIME_FWD_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN', 'Con Flow count',
                  'Conn Timedelta Min', 'Conn Timedelta Max', 'Conn Timedelta Mean', 'Conn Pkt Num Min',
                  'Conn Pkt Num Max', 'Conn Pkt Num Mean', 'Conn Pkt Num Tot', 'Conn Pkt Len Min', 'Conn Pkt Len Max',
                  'Conn Pkt Len Mean', 'Conn Pkt Len Tot', 'Conn-Rev Flow count', 'Conn-Rev Timedelta Min',
                  'Conn-Rev Timedelta Max', 'Conn-Rev Timedelta Mean', 'Conn-Rev Pkt Num Min', 'Conn-Rev Pkt Num Max',
                  'Conn-Rev Pkt Num Mean', 'Conn-Rev Pkt Num Tot', 'Conn-Rev Pkt Len Min', 'Conn-Rev Pkt Len Max',
                  'Conn-Rev Pkt Len Mean', 'Conn-Rev Pkt Len Tot', 'Time-Rev Flow count', 'Time-Rev Timedelta Min',
                  'Time-Rev Timedelta Max', 'Time-Rev Timedelta Mean', 'Time-Rev Pkt Num Min', 'Time-Rev Pkt Num Max',
                  'Time-Rev Pkt Num Mean', 'Time-Rev Pkt Num Tot', 'Time-Rev Pkt Len Min', 'Time-Rev Pkt Len Max',
                  'Time-Rev Pkt Len Mean', 'Time-Rev Pkt Len Tot']

TIME_BWD_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN', 'Con Flow count',
                  'Conn Timedelta Min', 'Conn Timedelta Max', 'Conn Timedelta Mean', 'Conn Pkt Num Min',
                  'Conn Pkt Num Max', 'Conn Pkt Num Mean', 'Conn Pkt Num Tot', 'Conn Pkt Len Min', 'Conn Pkt Len Max',
                  'Conn Pkt Len Mean', 'Conn Pkt Len Tot', 'Conn-Rev Flow count', 'Conn-Rev Timedelta Min',
                  'Conn-Rev Timedelta Max', 'Conn-Rev Timedelta Mean', 'Conn-Rev Pkt Num Min', 'Conn-Rev Pkt Num Max',
                  'Conn-Rev Pkt Num Mean', 'Conn-Rev Pkt Num Tot', 'Conn-Rev Pkt Len Min', 'Conn-Rev Pkt Len Max',
                  'Conn-Rev Pkt Len Mean', 'Conn-Rev Pkt Len Tot', 'Time Flow count', 'Time Timedelta Min',
                  'Time Timedelta Max', 'Time Timedelta Mean', 'Time Pkt Num Min', 'Time Pkt Num Max',
                  'Time Pkt Num Mean', 'Time Pkt Num Tot', 'Time Pkt Len Min', 'Time Pkt Len Max', 'Time Pkt Len Mean',
                  'Time Pkt Len Tot']

COMB_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'VPN', 'Tot Pkts', 'TotLen']

COMB_FWD_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN', 'Time-Rev Flow count',
                  'Time-Rev Timedelta Min', 'Time-Rev Timedelta Max', 'Time-Rev Timedelta Mean', 'Time-Rev Pkt Num Min',
                  'Time-Rev Pkt Num Max', 'Time-Rev Pkt Num Mean', 'Time-Rev Pkt Num Tot', 'Time-Rev Pkt Len Min',
                  'Time-Rev Pkt Len Max', 'Time-Rev Pkt Len Mean', 'Time-Rev Pkt Len Tot', 'Conn-Rev Flow count',
                  'Conn-Rev Timedelta Min', 'Conn-Rev Timedelta Max', 'Conn-Rev Timedelta Mean', 'Conn-Rev Pkt Num Min',
                  'Conn-Rev Pkt Num Max', 'Conn-Rev Pkt Num Mean', 'Conn-Rev Pkt Num Tot', 'Conn-Rev Pkt Len Min',
                  'Conn-Rev Pkt Len Max', 'Conn-Rev Pkt Len Mean', 'Conn-Rev Pkt Len Tot']

COMB_BWD_DS_DT = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Tot Pkts', 'TotLen', 'VPN', 'Time Flow count',
                  'Time Timedelta Min', 'Time Timedelta Max', 'Time Timedelta Mean', 'Time Pkt Num Min',
                  'Time Pkt Num Max', 'Time Pkt Num Mean', 'Time Pkt Num Tot', 'Time Pkt Len Min', 'Time Pkt Len Max',
                  'Time Pkt Len Mean', 'Time Pkt Len Tot', 'Con Flow count', 'Conn Timedelta Min', 'Conn Timedelta Max',
                  'Conn Timedelta Mean', 'Conn Pkt Num Min', 'Conn Pkt Num Max', 'Conn Pkt Num Mean',
                  'Conn Pkt Num Tot', 'Conn Pkt Len Min', 'Conn Pkt Len Max', 'Conn Pkt Len Mean', 'Conn Pkt Len Tot']
