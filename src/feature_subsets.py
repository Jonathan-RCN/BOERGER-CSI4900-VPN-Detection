import src.config as cfg
import numpy as np
import pandas as pd



def get_target(csv_file):
    features_df = pd.read_csv(csv_file)
    return np.array(features_df['VPN'])



def get_baseline_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    baseline_ft_isolated = features_df.drop(cfg.BASELINE_DS_DT, axis=1)
    baseline_ft_name = list(baseline_ft_isolated.columns)
    baseline_ft_np=np.array(baseline_ft_isolated)
    return baseline_ft_np, baseline_ft_name


def get_time_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    time_ft_isolated = features_df.drop(cfg.TIME_DS_DT, axis=1)
    time_ft_name = list(time_ft_isolated.columns)
    time_ft_np = np.array(time_ft_isolated)
    return time_ft_np, time_ft_name

def get_conn_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    conn_ft_isolated = features_df.drop(cfg.CONN_DS_DT, axis=1)
    conn_ft_name = list(conn_ft_isolated.columns)
    conn_ft_np = np.array(conn_ft_isolated)
    return conn_ft_np, conn_ft_name

def get_comb_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    comb_ft_isolated = features_df.drop(cfg.COMB_DS_DT, axis=1)
    comb_ft_name = list(comb_ft_isolated.columns)
    comb_ft_np = np.array(comb_ft_isolated)
    return comb_ft_np, comb_ft_name

def get_time_fwd_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    time_fwd_ft_isolated = features_df.drop(cfg.TIME_FWD_DS_DT, axis=1)
    time_fwd_ft_name = list(time_fwd_ft_isolated.columns)
    time_fwd_ft_np = np.array(time_fwd_ft_isolated)
    return time_fwd_ft_np, time_fwd_ft_name

def get_time_bwd_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    time_bwd_ft_isolated = features_df.drop(cfg.TIME_BWD_DS_DT, axis=1)
    time_bwd_ft_name = list(time_bwd_ft_isolated.columns)
    time_bwd_ft_np = np.array(time_bwd_ft_isolated)
    return time_bwd_ft_np, time_bwd_ft_name

def get_conn_fwd_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    conn_fwd_ft_isolated = features_df.drop(cfg.CONN_FWD_DS_DT, axis=1)
    conn_fwd_ft_name = list(conn_fwd_ft_isolated.columns)
    conn_fwd_ft_np = np.array(conn_fwd_ft_isolated)
    return conn_fwd_ft_np, conn_fwd_ft_name

def get_conn_bwd_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    conn_bwd_ft_isolated = features_df.drop(cfg.CONN_BWD_DS_DT, axis=1)
    conn_bwd_ft_name = list(conn_bwd_ft_isolated.columns)
    conn_bwd_ft_np = np.array(conn_bwd_ft_isolated)
    return conn_bwd_ft_np, conn_bwd_ft_name

def get_comb_fwd_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    comb_fwd_ft_isolated = features_df.drop(cfg.COMB_FWD_DS_DT, axis=1)
    comb_fwd_ft_name = list(comb_fwd_ft_isolated.columns)
    comb_fwd_ft_np = np.array(comb_fwd_ft_isolated)
    return comb_fwd_ft_np, comb_fwd_ft_name

def get_comb_bwd_ft(csv_file):
    features_df = pd.read_csv(csv_file)
    comb_bwd_ft_isolated = features_df.drop(cfg.COMB_BWD_DS_DT, axis=1)
    comb_bwd_ft_name = list(comb_bwd_ft_isolated.columns)
    comb_bwd_ft_np = np.array(comb_bwd_ft_isolated)
    return comb_bwd_ft_np, comb_bwd_ft_name

