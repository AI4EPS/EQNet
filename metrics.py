# %%
import csv
import json
import os
import sys
import re
import h5py
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import obspy
from collections import OrderedDict
from tqdm import tqdm
import itertools
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN

import datasets
import torch
import torch.nn as nn
import torch.functional as F

sns.set_theme()
# %% [markdown]
# ## Metrics for NC dataset

# %% [markdown]
# ### Config

# %%

parser = argparse.ArgumentParser(description='Metrics for NC dataset', add_help=True)

parser.add_argument('--h5_file', type=str, default="/home/wanghy/tests/quakeflow_nc/data/NC2020.h5", help='h5 file path')
parser.add_argument('--pick_dir', type=str, default="/home/wanghy/tests/EQNet/test_eqnet_predict/picks_eqnet", help='pick dir path')
parser.add_argument('--event_dir', type=str, default="/home/wanghy/tests/EQNet/test_eqnet_predict/events_eqnet", help='event dir path')
parser.add_argument('--suffix', type=str, default="", help='suffix of the pics')
parser.add_argument('--model', type=str, default="eqnet", help='the model that performed prediction')

args = parser.parse_args()
h5_file = args.h5_file
pick_dir = args.pick_dir
event_dir = args.event_dir
csv_file = f"{h5_file.split('.')[0]}.csv"
suffix = args.suffix
if suffix != "":
    suffix = f"_{suffix}"

dt_bar = 0.5 # s
distribution_bar = 0.5 # s
station_threshold = 10
event_dt_bar = 3 # s
dbscan_epsilon = 5
time_space_weight = [1, 0.5, 0.5, 1]

sample_rate = 100
feature_scale = 16
degree2km = 111.32

# %% [markdown]
# ### Phase picking

# %%
# 获取csv_dir下所有的CSV文件的路径
if args.model == "eqnet":
    pick_files = [os.path.join(pick_dir, f) for f in os.listdir(pick_dir) if f.endswith(".csv")]
elif args.model == "phasenet_plus":
    pick_files = [os.path.join(pick_dir, f) for f in os.listdir(pick_dir) if os.path.isdir(os.path.join(pick_dir, f))]

class PickMetric:
    def __init__(self, sample_rate=100):
        self.num_p_picks = 0
        self.num_s_picks = 0
        self.num_p_picks_label = 0
        self.num_s_picks_label = 0

        self.time_res_p = []
        self.time_res_s = []
        self.magnitude_p = []
        self.snr_p = []
        self.epicenter_distance_p = []
        self.magnitude_s = []
        self.snr_s = []
        self.epicenter_distance_s = []
        self.scores_p = []
        self.scores_s = []

        self.empty_files = 0
        
        self._sample_rate=sample_rate
    
    def list2numpy(self):
        self.time_res_p = np.array(self.time_res_p)/self._sample_rate
        self.time_res_s = np.array(self.time_res_s)/self._sample_rate
        self.magnitude_p = np.array(self.magnitude_p)
        self.snr_p = np.array(self.snr_p)
        self.epicenter_distance_p =  np.array(self.epicenter_distance_p)
        self.magnitude_s = np.array(self.magnitude_s)
        self.snr_s = np.array(self.snr_s)
        self.epicenter_distance_s = np.array(self.epicenter_distance_s)
        self.scores_p = np.array(self.scores_p)
        self.scores_s = np.array(self.scores_s)
        
        return
        
    def update_matched_picks(self, min_res_p, min_res_s, min_scores_p, min_scores_s, magnitude, distance_km, snr_sta):
        self.time_res_p+=min_res_p
        self.time_res_s+=min_res_s
        self.scores_p+=min_scores_p
        self.scores_s+=min_scores_s
        self.magnitude_p+=magnitude*len(min_res_p)
        self.magnitude_s+=magnitude*len(min_res_s)
        self.epicenter_distance_p+=distance_km*len(min_res_p)
        self.epicenter_distance_s+=distance_km*len(min_res_s)
        self.snr_p+=snr_sta*len(min_res_p)
        self.snr_s+=snr_sta*len(min_res_s)

        return
    
    def calc_metrics(self, dt_bar=0.5, distribution_bar=0.5):
        index_p = abs(self.time_res_p) < dt_bar
        index_s = abs(self.time_res_s) < dt_bar
        self.tp_p_picks = self.time_res_p[index_p]
        self.tp_s_picks = self.time_res_s[index_s]
        self.tp_p_magnitude = self.magnitude_p[index_p]
        self.tp_s_magnitude = self.magnitude_s[index_s]
        self.tp_p_distance = self.epicenter_distance_p[index_p]
        self.tp_s_distance = self.epicenter_distance_s[index_s]
        self.tp_p_snr = self.snr_p[index_p]
        self.tp_s_snr = self.snr_s[index_s]
        self.tp_p_scores = self.scores_p[index_p]
        self.tp_s_scores = self.scores_s[index_s]

        true_positives_p = len(self.tp_p_picks)
        true_positives_s = len(self.tp_s_picks)
        false_positives_p = self.num_p_picks - true_positives_p
        false_positives_s = self.num_s_picks - true_positives_s
        false_negatives_p = self.num_p_picks_label - true_positives_p
        false_negatives_s = self.num_s_picks_label - true_positives_s

        precision_p = true_positives_p / (true_positives_p + false_positives_p)
        precision_s = true_positives_s / (true_positives_s + false_positives_s)
        recall_p = true_positives_p / (true_positives_p + false_negatives_p)
        recall_s = true_positives_s / (true_positives_s + false_negatives_s)
        f1_p = 2 * precision_p * recall_p / (precision_p + recall_p)
        f1_s = 2 * precision_s * recall_s / (precision_s + recall_s)

        # the distribution of time residual
        self.res_p = self.time_res_p[abs(self.time_res_p) < distribution_bar]
        self.res_s = self.time_res_s[abs(self.time_res_s) < distribution_bar]

        miu_p = self.res_p.mean()
        miu_s = self.res_s.mean()
        sigma_p = self.res_p.std()
        sigma_s = self.res_s.std()
        
        # metric dict
        self.metric_dict = {
            "num_p_picks": self.num_p_picks,
            "num_s_picks": self.num_s_picks,
            "num_p_picks_label": self.num_p_picks_label,
            "num_s_picks_label": self.num_s_picks_label,
            "true_positives_p": true_positives_p,
            "true_positives_s": true_positives_s,
            "false_positives_p": false_positives_p,
            "false_positives_s": false_positives_s,
            "false_negatives_p": false_negatives_p,
            "false_negatives_s": false_negatives_s,
            "precision_p": precision_p,
            "precision_s": precision_s,
            "recall_p": recall_p,
            "recall_s": recall_s,
            "f1_p": f1_p,
            "f1_s": f1_s,
            "miu_p": miu_p,
            "miu_s": miu_s,
            "sigma_p": sigma_p,
            "sigma_s": sigma_s,
        }
        
        return
    
    def plot_residual(self, save_path=None):
        print(f"p_picks: {self.num_p_picks}, s_picks: {self.num_s_picks}, p_picks_label: {self.num_p_picks_label}, s_picks_label: {self.num_s_picks_label}")
        print(f"TP_p: {self.metric_dict['true_positives_p']}, TP_s: {self.metric_dict['true_positives_s']}, FP_p: {self.metric_dict['false_positives_p']}, FP_s: {self.metric_dict['false_positives_s']}, FN_p: {self.metric_dict['false_negatives_p']}, FN_s: {self.metric_dict['false_negatives_s']}")
        print(f"precision_p: {self.metric_dict['precision_p']:.4f}, recall_p: {self.metric_dict['recall_p']:.4f}, f1_p: {self.metric_dict['f1_p']:.4f}")
        print(f"precision_s: {self.metric_dict['precision_s']:.4f}, recall_s: {self.metric_dict['recall_s']:.4f}, f1_s: {self.metric_dict['f1_s']:.4f}")
        print(f"miu_p: {self.metric_dict['miu_p']}, sigma_p: {self.metric_dict['sigma_p']}")
        print(f"miu_s: {self.metric_dict['miu_s']}, sigma_s: {self.metric_dict['sigma_s']}")
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.hist(self.res_p, bins=11, range=(-distribution_bar, distribution_bar))
        plt.title("P phase")
        plt.subplot(122)
        plt.hist(self.res_s, bins=11, range=(-distribution_bar, distribution_bar))
        plt.title("S phase")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        
        return

# %%
pick_metric = PickMetric(sample_rate=sample_rate)
pick_metric_thld = PickMetric(sample_rate=sample_rate)

# %%
with h5py.File(h5_file, "r") as fp:
    for pick in tqdm(pick_files):
        if args.model == "eqnet":
            event_id = re.sub(r'\.csv$', '', os.path.basename(pick))
        elif args.model == "phasenet_plus":
            event_id = pick.split('/')[-1]

        try:
            event = fp[event_id]
        except:
            print(f"event {event_id} not in {h5_file}")
            continue
        num_stations = len(list(event.keys()))

        # skip the empty pick file
        empty = False
        if args.model == "eqnet" and os.path.getsize(pick) == 0:
            empty = True
        elif args.model == "phasenet_plus":
            empty = True
            stations=[os.path.join(pick, f) for f in os.listdir(pick) if f.endswith(".csv")]
            for sta in stations:
                if os.path.getsize(sta) != 0:
                    empty = False
                    break
        if empty:
            pick_metric.empty_files += 1
            pick_metric_thld.empty_files +=1 if num_stations>=station_threshold else 0
            
            for sta_id in list(event.keys()):
                phase_type_gt = event[sta_id].attrs["phase_type"]
                try:
                    if len(phase_type_gt)==1 or event[sta_id].attrs["phase_status"]=='automatic':
                        continue
                except:
                    pass
                p_picks_gt = np.sort(event[sta_id].attrs["phase_index"][phase_type_gt=='P'])
                s_picks_gt = np.sort(event[sta_id].attrs["phase_index"][phase_type_gt=='S'])
                pick_metric.num_p_picks_label+=len(p_picks_gt)
                pick_metric.num_s_picks_label+=len(s_picks_gt)
                pick_metric_thld.num_p_picks_label += len(p_picks_gt) if num_stations>=station_threshold else 0
                pick_metric_thld.num_s_picks_label += len(s_picks_gt) if num_stations>=station_threshold else 0
            continue
        
        if args.model == "eqnet":
            df = pd.read_csv(pick)
        elif args.model == "phasenet_plus":
            dfs = []
            stations=[os.path.join(pick, f) for f in os.listdir(pick) if f.endswith(".csv")]
            for sta in stations:
                if os.path.getsize(sta) == 0:
                    continue
                dfs.append(pd.read_csv(sta))
            df = pd.concat(dfs)
        df = df.sort_values(by=["station_id", "phase_index"], ignore_index=True)
        
        for sta_id in list(event.keys()):
            phase_type_gt = event[sta_id].attrs["phase_type"]
            
            try:
                if len(phase_type_gt)==1 or event[sta_id].attrs["phase_status"]=='automatic':
                    continue
            except:
                pass
            p_picks_gt = np.sort(event[sta_id].attrs["phase_index"][phase_type_gt=='P'])
            s_picks_gt = np.sort(event[sta_id].attrs["phase_index"][phase_type_gt=='S'])
            pick_metric.num_p_picks_label+=len(p_picks_gt)
            pick_metric.num_s_picks_label+=len(s_picks_gt)
            pick_metric_thld.num_p_picks_label += len(p_picks_gt) if num_stations>=station_threshold else 0
            pick_metric_thld.num_s_picks_label += len(s_picks_gt) if num_stations>=station_threshold else 0
            if sta_id not in df["station_id"].values:
                continue
            
            df_sta = df[df["station_id"] == sta_id]
            df_sta_p = df_sta[df_sta["phase_type"] == "P"]
            df_sta_s = df_sta[df_sta["phase_type"] == "S"]
            pick_metric.num_p_picks += len(df_sta_p)
            pick_metric.num_s_picks += len(df_sta_s)
            pick_metric_thld.num_p_picks += len(df_sta_p) if num_stations>=station_threshold else 0
            pick_metric_thld.num_s_picks += len(df_sta_s) if num_stations>=station_threshold else 0
            if len(df_sta_p) == 0:
                min_res_p = [np.inf]
                min_scores_p = [np.inf]
            else:
                diff_p = df_sta_p["phase_index"].values[:, None] - p_picks_gt[None, :]
                index_p = np.argmin(np.abs(diff_p), axis=0)
                #assert len(np.unique(index_p)) == len(index_p), f"duplicated matched p picks in {event_id} {sta_id} {phase_type_gt}"
                #if len(np.unique(index_p)) != len(index_p):
                #    print(f"duplicated matched p picks in {event_id} {sta_id} {phase_type_gt} {len(df_sta_p)} {index_p}")
                min_res_p = list(diff_p[index_p, np.arange(len(index_p))])
                min_scores_p = list(df_sta_p["phase_score"].values[index_p])
            if len(df_sta_s) == 0:
                min_res_s = [np.inf]
                min_scores_s = [np.inf]
            else:
                diff_s = df_sta_s["phase_index"].values[:, None] - s_picks_gt[None, :]
                index_s = np.argmin(np.abs(diff_s), axis=0)
                #assert len(np.unique(index_s)) == len(index_s), f"duplicated matched s picks in {event_id} {sta_id} {phase_type_gt}"
                #if len(np.unique(index_s)) != len(index_s):
                #    print(f"duplicated matched s picks in {event_id} {sta_id} {phase_type_gt} {len(df_sta_s)} {index_s}")
                min_res_s = list(diff_s[index_s, np.arange(len(index_s))])
                min_scores_s = list(df_sta_s["phase_score"].values[index_s])
                
            snr_sta = event[sta_id].attrs["snr"]
            snr_sta = snr_sta[snr_sta > 0]
            if len(snr_sta) > 0:
                snr_sta = snr_sta.mean()
            else:
                snr_sta = 0
            pick_metric.update_matched_picks(min_res_p, min_res_s, min_scores_p, min_scores_s, [event.attrs["magnitude"]], [event[sta_id].attrs["distance_km"]], [snr_sta])
            if num_stations>=station_threshold:
                pick_metric_thld.update_matched_picks(min_res_p, min_res_s, min_scores_p, min_scores_s, [event.attrs["magnitude"]], [event[sta_id].attrs["distance_km"]], [snr_sta])

pick_metric.list2numpy()
pick_metric_thld.list2numpy()

print(f"whole dataset empty files: {pick_metric.empty_files}")
print(f"num_station >= {station_threshold} subset empty files: {pick_metric_thld.empty_files}")

# %%
pick_metric.calc_metrics(dt_bar=dt_bar, distribution_bar=distribution_bar)
pick_metric_thld.calc_metrics(dt_bar=dt_bar, distribution_bar=distribution_bar)

# %%
print("\nWhole dataset:\n")
pick_metric.plot_residual(save_path=f"./metrics/phase_picking{suffix}.png")
print(f"\nnum_station >= {station_threshold} subset:\n")
pick_metric_thld.plot_residual(save_path=f"./metrics/phase_picking_threshold{station_threshold}{suffix}.png")
print("\n")

# %% [markdown]
# ### Event detection

# %%
# 获取csv_dir下所有的CSV文件的路径
if args.model == "eqnet":
    event_files = [os.path.join(event_dir, f) for f in os.listdir(event_dir) if f.endswith(".csv")]
elif args.model == "phasenet_plus":
    event_files = [os.path.join(event_dir, f) for f in os.listdir(event_dir) if os.path.isdir(os.path.join(event_dir, f))]

class EventMetric:
    def __init__(self):
        self.num_event_picks = 0
        self.num_event_picks_label = 0
        
        self.time_res = []
        self.x_res = []
        self.y_res = []
        self.z_res = []
        self.x = []
        self.y = []
        self.z = []  
        self.magnitude = []
        self.magnitude_pred = []
        self.snr = []
        self.epicenter_distance_pred = []
        self.epicenter_distance_res = []
        self.azimuth_res = []
        self.empty_files = 0
        self.no_match_files = 0
    
    def list2numpy(self):
        self.time_res = np.array(self.time_res)
        self.x_res = np.array(self.x_res)
        self.y_res = np.array(self.y_res)
        self.z_res = np.array(self.z_res)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        self.magnitude_pred = np.array(self.magnitude_pred)
        self.magnitude = np.array(self.magnitude)
        self.epicenter_distance_pred = np.concatenate(self.epicenter_distance_pred)
        self.epicenter_distance_res = np.concatenate(self.epicenter_distance_res)
        self.azimuth_res = np.concatenate(self.azimuth_res)
        
        return
    
    def update_matched_picks(self, min_time_diff, x_diff, y_diff, z_diff, x_pred, y_pred, z_pred, magnitude_true, magnitude_pred, epicenter_distance_pred, epicenter_distance_res, azimuth_res):
        self.time_res+=min_time_diff
        self.x_res+=x_diff
        self.y_res+=y_diff
        self.z_res+=z_diff
        self.x+=x_pred
        self.y+=y_pred
        self.z+=z_pred
        self.magnitude+=magnitude_true
        self.magnitude_pred+=magnitude_pred
        for i, min_t_diff in enumerate(min_time_diff):
            if abs(min_t_diff) < 3:
                self.epicenter_distance_pred.append(epicenter_distance_pred[i])
                self.epicenter_distance_res.append(epicenter_distance_res[i])
                self.azimuth_res.append(azimuth_res[i])
        if abs(min_time_diff[0]) > 10:
            self.no_match_files += 1
        
        return
    
    def calc_metrics(self, event_dt_bar=3):
        index = abs(self.time_res) < event_dt_bar
        self.tp_picks = self.time_res[index]
        self.tp_x_res = self.x_res[index]
        self.tp_y_res = self.y_res[index]
        self.tp_z_res = self.z_res[index]
        self.tp_x = self.x[index]
        self.tp_y = self.y[index]
        self.tp_z = self.z[index]
        self.tp_magnitude = self.magnitude[index]
        self.tp_magnitude_pred = self.magnitude_pred[index]
        #self.tp_epicenter_distance_pred = self.epicenter_distance_pred[index]
        #self.tp_epicenter_distance_res = self.epicenter_distance_res[index]

        true_positives = len(self.tp_picks)
        false_positives = self.num_event_picks - true_positives
        false_negatives = self.num_event_picks_label - true_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)

        # the distribution of time residual
        miu_t = self.tp_picks.mean()
        miu_x = self.tp_x_res.mean()
        miu_y = self.tp_y_res.mean()
        miu_z = self.tp_z_res.mean()
        miu_m = (self.tp_magnitude_pred - self.tp_magnitude).mean()
        miu_dist = self.epicenter_distance_res.flatten().mean()
        miu_azi = self.azimuth_res.flatten().mean()
        sigma_t = self.tp_picks.std()
        sigma_x = self.tp_x_res.std()
        sigma_y = self.tp_y_res.std()
        sigma_z = self.tp_z_res.std()
        sigma_m = (self.tp_magnitude_pred - self.tp_magnitude).std()
        sigma_dist = self.epicenter_distance_res.flatten().std()
        sigma_azi = self.azimuth_res.flatten().std()
        
        # metric dict
        self.metric_dict = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "miu_t": miu_t,
            "miu_x": miu_x,
            "miu_y": miu_y,
            "miu_z": miu_z,
            "miu_m": miu_m,
            "miu_dist": miu_dist,
            "miu_azi": miu_azi,
            "sigma_t": sigma_t,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_z": sigma_z,
            "sigma_m": sigma_m,
            "sigma_dist": sigma_dist,
            "sigma_azi": sigma_azi,
        }
        
        return
    
    def plot_residual(self, save_path=None):
        print(f"picks: {self.num_event_picks}, picks_label: {self.num_event_picks_label}")
        print(f"TP: {self.metric_dict['true_positives']}, FP: {self.metric_dict['false_positives']}, FN: {self.metric_dict['false_negatives']}")
        print(f"precision: {self.metric_dict['precision']:.4f}, recall: {self.metric_dict['recall']:.4f}, f1: {self.metric_dict['f1']:.4f}")
        print(f"miu_t: {self.metric_dict['miu_t']:.4f}, miu_x: {self.metric_dict['miu_x']:.4f}, miu_y: {self.metric_dict['miu_y']:.4f}, miu_z: {self.metric_dict['miu_z']:.4f}, miu_m: {self.metric_dict['miu_m']:.4f}, miu_dist: {self.metric_dict['miu_dist']:.4f}, miu_azi: {self.metric_dict['miu_azi']:.4f}")
        print(f"sigma_t: {self.metric_dict['sigma_t']:.4f}, sigma_x: {self.metric_dict['sigma_x']:.4f}, sigma_y: {self.metric_dict['sigma_y']:.4f}, sigma_z: {self.metric_dict['sigma_z']:.4f}, sigma_m: {self.metric_dict['sigma_m']:.4f}, sigma_dist: {self.metric_dict['sigma_dist']:.4f}, sigma_azi: {self.metric_dict['sigma_azi']:.4f}")

        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        ax[0, 0].hist(self.tp_picks, bins=31, range=(-3, 3))
        ax[0, 0].set_title("event time residual")
        ax[0, 1].hist(self.tp_x_res, bins=31, range=(-30, 30))
        ax[0, 1].set_title("event x residual")
        ax[0, 2].hist(self.tp_y_res, bins=31, range=(-30, 30))
        ax[0, 2].set_title("event y residual")
        ax[0, 3].hist(self.tp_z_res, bins=21, range=(-10, 10))
        ax[0, 3].set_title("event z residual")
        ax[1, 0].hist(self.tp_magnitude_pred-self.tp_magnitude, bins=21, range=(-2, 2))
        ax[1, 0].set_title("magnitude residual")
        ax[1, 1].hist(self.epicenter_distance_res, bins=41, range=(-20, 20))
        ax[1, 1].set_title("epicenter distance residual") 
        ax[1, 2].hist(self.azimuth_res, bins=141, range=(-360, 360))
        ax[1, 2].set_title("azimuth residual")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        
        return
    
def average_location(locations):
    # 将经纬度从度数转换为弧度
    locations = np.radians(locations)

    # 将经纬度转换为笛卡尔坐标
    x = np.cos(locations[:, 1]) * np.cos(locations[:, 0])
    y = np.cos(locations[:, 1]) * np.sin(locations[:, 0])
    z = np.sin(locations[:, 1])

    # 计算平均的笛卡尔坐标
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    # 将平均的笛卡尔坐标转换回经纬度
    lon = np.arctan2(y_mean, x_mean)
    hyp = np.sqrt(x_mean * x_mean + y_mean * y_mean)
    lat = np.arctan2(z_mean, hyp)

    # 将经纬度从弧度转换为度数
    return np.degrees([lon, lat])

# %%
event_metric = EventMetric()
event_metric_thld = EventMetric()

dbscan = DBSCAN(eps=dbscan_epsilon, min_samples=1)

with h5py.File(h5_file, "r") as fp:
    for pick in tqdm(event_files):
        if args.model == "eqnet":
            event_id = re.sub(r'\.csv$', '', os.path.basename(pick))
        elif args.model == "phasenet_plus":
            event_id = pick.split('/')[-1]

        event = fp[event_id]
        num_stations = len(list(event.keys()))
        event_metric.num_event_picks_label += 1
        event_metric_thld.num_event_picks_label += 1 if num_stations>=station_threshold else 0

        # skip the empty pick file
        if os.path.getsize(pick) == 0:
            event_metric.empty_files += 1
            event_metric_thld.empty_files +=1 if num_stations>=station_threshold else 0
            continue
        if args.model == "eqnet":
            df = pd.read_csv(pick)
        elif args.model == "phasenet_plus":
            df = pd.concat(
                [pd.read_csv(os.path.join(pick, f)) for f in os.listdir(pick) if f.endswith(".csv")]
            )
        assert 'event_center_index' in df.columns, f'event_center_index not in {event_id}.columns'
        assert 'p_index' in df.columns, f'p_index not in {event_id}.columns'
        assert 's_index' in df.columns, f's_index not in {event_id}.columns'
        assert 'event_original_time' in df.columns, f'event_original_time not in {event_id}.columns'
        
        stations = df["station_id"].values
        hypoparameters = np.zeros((len(df),9))
        hypoparameters[:,0] = df["event_index"].values
        hypoparameters[:,1] = df["event_location_x"].values
        hypoparameters[:,2] = df["event_location_y"].values
        hypoparameters[:,3] = df["event_location_z"].values
        hypoparameters[:,4] = df["event_longitude"].values
        hypoparameters[:,5] = df["event_latitude"].values
        hypoparameters[:,6] = df["epicentral_distance"].values
        hypoparameters[:,7] = df["azimuth"].values
        hypoparameters[:,8] = df["magnitude"].values
        
        labels = dbscan.fit_predict(hypoparameters[:,:4]*np.array(time_space_weight)[None,:]) # weighted
        classes, counts = np.unique(labels, return_counts=True)
        if num_stations > 5 and len(classes) > 1:
            if classes[0] == -1:
                classes = classes[1:]
                counts = counts[1:]
            classes = classes[counts>1]
            
        event_metric.num_event_picks += len(classes)
        event_metric_thld.num_event_picks += len(classes) if num_stations>=station_threshold else 0
        min_time_diff = np.inf
        x_diff = np.inf
        y_diff = np.inf
        z_diff = np.inf
        magnitude_pr = np.inf
        x_pred = np.inf
        y_pred = np.inf
        z_pred = np.inf
        station_ids = list(event.keys())
        label_x = event.attrs["longitude"]*degree2km*np.cos(np.radians(event.attrs["latitude"]))
        label_y = event.attrs["latitude"]*degree2km
        label_z = event.attrs["depth_km"]
        labels_dist = np.array([event[sta_id].attrs["distance_km"] for sta_id in stations])
        labels_lat = np.array([event[sta_id].attrs["latitude"] for sta_id in stations])
        labels_lon = np.array([event[sta_id].attrs["longitude"] for sta_id in stations])
        dlon = event.attrs["longitude"] - labels_lon
        dlat = event.attrs["latitude"] - labels_lat
        y = np.sin(np.radians(dlon)) * np.cos(np.radians(event.attrs["latitude"]))
        x = np.cos(np.radians(labels_lat)) * np.sin(np.radians(event.attrs["latitude"])) \
            - np.sin(np.radians(labels_lat)) * np.cos(np.radians(event.attrs["latitude"])) * np.cos(np.radians(dlon))
        labels_azi = np.degrees(np.arctan2(y, x))
        for c in classes:
            cluster_picks = hypoparameters[labels==c]
            idx = np.argsort(cluster_picks[:,6])
            label_dist = labels_dist[labels==c][idx]
            label_azi = labels_azi[labels==c][idx]
            cluster_picks = cluster_picks[idx]#[:5]
            time_diff = (np.median(cluster_picks[:5,0])*feature_scale - event.attrs["event_time_index"])/sample_rate
            if abs(time_diff) < abs(min_time_diff):
                #avg_center = average_location(cluster_picks[:,4:6])
                avg_center = np.median(cluster_picks[:,4:6], axis=0)
                x_pred = avg_center[0]*np.cos(np.radians(avg_center[1])) * degree2km
                y_pred = avg_center[1] * degree2km
                z_pred = np.median(cluster_picks[:,3])
                x_diff = (x_pred - label_x)
                y_diff = (y_pred - label_y)
                z_diff = (z_pred - label_z)
                dist_diff = (cluster_picks[:,6] - label_dist)
                azi_diff = (cluster_picks[:,7] - label_azi) % 360
                azi_diff[azi_diff>180] = azi_diff[azi_diff>180] - 360
                magnitude_pr = np.median(cluster_picks[:,8])
                min_time_diff = time_diff
        event_metric.update_matched_picks([min_time_diff], [x_diff], [y_diff], [z_diff], [x_pred], [y_pred], [z_pred], [event.attrs["magnitude"]], [magnitude_pr], [cluster_picks[:,6]], [dist_diff], [azi_diff])
        if num_stations>=station_threshold:
            event_metric_thld.update_matched_picks([min_time_diff], [x_diff], [y_diff], [z_diff], [x_pred], [y_pred], [z_pred], [event.attrs["magnitude"]], [magnitude_pr], [cluster_picks[:,6]], [dist_diff], [azi_diff])
        
event_metric.list2numpy()
event_metric_thld.list2numpy()

print(f"whole dataset empty files: {event_metric.empty_files}")
print(f"whole dataset no match files: {event_metric.no_match_files}")
print("\n")
print(f"num_station >= {station_threshold} subset empty files: {event_metric_thld.empty_files}")
print(f"num_station >= {station_threshold} subset no match files: {event_metric_thld.no_match_files}")

# %%
event_metric.calc_metrics(event_dt_bar=event_dt_bar)
event_metric_thld.calc_metrics(event_dt_bar=event_dt_bar)

# %%
print("\nWhole dataset:\n")
event_metric.plot_residual(f"./metrics/event_residual{suffix}.png")
print(f"azimuth residual: len: {len(event_metric.azimuth_res.flatten())}, abs medium: {np.median(np.abs(event_metric.azimuth_res.flatten()))}, max: {event_metric.azimuth_res.flatten()[np.argmax(np.abs(event_metric.azimuth_res.flatten()))]}, min: {event_metric.azimuth_res.flatten()[np.argmin(np.abs(event_metric.azimuth_res.flatten()))]}")
print(f"\nnum_station >= {station_threshold} subset:\n")
event_metric_thld.plot_residual(f"./metrics/event_residual_threshold{station_threshold}{suffix}.png")
print(f"azimuth residual: len: {len(event_metric_thld.azimuth_res.flatten())}, abs medium: {np.median(np.abs(event_metric_thld.azimuth_res.flatten()))}, max: {event_metric_thld.azimuth_res.flatten()[np.argmax(np.abs(event_metric_thld.azimuth_res.flatten()))]}, min: {event_metric_thld.azimuth_res.flatten()[np.argmin(np.abs(event_metric_thld.azimuth_res.flatten()))]}")