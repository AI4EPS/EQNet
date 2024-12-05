# %%
import json
import os
import sys
import re
import h5py
import errno
import pickle
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
import torch.nn.functional as F

sns.set_theme(style="whitegrid")

# %%
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
        self.lon_pred = []
        self.lat_pred = []
        self.lon_res = []
        self.lat_res = []
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
        self.lon_pred = np.array(self.lon_pred)
        self.lat_pred = np.array(self.lat_pred)
        self.lon_res = np.array(self.lon_res)
        self.lat_res = np.array(self.lat_res)
        if len(self.epicenter_distance_res)!=0:
            self.epicenter_distance_pred = np.concatenate(self.epicenter_distance_pred)
            self.epicenter_distance_res = np.concatenate(self.epicenter_distance_res)
            self.azimuth_res = np.concatenate(self.azimuth_res)
        
        return
    
    def update_matched_picks(self, min_time_diff:list, x_diff:list, y_diff:list, z_diff:list, x_pred:list, y_pred:list, z_pred:list, magnitude_true:list, magnitude_pred:list, epicenter_distance_pred:list, epicenter_distance_res:list, azimuth_res:list, lon_pred:list=[], lat_pred:list=[], lon_res:list=[], lat_res:list=[]):
        self.time_res+=min_time_diff
        self.x_res+=x_diff
        self.y_res+=y_diff
        self.z_res+=z_diff
        self.x+=x_pred
        self.y+=y_pred
        self.z+=z_pred
        self.magnitude+=magnitude_true
        self.magnitude_pred+=magnitude_pred
        self.lon_pred+=lon_pred
        self.lat_pred+=lat_pred
        self.lon_res+=lon_res
        self.lat_res+=lat_res
        for i, min_t_diff in enumerate(min_time_diff):
            if abs(min_t_diff) < 3 and len(epicenter_distance_pred)!=0:
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
        if len(self.lon_res) > 0:
            self.tp_lon_res = self.lon_res[index]
            self.tp_lat_res = self.lat_res[index]
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
        res_m = self.tp_magnitude_pred - self.tp_magnitude
        miu_m = (res_m[~np.isnan(res_m)]).mean()
        sigma_t = self.tp_picks.std()
        sigma_x = self.tp_x_res.std()
        sigma_y = self.tp_y_res.std()
        sigma_z = self.tp_z_res.std()
        sigma_m = (res_m[~np.isnan(res_m)]).std()
        if len(self.epicenter_distance_res)!=0:
            miu_dist = self.epicenter_distance_res.flatten().mean()
            miu_azi = self.azimuth_res.flatten().mean()
            sigma_dist = self.epicenter_distance_res.flatten().std()
            sigma_azi = self.azimuth_res.flatten().std()
        if len(self.lon_res) != 0:
            miu_lon = self.tp_lon_res.mean()
            miu_lat = self.tp_lat_res.mean()
            sigma_lon = self.tp_lon_res.std()
            sigma_lat = self.tp_lat_res.std()
        
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
            "sigma_t": sigma_t,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_z": sigma_z,
            "sigma_m": sigma_m,
        }

        if len(self.lon_res) != 0:
            self.metric_dict["miu_lon"] = miu_lon
            self.metric_dict["miu_lat"] = miu_lat
            self.metric_dict["sigma_lon"] = sigma_lon
            self.metric_dict["sigma_lat"] = sigma_lat
        
        return
    
    def plot_residual(self, save_path=None):
        print(f"picks: {self.num_event_picks}, picks_label: {self.num_event_picks_label}")
        print(f"TP: {self.metric_dict['true_positives']}, FP: {self.metric_dict['false_positives']}, FN: {self.metric_dict['false_negatives']}")
        print(f"precision: {self.metric_dict['precision']:.4f}, recall: {self.metric_dict['recall']:.4f}, f1: {self.metric_dict['f1']:.4f}")
        print(f"miu_t: {self.metric_dict['miu_t']:.4f}, miu_x: {self.metric_dict['miu_x']:.4f}, miu_y: {self.metric_dict['miu_y']:.4f}, miu_z: {self.metric_dict['miu_z']:.4f}, miu_m: {self.metric_dict['miu_m']:.4f}")
        print(f"sigma_t: {self.metric_dict['sigma_t']:.4f}, sigma_x: {self.metric_dict['sigma_x']:.4f}, sigma_y: {self.metric_dict['sigma_y']:.4f}, sigma_z: {self.metric_dict['sigma_z']:.4f}, sigma_m: {self.metric_dict['sigma_m']:.4f}")
        if len(self.lon_res) != 0:
            print(f"miu_lon: {self.metric_dict['miu_lon']:.4f}, miu_lat: {self.metric_dict['miu_lat']:.4f}")
            print(f"sigma_lon: {self.metric_dict['sigma_lon']:.4f}, sigma_lat: {self.metric_dict['sigma_lat']:.4f}")

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
        if len(self.epicenter_distance_res)!=0:
            ax[1, 1].hist(self.epicenter_distance_res, bins=41, range=(-20, 20))
            ax[1, 1].set_title("epicenter distance residual") 
            ax[1, 2].hist(self.azimuth_res, bins=121, range=(-300, 300))
            ax[1, 2].set_title("azimuth residual")
        if len(self.epicenter_distance_res)==0 and len(self.lon_res) != 0:
            ax[1, 1].hist(self.lon_res, bins=31, range=(-0.5, 0.5))
            ax[1, 1].set_title("event longitude residual")
            ax[1, 2].hist(self.lat_res, bins=31, range=(-0.5, 0.5))
            ax[1, 2].set_title("event latitude residual")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        
        return
    
# %% [markdown]
# ## Metrics for Ridgecrest
# %%
parser = argparse.ArgumentParser(description='Metrics for NC dataset', add_help=True)

parser.add_argument('--pick_dir', type=str, default="/home/wanghy/tests/EQNet/test_eqnet_mseed/picks_eqnet", help='pick dir path')
parser.add_argument('--event_dir', type=str, default="/home/wanghy/tests/EQNet/test_eqnet_mseed/events_eqnet", help='event dir path')
parser.add_argument('--catalog_dir', type=str, default="/home/wanghy/tests/QuakeFlow/scripts/local/Ridgecrest/obspy/catalog.csv", help="true catalog dir")
parser.add_argument('--catalog_pred_dir', type=str, default="/home/wanghy/tests/EQNet/test_eqnet_mseed/catalog_pred.csv", help="Predicted catalog dir")
parser.add_argument('--station_info_dir', type=str, default="/home/wanghy/tests/QuakeFlow/scripts/local/Ridgecrest/obspy/stations.json", help="station information dir (json)")
parser.add_argument('--save_path', type=str, default="./ridgecrest_metrics/", help='save path')
parser.add_argument('--score_threshold', type=float, default=0.3, help='the threshold of phase score')
parser.add_argument('--suffix', type=str, default="", help='suffix of the pics')
parser.add_argument('--cluster_method', type=str, default="dbscan", help='cluster method')
parser.add_argument('--test_only', action='store_true', help='test only')
parser.add_argument('--use_depth', action='store_true', help='use depth during matching')

args = parser.parse_args()
pick_dir = args.pick_dir
event_dir = args.event_dir
pick_dir = os.path.dirname(event_dir)+"/"+event_dir.split("/")[-1].replace("events", "picks")
check_point_dir = os.path.dirname(event_dir)+"/metrics"
catalog_dir = args.catalog_dir
catalog_pred_dir = args.catalog_pred_dir
station_info_dir = args.station_info_dir
save_path = args.save_path
suffix = args.suffix
if suffix != "":
    suffix = f"_{suffix}"

os.makedirs(save_path, exist_ok=True)
os.makedirs(check_point_dir, exist_ok=True)

event_dt_bar = 3 # s
dbscan_epsilon = 2.56
eps_xy = 20.0
eps_z = 8.0
time_space_weight = [1, dbscan_epsilon/eps_xy, dbscan_epsilon/eps_xy, dbscan_epsilon/eps_z]
#time_space_weight = [1, 3.3, 3.3, 1]
DT = 2.0 # s
MIN_STATION = 3
region="ridgecrest"
config={}
config[region]={
    "VPVS_RATIO": 1.73,
    "VP": 6.0,
}

sample_rate = 100
feature_scale = 16
degree2km = 111.19492664455873

# %%
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
ridgecrest_metric = EventMetric()
station_info = json.load(open(station_info_dir, "r"))
stations = pd.DataFrame(station_info).T
stations['station_id'] = stations.index
stations.reset_index(drop=True, inplace=True)

if not args.test_only:
    # 获取csv_dir下所有的CSV文件的路径
    pick_files = [os.path.join(pick_dir, f) for f in os.listdir(pick_dir) if f.endswith(".csv")]
    event_files = [os.path.join(event_dir, f) for f in os.listdir(event_dir) if f.endswith(".csv")]

    #dbscan = DBSCAN(eps=dbscan_epsilon, min_samples=1)
    dbscan = DBSCAN(eps=dbscan_epsilon, min_samples=3)

    events = []
    picks = []

    for pick in tqdm(pick_files):
        if os.path.getsize(pick) == 0:
            #ridgecrest_metric.empty_files += 1
            continue
        df = pd.read_csv(pick)
        picks.append(df)

    for pick in tqdm(event_files):
        # skip the empty pick file
        if os.path.getsize(pick) == 0:
            ridgecrest_metric.empty_files += 1
            continue
        df = pd.read_csv(pick)
        events.append(df)
        #df = df[df["phase_score"] >= args.score_threshold]
        #if len(df) == 0:
        #    ridgecrest_metric.empty_files += 1
        #    continue
        #
        #stations = df["station_id"].values
        #time_stamps = pd.to_datetime(df["event_original_time"])
        #assert type(time_stamps) == pd.core.series.Series
        #hypoparameters = np.zeros((len(df),9))
        #hypoparameters[:,0] = df["event_index"].values
        #hypoparameters[:,1] = df["event_location_x"].values
        #hypoparameters[:,2] = df["event_location_y"].values
        #hypoparameters[:,3] = df["event_location_z"].values
        #hypoparameters[:,4] = df["event_longitude"].values
        #hypoparameters[:,5] = df["event_latitude"].values
        #hypoparameters[:,6] = df["epicentral_distance"].values
        #hypoparameters[:,7] = df["azimuth"].values
        #hypoparameters[:,8] = df["magnitude"].values
        #
        #raw_picks.append(df[['event_longitude', 'event_latitude', 'event_location_x', 'event_location_y', 'event_location_z', 'magnitude']])
        #
        #labels = dbscan.fit_predict(hypoparameters[:,:4]*np.array(time_space_weight)[None,:]) # weighted
        #classes, counts = np.unique(labels, return_counts=True)
        #if len(df) > 5 and len(classes) > 1:
        #    if classes[0] == -1:
        #        classes = classes[1:]
        #        counts = counts[1:]
        #    classes = classes[counts>1]
        #    
        #for c in classes:
        #    cluster_picks = hypoparameters[labels==c]
        #    time_stamp = time_stamps[labels==c]
        #    time_stamp = time_stamp.iloc[np.argsort(cluster_picks[:,6])][:30]
        #    cluster_picks = cluster_picks[np.argsort(cluster_picks[:,6])][:80]
        #    info = {}
        #    info["event_id"] = f'{ridgecrest_metric.num_event_picks}'
        #    info["time"] = time_stamp.median()
        #    info["magnitude"] = np.median(cluster_picks[:10,8])
        #    #avg_center = average_location(cluster_picks[:,4:6])
        #    avg_center = np.median(cluster_picks[:10,4:6], axis=0)
        #    x_pred = avg_center[0]*np.cos(np.radians(avg_center[1])) * degree2km
        #    y_pred = avg_center[1] * degree2km
        #    z_pred = np.median(cluster_picks[:10,3])
        #    info["latitude"] = avg_center[1]
        #    info["longitude"] = avg_center[0]
        #    info["depth_km"] = z_pred
        #    info["x_km"] = x_pred
        #    info["y_km"] = y_pred
        #    info["z_km"] = z_pred
        #    
        #    events.append(pd.DataFrame(info, index=[0]))
        #    ridgecrest_metric.num_event_picks += 1

    picks = pd.concat(picks, ignore_index=True)
    events = pd.concat(events, ignore_index=True)
    picks['phase_time'] = pd.to_datetime(picks['phase_time'])
    events['event_original_time'] = pd.to_datetime(events['event_original_time'])
    events['phase_time'] = pd.to_datetime(events['phase_time'])
    picks = picks.sort_values(by="phase_time", ascending=True)
    events = events.sort_values(by="event_original_time", ascending=True)
    picks = picks[picks["phase_score"] >= args.score_threshold]
    events = events[events["phase_score"] >= args.score_threshold]

    time_stamp0 = min(picks["phase_time"].min(), events["event_original_time"].min()) - pd.Timedelta(seconds=10)
    events['timestamp'] = events['event_original_time'].apply(lambda x: (x - time_stamp0).total_seconds())
    events['timestamp_center'] = events['phase_time'].apply(lambda x: (x - time_stamp0).total_seconds())
    if "travel_time_s" not in events.columns:
        events['travel_time_s'] = events['timestamp_center'] - events['timestamp']
    picks['timestamp'] = picks['phase_time'].apply(lambda x: (x - time_stamp0).total_seconds())

    t0, t1 = min(events['timestamp'].min(), picks['timestamp'].min()), max(events['timestamp'].max(), picks['timestamp'].max())

    events['event_index'] = -1
    if args.cluster_method == "dbscan":
        event_indexes = dbscan.fit_predict(events[['timestamp', 'event_location_x', 'event_location_y', 'event_location_z']].values*np.array(time_space_weight)[None,:])
        events['event_index'] = event_indexes
    elif args.cluster_method == "time":
        t = np.arange(t0, t1, DT)
        hist, edge = np.histogram(events["timestamp"], bins=t, weights=events["phase_score"])
        xt = (edge[:-1] + edge[1:]) / 2  # center of the bin
        # hist_numpy = hist.copy()

        hist = torch.from_numpy(hist).float().unsqueeze(0).unsqueeze(0)
        hist_pool = F.max_pool1d(hist, kernel_size=3, padding=1, stride=1)
        mask = hist_pool == hist
        hist = hist * mask
        hist = hist.squeeze(0).squeeze(0)
        K = int((t[-1] - t[0]) / 5)  # assume max 1 event per 10 seconds on average
        topk_score, topk_index = torch.topk(hist, k=K)
        topk_index = topk_index[topk_score >= MIN_STATION]  # min 3 stations
        topk_score = topk_score[topk_score >= MIN_STATION]
        topk_index = topk_index.numpy()
        topk_score = topk_score.numpy()
        num_events = len(topk_index)
        t00 = xt[topk_index - 1]
        t11 = xt[topk_index + 1]
        timestamp = events["timestamp"].values
        for i in tqdm(range(num_events), desc="Assigning event index"):
            mask = (timestamp >= t00[i]) & (timestamp <= t11[i])
            events.loc[mask, "event_index"] = i
    else:
        raise ValueError(f"Cluster method {args.cluster_method} not supported")
    
    events['num_picks'] = events.groupby('event_index').transform('size')

    picks['event_index'] = -1
    picks.set_index('station_id', inplace=True)

    VP = config[region]["VP"]
    VPVS_RATIO = config[region]["VPVS_RATIO"]

    for group_id, event in tqdm(events.groupby("station_id"), desc="Linking picks to events"):
        # travel time tt = (tp + ts) / 2 = (1 + ps_ratio)/2 * tp => tp = tt * 2 / (1 + ps_ratio)
        # (ts - tp) = (ps_ratio - 1) tp = tt * 2 * (ps_ratio - 1) / (ps_ratio + 1)
        event = event.sort_values(by='num_picks', ascending=True)
        #ps_delta = event["travel_time_s"].values * 2 * (VPVS_RATIO - 1) / (VPVS_RATIO + 1)
        ps_delta = np.max(np.stack([event["ps_delta_s"].values, event["travel_time_s"].values * 2 * (VPVS_RATIO - 1) / (VPVS_RATIO + 1)], axis=0), axis=0)
        #t1 = event["timestamp_center"].values - np.min(np.stack([ps_delta*0.6, ps_delta*0.5 + 0.5], axis=0), axis=0) # the width is half ps_delta with a extension of 0.5s
        #t2 = event["timestamp_center"].values + np.min(np.stack([ps_delta*0.6, ps_delta*0.5 + 0.5], axis=0), axis=0)
        t1 = event["timestamp_center"].values - (ps_delta*0.6 + 1.0) # the width is half ps_delta with a extension of 0.5s
        t2 = event["timestamp_center"].values + (ps_delta*0.6 + 1.0)
        picks_ = picks.loc[group_id, 'timestamp'].values # (Npk, )
        mask = (picks_[None, :] >= t1[:, None]) & (picks_[None, :] <= t2[:, None])  # (Nev, Npk)
        # picks.loc[group_id, "event_index"] = np.where(
        #     mask.any(axis=0), index.values[mask.argmax(axis=0)], picks.loc[group_id, "event_index"]
        # )
        picks.loc[group_id, "event_index"] = np.where(
            mask.any(axis=0), event["event_index"].values[mask.argmax(axis=0)], -1
        )

    picks.reset_index(inplace=True)

    picks.drop(columns=['timestamp'], inplace=True)
    events.drop(columns=['timestamp', 'timestamp_center'], inplace=True)
    #events = events.merge(stations[['station_id', 'latitude', 'longitude']], on='station_id', how='left')
    events = events.groupby('event_index').agg({
        'event_original_time': 'median',
        'phase_score': 'sum',
        'event_latitude': 'median',
        'event_longitude': 'median',
        'event_location_x': 'median',
        'event_location_y': 'median',
        'event_location_z': 'median',
        'magnitude': 'median',
    }).reset_index()
    events.rename(columns={
        "event_original_time": "time",
        "event_latitude": "latitude",
        "event_longitude": "longitude",
        "event_location_x": "x_km",
        "event_location_y": "y_km",
        "event_location_z": "depth_km",
    }, inplace=True)
    events['z_km'] = events['depth_km']
    events = events[events['event_index'] != -1]

    print(f"Number of associated events: {len(events['event_index'].unique()):,}")
    print(f"Number of associated picks: {len(picks[picks['event_index'] != -1]):,} / {len(picks):,}")

    # add timezone info (UTC)
    picks['phase_time'] = picks['phase_time'].dt.tz_localize('UTC')
    events['time'] = events['time'].dt.tz_localize('UTC')
    picks['phase_time'] = picks['phase_time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    events['time'] = events['time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    events.to_csv(catalog_pred_dir.replace('catalog', 'events'), index=False)
    picks.to_csv(catalog_pred_dir.replace('catalog', 'picks'), index=False)

# %%

# 绘制直方图
# plt.figure(figsize=(9, 6), dpi=300)
# plt.bar(counts_pred.index.strftime('%Y-%m-%d-%H'), counts_pred.values, color='C0', label='Prediction')
# plt.bar(counts_catalog.index.strftime('%Y-%m-%d-%H'), counts_catalog.values, color='C3', label='SCSN')
# plt.xlim(left=4, right=len(counts_catalog))
# plt.xticks(ticks=np.arange(4, len(counts_catalog), 1), labels=[idx[2:] if idx.endswith('00') else '' for idx in counts_catalog.index.strftime('%Y-%m-%d-%H')[4:len(counts_catalog)]],
#            rotation=30, fontsize=8)
# plt.xlabel('Date')
# plt.ylabel('Frequency')
# plt.legend()
# if save_path is not None:
#     plt.savefig(save_path+f'catalog{suffix}.png')
# else:
#     plt.show()

# %%
catalog = pd.read_csv(catalog_dir)
catalog_pred = pd.read_csv(catalog_pred_dir.replace('catalog', 'events') if not args.test_only else catalog_pred_dir)

ridgecrest_metric.num_event_picks += len(catalog_pred)

catalog['time'] = pd.to_datetime(catalog['time'])
catalog = catalog.sort_values(by='time', ascending=True)
catalog_pred['time'] = pd.to_datetime(catalog_pred['time'], utc=True)
catalog_pred = catalog_pred.sort_values(by='time', ascending=True)

groups_pred = catalog_pred.groupby(pd.Grouper(key='time', freq='3h', origin=pd.to_datetime('2019-07-04 00:00:00.000000+00:00')))
counts_pred = groups_pred.size()

catalog['proj_x'] = catalog['longitude'] * degree2km * np.cos(np.radians(catalog['latitude']))
catalog['proj_y'] = catalog['latitude'] * degree2km
catalog_pred['x_km'] = catalog_pred['longitude'] * degree2km * np.cos(np.radians(catalog_pred['latitude']))
catalog_pred['y_km'] = catalog_pred['latitude'] * degree2km
catalog_pred['z_km'] = catalog_pred['depth_km']

# select rows with 'ci' in 'event_id'
catalog = catalog[catalog['event_id'].str.contains('ci')]

# %%
#iterrows_pred = catalog_pred.iterrows()
#
#for _, row in catalog.iterrows():
#    while True:
#        try:
#            index, row_pred = next(iterrows_pred)
#            time_diff = (row_pred['time'].timestamp() - row['time'].timestamp())
#            if time_diff > event_dt_bar:
#                # undo the next() call
#                iterrows_pred = itertools.chain([(index, row_pred)], iterrows_pred)
#                break
#            else:
#                x_diff = row_pred['x_km'] - row['proj_x']
#                y_diff = row_pred['y_km'] - row['proj_y']
#                z_diff = row_pred['z_km'] - row['z_km']
#                lon_diff = row_pred['longitude'] - row['longitude']
#                lat_diff = row_pred['latitude'] - row['latitude']
#                magnitude_diff = row_pred['magnitude'] - row['magnitude']
#                if np.isnan(magnitude_diff):
#                    print(row_pred['magnitude'], row['magnitude'], row['event_id'], index)
#                if (abs(time_diff)<=event_dt_bar) and (abs(x_diff)<30) and (abs(y_diff)<30) and (abs(z_diff)<10):
#                    ridgecrest_metric.update_matched_picks([time_diff], [x_diff], [y_diff], [z_diff],
#                                                            [row_pred['x_km']], [row_pred['y_km']], [row_pred['z_km']],
#                                                            [row['magnitude']], [row_pred['magnitude']],[],[],[],
#                                                            [row_pred['longitude']], [row_pred['latitude']], [row_pred['longitude'] - row['longitude']], [row_pred['latitude'] - row['latitude']])
#                    break
#        except StopIteration:
#            break

event_time_stamp0 = min(catalog['time'].min(), catalog_pred['time'].min()) - pd.Timedelta(seconds=10)
print(f"event_time_stamp0: {event_time_stamp0}, min_time: {catalog['time'].min()}, min_time_pred: {catalog_pred['time'].min()}")
catalog_pred['timestamp'] = catalog_pred['time'].apply(lambda x: (x - event_time_stamp0).total_seconds())
catalog['timestamp'] = catalog['time'].apply(lambda x: (x - event_time_stamp0).total_seconds())
event_params_pred = catalog_pred[['timestamp', 'x_km', 'y_km', 'z_km', 'longitude', 'latitude', 'depth_km', 'magnitude']].values
event_params = catalog[['timestamp', 'proj_x', 'proj_y', 'z_km', 'longitude', 'latitude', 'depth_km', 'magnitude']].values
event_location_pred = event_params_pred[:, :4].copy() # t, x, y, z
event_location = event_params[:, :4].copy()
event_location_pred[:, 3] = event_location_pred[:, 3] // 2
event_location[:, 3] = event_location[:, 3] // 2

avg_vel = 6.0 # km/s

matched_pred_idxs = []
temp = []
for i, gt_loc in enumerate(event_location):
    time_diff = event_params_pred[:, 0] - gt_loc[0]
    # get idxs with abs(time_diff) < 3s
    candidate_idxs = np.where(abs(time_diff) < event_dt_bar)[0]
    # remove the ones that are already matched
    candidate_idxs = np.setdiff1d(candidate_idxs, temp)
    if len(candidate_idxs) == 0:
        #matched_pred_idxs.append(np.argmin(abs(time_diff))) # get the closest one
        # get the closest one with abs(time_diff) > 3s
        candidate_idxs = np.where(abs(time_diff) >= event_dt_bar)[0]
        matched_pred_idxs.append(candidate_idxs[np.argmin(abs(time_diff[candidate_idxs]))])
        temp.append(-1)
        continue
    candidate_locs = event_location_pred[candidate_idxs]
    filter_idxs = np.where(np.all((np.abs(candidate_locs[:, 1:] - gt_loc[1:])-np.array([30, 30, 10//2])[None,:])<0, axis=1))[0] # diff of x, y, z < 30, 30, 10
    if len(filter_idxs) == 0:
        #matched_pred_idxs.append(np.argmin(abs(time_diff)))
        # get the closest one with abs(time_diff) > 3s
        candidate_idxs = np.where(abs(time_diff) >= event_dt_bar)[0]
        matched_pred_idxs.append(candidate_idxs[np.argmin(abs(time_diff[candidate_idxs]))])
        temp.append(-1)
        continue
    candidate_idxs = candidate_idxs[filter_idxs]
    candidate_locs = candidate_locs[filter_idxs]
    if not args.use_depth:
        dist = np.abs(candidate_locs[:, 0] - gt_loc[0]) * avg_vel + np.linalg.norm(candidate_locs[:, 1:3] - gt_loc[1:3], axis=1)
    else:
        dist = np.abs(candidate_locs[:, 0] - gt_loc[0]) * avg_vel + np.linalg.norm(candidate_locs[:, 1:] - gt_loc[1:], axis=1)
    matched_pred_idxs.append(candidate_idxs[np.argmin(dist)])
    temp.append(candidate_idxs[np.argmin(dist)])
    
matched_pred_idxs = np.array(matched_pred_idxs)
temp = np.array(temp)
assert np.all(matched_pred_idxs[temp>-1] == temp[temp>-1])
uniqued_matched_pred_idxs, idxs, counts = np.unique(temp[temp>-1], return_counts=True, return_index=True)
if len(uniqued_matched_pred_idxs) < len(temp[temp>-1]):
    duplicated = uniqued_matched_pred_idxs[counts>1]
    for i in duplicated:
        idx = np.where(matched_pred_idxs == i)[0]
        print(f"{catalog.iloc[idx]['event_id'].values} match the same prediction {i}")

matched_pred_params = event_params_pred[matched_pred_idxs]
time_diff, x_diff, y_diff, z_diff, lon_diff, lat_diff, _, magnitude_diff = matched_pred_params.T - event_params.T
nan_idxs = np.isnan(magnitude_diff)
if np.any(nan_idxs):
    for pred_mag, mag, event_id, index in zip(catalog_pred.iloc[matched_pred_idxs[nan_idxs]]['magnitude'], catalog.iloc[nan_idxs]['magnitude'], catalog.iloc[nan_idxs]['event_id'], matched_pred_idxs[nan_idxs]):
        print(f"event_id: {event_id}, pred_mag: {pred_mag}, mag: {mag}, index: {index}")
ridgecrest_metric.update_matched_picks(time_diff.tolist(), x_diff.tolist(), y_diff.tolist(), z_diff.tolist(),
                                        matched_pred_params[:, 1].tolist(), matched_pred_params[:, 2].tolist(), matched_pred_params[:, 3].tolist(),
                                        event_params[:, 7].tolist(), matched_pred_params[:, 7].tolist(), [], [], [],
                                        matched_pred_params[:, 4].tolist(), matched_pred_params[:, 5].tolist(), lon_diff.tolist(), lat_diff.tolist())

ridgecrest_metric.list2numpy()
print(f"empty files: {ridgecrest_metric.empty_files}")

# %%
assert ridgecrest_metric.num_event_picks == len(catalog_pred), f"predict catalog should have {len(catalog_pred)} events, but have {ridgecrest_metric.num_event_picks}"
ridgecrest_metric.num_event_picks_label = len(catalog)

ridgecrest_metric.calc_metrics()
if check_point_dir is not None:
    with open(f"{check_point_dir}/ridgecrest_metric{suffix}.pkl", "wb") as f:
        pickle.dump(ridgecrest_metric, f)
# %%
ridgecrest_metric.plot_residual(save_path=f"./ridgecrest_metrics/event_residual{suffix}.png")

# %%
valid_stations = stations[stations['station_id'].isin(picks['station_id'].unique())] if not args.test_only else stations

# %%
catalog = pd.read_csv(catalog_dir)
# sort by time, acending order
catalog = catalog.sort_values(by=['time'], ascending=True)
catalog['time'] = pd.to_datetime(catalog['time'])
# select rows with 'ci' in 'event_id'
catalog = catalog[catalog['event_id'].str.contains('ci')]
groups = catalog.groupby(pd.Grouper(key='time', freq='3h', origin=pd.to_datetime('2019-07-04 00:00:00.000000+00:00')))
counts_catalog = groups.size()
# %%
# plot the distribution of the event location
fig = plt.figure(figsize=(14, 24), dpi=300)
axes=[]
axes.append(plt.subplot2grid((39, 13), (0, 0), colspan=8, rowspan=11))
axes.append(plt.subplot2grid((39, 13), (0, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (6, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (13, 0), colspan=8, rowspan=11))
axes.append(plt.subplot2grid((39, 13), (13, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (19, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (26, 0), colspan=8, rowspan=11))
axes.append(plt.subplot2grid((39, 13), (26, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (32, 8), colspan=6, rowspan=5))

## whole
#axes[0].scatter(catalog_pred['longitude'], catalog_pred['latitude'], s=1, c='C0', label='Prediction')
axes[0].scatter(ridgecrest_metric.lon_pred, ridgecrest_metric.lat_pred, s=1, c='C0', label='Prediction')
axes[0].scatter(catalog['longitude'], catalog['latitude'], s=1, c='C3', label='SCSN')
axes[0].scatter(valid_stations['longitude'], valid_stations['latitude'], s=10, marker='^', c='k', label='Stations')
axes[0].set_xlim(-118.2, -117.0)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_aspect("equal")
axes[0].legend()

axes[1].scatter(catalog['longitude'], catalog['depth_km'], s=1, c='C3', label='SCSN')
#axes[1].scatter(catalog_pred['longitude'], catalog_pred['z_km'], s=1, c='C0', label='Prediction')
axes[1].scatter(ridgecrest_metric.lon_pred, ridgecrest_metric.z, s=1, c='C0', label='Prediction')
# reverse the y axis
axes[1].set_ylim(axes[1].get_ylim()[::-1])
axes[1].set_xlim(-118.2, -117.0)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Depth (km)')

axes[2].scatter(catalog['latitude'], catalog['depth_km'], s=1, c='C3', label='SCSN')
#axes[2].scatter(catalog_pred['latitude'], catalog_pred['z_km'], s=1, c='C0', label='Prediction')
axes[2].scatter(ridgecrest_metric.lat_pred, ridgecrest_metric.z, s=1, c='C0', label='Prediction')
# reverse the y axis
axes[2].set_ylim(axes[2].get_ylim()[::-1])
axes[2].set_xlabel('Latitude')
axes[2].set_ylabel('Depth (km)')

## prediction
axes[3].scatter(ridgecrest_metric.lon_pred, ridgecrest_metric.lat_pred, s=1, c='C0', label='Prediction')
axes[3].scatter(valid_stations['longitude'], valid_stations['latitude'], s=10, marker='^', c='k', label='Stations')
axes[3].set_xlim(-118.2, -117.0)
axes[3].set_xlabel('Longitude')
axes[3].set_ylabel('Latitude')
axes[3].set_aspect("equal")
axes[3].legend()

#axes[4].scatter(catalog_pred['longitude'], catalog_pred['z_km'], s=1, c='C0', label='Prediction')
axes[4].scatter(ridgecrest_metric.lon_pred, ridgecrest_metric.z, s=1, c='C0', label='Prediction')
# reverse the y axis
axes[4].set_ylim(axes[4].get_ylim()[::-1])
axes[4].set_xlim(-118.2, -117.0)
axes[4].set_xlabel('Longitude')
axes[4].set_ylabel('Depth (km)')

#axes[5].scatter(catalog_pred['latitude'], catalog_pred['z_km'], s=1, c='C0', label='Prediction')
axes[5].scatter(ridgecrest_metric.lat_pred, ridgecrest_metric.z, s=1, c='C0', label='Prediction')
# reverse the y axis
axes[5].set_ylim(axes[5].get_ylim()[::-1])
axes[5].set_xlabel('Latitude')
axes[5].set_ylabel('Depth (km)')

## SCSN
axes[6].scatter(catalog['longitude'], catalog['latitude'], s=1, c='C3', label='SCSN')
axes[6].scatter(valid_stations['longitude'], valid_stations['latitude'], s=10, marker='^', c='k', label='Stations')
axes[6].set_xlim(-118.2, -117.0)
axes[6].set_xlabel('Longitude')
axes[6].set_ylabel('Latitude')
axes[6].set_aspect("equal")
axes[6].legend()

axes[7].scatter(catalog['longitude'], catalog['depth_km'], s=1, c='C3', label='SCSN')
# reverse the y axis
axes[7].set_ylim(axes[7].get_ylim()[::-1])
axes[7].set_xlim(-118.2, -117.0)
axes[7].set_xlabel('Longitude')
axes[7].set_ylabel('Depth (km)')

axes[8].scatter(catalog['latitude'], catalog['depth_km'], s=1, c='C3', label='SCSN')
# reverse the y axis
axes[8].set_ylim(axes[8].get_ylim()[::-1])
axes[8].set_xlabel('Latitude')
axes[8].set_ylabel('Depth (km)')

if save_path is not None:
    plt.savefig(save_path+f'event_location{suffix}.png')
else:
    plt.show()

raw_picks = catalog_pred[catalog_pred['magnitude']>=0.0]
# %%
# plot the distribution of the event location (raw picks)
fig = plt.figure(figsize=(14, 24), dpi=300)
axes=[]
axes.append(plt.subplot2grid((39, 13), (0, 0), colspan=8, rowspan=11))
axes.append(plt.subplot2grid((39, 13), (0, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (6, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (13, 0), colspan=8, rowspan=11))
axes.append(plt.subplot2grid((39, 13), (13, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (19, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (26, 0), colspan=8, rowspan=11))
axes.append(plt.subplot2grid((39, 13), (26, 8), colspan=6, rowspan=5))
axes.append(plt.subplot2grid((39, 13), (32, 8), colspan=6, rowspan=5))

## whole
axes[0].scatter(raw_picks['longitude'], raw_picks['latitude'], s=1, c='C0', label='Prediction')
#axes[0].scatter(raw_picks['event_longitude'], raw_picks['event_latitude'], s=1, c='C0', label='Prediction')
axes[0].scatter(catalog['longitude'], catalog['latitude'], s=1, c='C3', label='SCSN')
axes[0].scatter(valid_stations['longitude'], valid_stations['latitude'], s=10, marker='^', c='k', label='Stations')
axes[0].set_xlim(-118.2, -117.0)
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
axes[0].set_aspect("equal")
axes[0].legend()

axes[1].scatter(catalog['longitude'], catalog['depth_km'], s=1, c='C3', label='SCSN')
axes[1].scatter(raw_picks['longitude'], raw_picks['z_km'], s=1, c='C0', label='Prediction')
#axes[1].scatter(raw_picks['event_longitude'], raw_picks['event_location_z'], s=1, c='C0', label='Prediction')
# reverse the y axis
axes[1].set_ylim(axes[1].get_ylim()[::-1])
axes[1].set_xlim(-118.2, -117.0)
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Depth (km)')

axes[2].scatter(catalog['latitude'], catalog['depth_km'], s=1, c='C3', label='SCSN')
axes[2].scatter(raw_picks['latitude'], raw_picks['z_km'], s=1, c='C0', label='Prediction')
#axes[2].scatter(raw_picks['event_latitude'], raw_picks['event_location_z'], s=1, c='C0', label='Prediction')
# reverse the y axis
axes[2].set_ylim(axes[2].get_ylim()[::-1])
axes[2].set_xlabel('Latitude')
axes[2].set_ylabel('Depth (km)')

## prediction
axes[3].scatter(raw_picks['longitude'], raw_picks['latitude'], s=1, c='C0', label='Prediction')
axes[3].scatter(valid_stations['longitude'], valid_stations['latitude'], s=10, marker='^', c='k', label='Stations')
axes[3].set_xlim(-118.2, -117.0)
axes[3].set_xlabel('Longitude')
axes[3].set_ylabel('Latitude')
axes[3].set_aspect("equal")
axes[3].legend()

axes[4].scatter(raw_picks['longitude'], raw_picks['z_km'], s=1, c='C0', label='Prediction')
#axes[4].scatter(raw_picks['event_longitude'], raw_picks['event_location_z'], s=1, c='C0', label='Prediction')
# reverse the y axis
axes[4].set_ylim(axes[4].get_ylim()[::-1])
axes[4].set_xlim(-118.2, -117.0)
axes[4].set_xlabel('Longitude')
axes[4].set_ylabel('Depth (km)')

axes[5].scatter(raw_picks['latitude'], raw_picks['z_km'], s=1, c='C0', label='Prediction')
#axes[5].scatter(raw_picks['event_latitude'], raw_picks['event_location_z'], s=1, c='C0', label='Prediction')
# reverse the y axis
axes[5].set_ylim(axes[5].get_ylim()[::-1])
axes[5].set_xlabel('Latitude')
axes[5].set_ylabel('Depth (km)')

## SCSN
axes[6].scatter(catalog['longitude'], catalog['latitude'], s=1, c='C3', label='SCSN')
axes[6].scatter(valid_stations['longitude'], valid_stations['latitude'], s=10, marker='^', c='k', label='Stations')
axes[6].set_xlim(-118.2, -117.0)
axes[6].set_xlabel('Longitude')
axes[6].set_ylabel('Latitude')
axes[6].set_aspect("equal")
axes[6].legend()

axes[7].scatter(catalog['longitude'], catalog['depth_km'], s=1, c='C3', label='SCSN')
# reverse the y axis
axes[7].set_ylim(axes[7].get_ylim()[::-1])
axes[7].set_xlim(-118.2, -117.0)
axes[7].set_xlabel('Longitude')
axes[7].set_ylabel('Depth (km)')

axes[8].scatter(catalog['latitude'], catalog['depth_km'], s=1, c='C3', label='SCSN')
# reverse the y axis
axes[8].set_ylim(axes[8].get_ylim()[::-1])
axes[8].set_xlabel('Latitude')
axes[8].set_ylabel('Depth (km)')

if save_path is not None:
    plt.savefig(save_path+f'event_location_raw{suffix}.png')
else:
    plt.show()

# %%
# plot the picks, magnitude and residual of origin time
fig = plt.figure(figsize=(14, 7), dpi=300)
axes=[]
axes.append(plt.subplot2grid((15, 15), (7, 0), colspan=8, rowspan=8))
axes.append(plt.subplot2grid((15, 15), (0, 9), colspan=7, rowspan=7))
axes.append(plt.subplot2grid((15, 15), (8, 9), colspan=7, rowspan=7))
axes.append(plt.subplot2grid((15, 15), (0, 0), colspan=4, rowspan=5))
axes.append(plt.subplot2grid((15, 15), (0, 4), colspan=4, rowspan=5))

axes[0].bar(counts_pred.index.strftime('%Y-%m-%d-%H'), counts_pred.values, color='C0', label=f'Prediction: {len(catalog_pred)}', alpha=0.7)
axes[0].bar(counts_catalog.index.strftime('%Y-%m-%d-%H'), counts_catalog.values, color='C3', label=f'SCSN: {len(catalog)}', alpha=0.7)
axes[0].set_xlim(left=4, right=len(counts_catalog))
axes[0].set_xticks(ticks=np.arange(4, len(counts_catalog), 1), labels=[idx[2:] if idx.endswith('00') else '' for idx in counts_catalog.index.strftime('%Y-%m-%d-%H')[4:len(counts_catalog)]],
           rotation=30, fontsize=8)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# axes[1].hist(catalog_pred['magnitude'], bins=26, range=(-0.5, 7), color='C0', label='Prediction', log=True)
# axes[1].hist(catalog['magnitude'], bins=26, range=(-0.5, 7), color='C3', label='SCSN', log=True)
# axes[1].hist(catalog_pred['magnitude'][catalog_pred['magnitude']>=1.81], bins=26, range=(-0.5, 7), color='C0', label='Prediction', log=True)
# 
# axes[1].set_xlabel('Magnitude')
# axes[1].set_ylabel('Frequency')
# axes[1].legend()
# 
# axes[2].hist(res_event_time, bins=11, range=(-3, 3))
# axes[2].set_xlabel('Original Time residual (s)')
# axes[2].set_ylabel('Frequency')
# 假设df1和df2是两个DataFrame，每个DataFrame都有'time'和'magnitude'两列

labels = ['SCSN', 'Prediction']
heats=[]
# 对于每个DataFrame，计算热图数据并绘制热图
for df in [catalog, catalog_pred]:
    df['time_bin'] = pd.cut(df['time'], bins=pd.date_range(start='2019-07-04 00:00:00+00:00', end='2019-07-10 00:00:00+00:00', freq='3h'), include_lowest=True)
    df['magnitude_bin'] = pd.cut(df['magnitude'], bins=np.arange(-1, 8, 0.25), include_lowest=True)
    heatmap_data = df.groupby(['magnitude_bin', 'time_bin'], observed=False).size().unstack()
    heats.append(np.log1p(heatmap_data))

axes[1]=sns.heatmap(heats[0], cmap=sns.light_palette("C3", as_cmap=True), ax=axes[1], label=labels[0], xticklabels=False, vmin=0, vmax=5)
axes[2]=sns.heatmap(heats[1], cmap=sns.light_palette("C0", as_cmap=True), ax=axes[2], label=labels[1], vmin=0, vmax=5)
for i, ax in enumerate(axes[1:3]):
    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels([r'$10^0$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'])
    cbar.set_label(f'Frequency')
    ax.set_xlim(left=4, right=len(counts_catalog))
    ax.set_ylabel('Magnitude')
    ax.set_yticks(ticks=np.arange(0, 36, 4))
    ax.set_yticklabels(np.arange(-1, 8, 1))
    #reverse the y axis
    ax.set_ylim(ax.get_ylim()[::-1])

axes[2].set_title('Event magnitude & time distribution\nSCSN (Upper)  Prediction (Lower)', y=1.00)
axes[1].set_xlabel('')
axes[2].set_xlabel('Date')
axes[2].set_xticks(ticks=np.arange(4, len(counts_catalog), 1), labels=[idx[2:] if idx.endswith('00') else '' for idx in counts_catalog.index.strftime('%Y-%m-%d-%H')[4:len(counts_catalog)]],
           rotation=30, fontsize=8)
#axes[2].set_title('Prediction')

axes[3].hist(ridgecrest_metric.time_res, bins=31, range=(-3, 3))
axes[3].set_xlabel('Event origin time residual (s)')
axes[3].set_ylabel('Frequency')
#axes[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=1, fancybox=True, labels=['Prediction', 'SCSN'])
axes[4].hist(catalog_pred['magnitude'], bins=41, range=(-1, 7), color='C0', label='Prediction', alpha=0.7)
axes[4].hist(catalog['magnitude'], bins=41, range=(-1, 7), color='C3', label='SCSN', alpha=0.7)
axes[4].set_xlabel('Magnitude')
axes[4].set_ylabel('Frequency')
axes[4].legend()
axes[4].set_xlim(-1, 7)
axes[4].set_yscale('log')
plt.tight_layout()
if save_path is not None:
    plt.savefig(save_path+f'catalog{suffix}.png')
else:
    plt.show()
