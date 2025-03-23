# %%
import csv
import json
import os
import sys
import re
import errno
import pickle
import h5py
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import obspy
from collections import OrderedDict, defaultdict, deque
from tqdm import tqdm
import itertools
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

import datasets
import torch
import torch.nn as nn
import torch.functional as F

sns.set_theme(style="white")
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
parser.add_argument('--score_threshold', type=float, default=0.3, help='the threshold of phase score')
parser.add_argument('--polarity_threshold', type=float, default=0.3, help='the threshold of polarity score')
parser.add_argument('--save_path', type=str, default=None, help='the path to save the metric pickle file')
parser.add_argument('--main_event_only', action='store_true', help='exclude the picks not in the main event duration')
parser.add_argument('--filter_events', type=str, default=None, help='the path to the file containing the events to be filtered out')
parser.add_argument('--include_events', action='store_true', help='exclude the other events instead of excluding the specified events')

args = parser.parse_args()
h5_file = args.h5_file
pick_dir = args.pick_dir
event_dir = args.event_dir
csv_file = f"{h5_file.split('.')[0]}.csv"
suffix = args.suffix
if suffix != "":
    suffix = f"_{suffix}"

filter_events = json.load(open(args.filter_events)) if args.filter_events is not None else dict()
    
os.makedirs("./metrics", exist_ok=True)
if args.save_path is not None:
    os.makedirs(args.save_path, exist_ok=True)

dt_bar = 0.5 # s
distribution_bar = 0.5 # s
station_threshold = 10
event_dt_bar = 3 # s
dbscan_epsilon = 16
time_space_weight = [1, 0.8, 0.8, 2]

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
        self.p_trace_stat = []
        self.s_trace_stat = []
        self.p_trace_stat_label = []
        self.s_trace_stat_label = []

        self.time_res_p = []
        self.time_res_s = []
        self.magnitude_p = []
        self.snr_p = []
        self.epicenter_distance_p = []
        self.azimuth_p = []
        self.magnitude_s = []
        self.snr_s = []
        self.epicenter_distance_s = []
        self.azimuth_s = []
        self.scores_p = []
        self.scores_s = []
        self.polarity_score_p = []
        self.polarity_score_s = []
        self.polarity_pred_p = []
        self.polarity_pred_s = []
        self.polarity_manual_p = []
        self.polarity_manual_s = []

        self.empty_files = 0
        
        self._sample_rate=sample_rate
    
    def list2numpy(self):
        self.time_res_p = np.array(self.time_res_p)/self._sample_rate
        self.time_res_s = np.array(self.time_res_s)/self._sample_rate
        self.magnitude_p = np.array(self.magnitude_p)
        self.snr_p = np.array(self.snr_p)
        self.epicenter_distance_p =  np.array(self.epicenter_distance_p)
        self.azimuth_p = np.array(self.azimuth_p)
        self.magnitude_s = np.array(self.magnitude_s)
        self.snr_s = np.array(self.snr_s)
        self.epicenter_distance_s = np.array(self.epicenter_distance_s)
        self.azimuth_s = np.array(self.azimuth_s)
        self.scores_p = np.array(self.scores_p)
        self.scores_s = np.array(self.scores_s)
        self.polarity_score_p = np.array(self.polarity_score_p)
        self.polarity_score_s = np.array(self.polarity_score_s)
        self.polarity_pred_p = np.array(self.polarity_pred_p)
        self.polarity_pred_s = np.array(self.polarity_pred_s)
        self.polarity_manual_p = np.array(self.polarity_manual_p)
        self.polarity_manual_s = np.array(self.polarity_manual_s)
        assert all([len(self.p_trace_stat)==self.num_p_picks, len(self.s_trace_stat)==self.num_s_picks, len(self.p_trace_stat_label)==self.num_p_picks_label, len(self.s_trace_stat_label)==self.num_s_picks_label]), f"trace stat length not match, {len(self.p_trace_stat) = }, {self.num_p_picks = }, {len(self.s_trace_stat) = }, {self.num_s_picks = }, {len(self.p_trace_stat_label) = }, {self.num_p_picks_label = }, {len(self.s_trace_stat_label) = }, {self.num_s_picks_label = }"
        self.p_trace_stat = pd.DataFrame(self.p_trace_stat, columns=["trace_id", "snr", "magnitude", "distance", "azimuth"])
        self.s_trace_stat = pd.DataFrame(self.s_trace_stat, columns=["trace_id", "snr", "magnitude", "distance", "azimuth"])
        self.p_trace_stat_label = pd.DataFrame(self.p_trace_stat_label, columns=["trace_id", "snr", "magnitude", "distance", "azimuth"])
        self.s_trace_stat_label = pd.DataFrame(self.s_trace_stat_label, columns=["trace_id", "snr", "magnitude", "distance", "azimuth"])
        
        return
        
    def update_matched_picks(self, min_res_p, min_res_s, min_scores_p, min_scores_s, magnitude, distance_km, snr_sta, azimuth, polarity_score_p=None, polarity_score_s=None, polarity_pred_p=None, polarity_pred_s=None, polarity_manual_p=None, polarity_manual_s=None):
        self.time_res_p+=min_res_p
        self.time_res_s+=min_res_s
        self.scores_p+=min_scores_p
        self.scores_s+=min_scores_s
        self.magnitude_p+=magnitude*len(min_res_p)
        self.magnitude_s+=magnitude*len(min_res_s)
        self.epicenter_distance_p+=distance_km*len(min_res_p)
        self.epicenter_distance_s+=distance_km*len(min_res_s)
        self.azimuth_p+=azimuth*len(min_res_p)
        self.azimuth_s+=azimuth*len(min_res_s)
        self.snr_p+=snr_sta*len(min_res_p)
        self.snr_s+=snr_sta*len(min_res_s)
        self.polarity_score_p+=polarity_score_p if polarity_score_p is not None else [0]*len(min_res_p)
        self.polarity_score_s+=polarity_score_s if polarity_score_s is not None else [0]*len(min_res_s)
        self.polarity_pred_p+=polarity_pred_p if polarity_pred_p is not None else ['N']*len(min_res_p)
        self.polarity_pred_s+=polarity_pred_s if polarity_pred_s is not None else ['N']*len(min_res_s)
        self.polarity_manual_p+=polarity_manual_p if polarity_manual_p is not None else ['N']*len(min_res_p)
        self.polarity_manual_s+=polarity_manual_s if polarity_manual_s is not None else ['N']*len(min_res_s)

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
        self.tp_p_azimuth = self.azimuth_p[index_p]
        self.tp_s_azimuth = self.azimuth_s[index_s]
        self.tp_p_snr = self.snr_p[index_p]
        self.tp_s_snr = self.snr_s[index_s]
        self.tp_p_scores = self.scores_p[index_p]
        self.tp_s_scores = self.scores_s[index_s]
        self.tp_p_polarity_score = self.polarity_score_p[index_p]
        self.tp_s_polarity_score = self.polarity_score_s[index_s]
        self.tp_p_polarity_pred = self.polarity_pred_p[index_p]
        self.tp_s_polarity_pred = self.polarity_pred_s[index_s]
        self.tp_p_polarity_manual = self.polarity_manual_p[index_p]
        self.tp_s_polarity_manual = self.polarity_manual_s[index_s]

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
    
    def plot_residual(self, save_path=None, info_only=False):
        print(f"p_picks: {self.num_p_picks}, s_picks: {self.num_s_picks}, p_picks_label: {self.num_p_picks_label}, s_picks_label: {self.num_s_picks_label}")
        print(f"TP_p: {self.metric_dict['true_positives_p']}, TP_s: {self.metric_dict['true_positives_s']}, FP_p: {self.metric_dict['false_positives_p']}, FP_s: {self.metric_dict['false_positives_s']}, FN_p: {self.metric_dict['false_negatives_p']}, FN_s: {self.metric_dict['false_negatives_s']}")
        print(f"precision_p: {self.metric_dict['precision_p']:.4f}, recall_p: {self.metric_dict['recall_p']:.4f}, f1_p: {self.metric_dict['f1_p']:.4f}")
        print(f"precision_s: {self.metric_dict['precision_s']:.4f}, recall_s: {self.metric_dict['recall_s']:.4f}, f1_s: {self.metric_dict['f1_s']:.4f}")
        print(f"miu_p: {self.metric_dict['miu_p']}, sigma_p: {self.metric_dict['sigma_p']}")
        print(f"miu_s: {self.metric_dict['miu_s']}, sigma_s: {self.metric_dict['sigma_s']}")
        
        if info_only:
            return
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.hist(self.res_p, bins=11, range=(-distribution_bar, distribution_bar))
        plt.title("P phase")
        plt.subplot(122)
        plt.hist(self.res_s, bins=11, range=(-distribution_bar, distribution_bar))
        plt.title("S phase")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].scatter(self.tp_p_picks, self.tp_p_scores, s=10, c='orange', label='TP_P')
        ax[0].set_xlabel('pred-label time residual [s]')
        ax[0].set_ylabel('phase score')
        ax[1].scatter(self.tp_s_picks, self.tp_s_scores, s=10, c='#43BEFF', label='TP_S')
        ax[1].set_xlabel('pred-label time residual [s]')
        plt.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path.replace("phase_picking", "phase_score"))
        else:
            plt.show()
        plt.close()
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].hist(self.tp_p_scores, bins=41, range=(0, 1), color='orange', alpha=0.7, label='TP_P')
        ax[0].set_xlabel('phase score')
        ax[0].set_ylabel('count')
        ax[1].hist(self.tp_s_scores, bins=41, range=(0, 1), color='#43BEFF', alpha=0.7, label='TP_S')
        ax[1].set_xlabel('phase score')
        plt.legend()
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path.replace("phase_picking", "phase_score_hist"))
        else:
            plt.show()
        plt.close()
        return
    
    def plot_polarity(self, save_path=None):
        # 构建混淆矩阵
        conf_matrix_p = confusion_matrix(self.polarity_manual_p, self.polarity_pred_p, labels=["D", "N", "U"])[::-1]
        conf_matrix_s = confusion_matrix(self.polarity_manual_s, self.polarity_pred_s, labels=["D", "N", "U"])[::-1]

        # 创建热力图
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
        axes[0].set_title("P phase")
        sns.heatmap(
            conf_matrix_p, annot=True, fmt="d", cmap="Blues", linewidths=1.0,# linecolor="gray",
            xticklabels=["D", "N", "U"], yticklabels=["U", "N", "D"], cbar=False, ax=axes[0]
        )
        # 添加百分比文本
        conf_percentages_p = conf_matrix_p / conf_matrix_p.sum(axis=1, keepdims=True) * 100
        for i in range(conf_matrix_p.shape[0]):
            for j in range(conf_matrix_p.shape[1]):
                axes[0].text(j + 0.5, i + 0.6, f"{conf_percentages_p[i, j]:.2f}%", 
                        ha='center', va='center', fontsize=8, color='black' if conf_percentages_p[i, j]<50 else 'white')
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Manual")

        axes[1].set_title("S phase")
        sns.heatmap(
            conf_matrix_s, annot=True, fmt="d", cmap="Blues", linewidths=1.0,# linecolor="gray",
            xticklabels=["D", "N", "U"], yticklabels=["U", "N", "D"], cbar=False, ax=axes[1]
        )
        # 添加百分比文本
        conf_percentages_s = conf_matrix_s / conf_matrix_s.sum(axis=1, keepdims=True) * 100
        for i in range(conf_matrix_s.shape[0]):
            for j in range(conf_matrix_s.shape[1]):
                axes[1].text(j + 0.5, i + 0.6, f"{conf_percentages_s[i, j]:.2f}%", 
                        ha='center', va='center',fontsize=8, color='black' if conf_percentages_p[i, j]<50 else 'white')
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Manual")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        
    def calc_metric_vs_params(self, param, bins=None, save_path=None):
        #plot residual vs params, p&s picks residual, polarity confusion matrix if (has_polarity)
        num_p_picks_bin = np.histogram(self.p_trace_stat[param], bins=bins)[0]
        num_s_picks_bin = np.histogram(self.s_trace_stat[param], bins=bins)[0]
        num_p_picks_label_bin = np.histogram(self.p_trace_stat_label[param], bins=bins)[0]
        num_s_picks_label_bin = np.histogram(self.s_trace_stat_label[param], bins=bins)[0]
        tp_p_bin, tp_s_bin, fp_p_bin, fp_s_bin, fn_p_bin, fn_s_bin = [], [], [], [], [], []
        precision_p_bin, precision_s_bin, recall_p_bin, recall_s_bin, f1_p_bin, f1_s_bin = [], [], [], [], [], []
        
        for i, (l, r) in enumerate(zip(bins[:-1], bins[1:])):
            tp_p_picks = self.tp_p_picks[np.logical_and(eval(f"self.tp_p_{param}") >= l, eval(f"self.tp_p_{param}") < r)]
            tp_s_picks = self.tp_s_picks[np.logical_and(eval(f"self.tp_s_{param}") >= l, eval(f"self.tp_s_{param}") < r)]
            tp_p_params = eval(f"self.tp_p_{param}")[np.logical_and(eval(f"self.tp_p_{param}") >= l, eval(f"self.tp_p_{param}") < r)]
            tp_s_params = eval(f"self.tp_s_{param}")[np.logical_and(eval(f"self.tp_s_{param}") >= l, eval(f"self.tp_s_{param}") < r)]
            
            tp_p_per_bin, tp_s_per_bin = len(tp_p_picks), len(tp_s_picks)
            fp_p_per_bin, fp_s_per_bin = num_p_picks_bin[i] - tp_p_per_bin, num_s_picks_bin[i] - tp_s_per_bin
            fn_p_per_bin, fn_s_per_bin = num_p_picks_label_bin[i] - tp_p_per_bin, num_s_picks_label_bin[i] - tp_s_per_bin
            
            precision_p_per_bin = tp_p_per_bin / (tp_p_per_bin + fp_p_per_bin) if tp_p_per_bin + fp_p_per_bin > 0 else 0
            precision_s_per_bin = tp_s_per_bin / (tp_s_per_bin + fp_s_per_bin) if tp_s_per_bin + fp_s_per_bin > 0 else 0
            recall_p_per_bin = tp_p_per_bin / (tp_p_per_bin + fn_p_per_bin) if tp_p_per_bin + fn_p_per_bin > 0 else 0
            recall_s_per_bin = tp_s_per_bin / (tp_s_per_bin + fn_s_per_bin) if tp_s_per_bin + fn_s_per_bin > 0 else 0
            f1_p_per_bin = 2 * precision_p_per_bin * recall_p_per_bin / (precision_p_per_bin + recall_p_per_bin) if precision_p_per_bin + recall_p_per_bin > 0 else 0
            f1_s_per_bin = 2 * precision_s_per_bin * recall_s_per_bin / (precision_s_per_bin + recall_s_per_bin) if precision_s_per_bin + recall_s_per_bin > 0 else 0

            tp_p_bin.append(tp_p_per_bin); tp_s_bin.append(tp_s_per_bin)
            fp_p_bin.append(fp_p_per_bin); fp_s_bin.append(fp_s_per_bin)
            fn_p_bin.append(fn_p_per_bin); fn_s_bin.append(fn_s_per_bin)
            precision_p_bin.append(precision_p_per_bin); precision_s_bin.append(precision_s_per_bin)
            recall_p_bin.append(recall_p_per_bin); recall_s_bin.append(recall_s_per_bin)
            f1_p_bin.append(f1_p_per_bin); f1_s_bin.append(f1_s_per_bin)
            
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        x = np.arange(len(bins[:-1]))
        labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins) - 1)]
        width = 0.25
        axes[0].bar(x - width, precision_p_bin, width=width, label='precision_p')
        axes[0].bar(x, recall_p_bin, width=width, label='recall_p')
        axes[0].bar(x + width, f1_p_bin, width=width, label='f1_p')
        for bar in axes[0].patches:
            height = bar.get_height()
            axes[0].annotate(f'{height:.2f}', 
                            xy=(bar.get_x() + bar.get_width() / 2, height), 
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
        axes[0].set_xlabel(param)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].legend(loc='lower right')
        axes[0].set_ylim(0.9, 1.0)
        axes[1].bar(x - width,precision_s_bin, width=width, label='precision_s')
        axes[1].bar(x, recall_s_bin, width=width, label='recall_s')
        axes[1].bar(x + width,f1_s_bin, width=width, label='f1_s')
        axes[1].set_xlabel(param)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].legend(loc='lower right')
        axes[1].set_ylim(0.9, 1.0)
        for bar in axes[1].patches:
            height = bar.get_height()
            axes[1].annotate(f'{height:.2f}', 
                            xy=(bar.get_x() + bar.get_width() / 2, height), 
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        return precision_p_bin, precision_s_bin, recall_p_bin, recall_s_bin, f1_p_bin, f1_s_bin, tp_p_bin, tp_s_bin, fp_p_bin, fp_s_bin, fn_p_bin, fn_s_bin

# %%
pick_metric = PickMetric(sample_rate=sample_rate)
pick_metric_thld = PickMetric(sample_rate=sample_rate)

# %%
true_pick_files = 0
fn_p_trace = []
fn_s_trace = []
with h5py.File(h5_file, "r") as fp:
    for pick in tqdm(pick_files):
        if args.model == "eqnet":
            event_id = re.sub(r'\.csv$', '', os.path.basename(pick))
        elif args.model == "phasenet_plus":
            event_id = pick.split('/')[-1]

        if len(filter_events) > 0:
            if (event_id in filter_events.keys()) != args.include_events:
                continue
        true_pick_files += 1
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
        
        if args.model == "eqnet" and not empty:
            df = pd.read_csv(pick)
        elif args.model == "phasenet_plus" and not empty:
            dfs = []
            stations=[os.path.join(pick, f) for f in os.listdir(pick) if f.endswith(".csv")]
            for sta in stations:
                if os.path.getsize(sta) == 0:
                    continue
                dfs.append(pd.read_csv(sta))
            df = pd.concat(dfs)
            
        df = df[df["phase_score"] >= args.score_threshold]
        if len(df) == 0:
            empty = True
        
        station_ids = list(event.keys()) if len(filter_events) == 0 or (not args.include_events) else list(set(filter_events[event_id]).intersection(set(event.keys())))
        if empty:
            pick_metric.empty_files += 1
            pick_metric_thld.empty_files +=1 if num_stations>=station_threshold else 0
            
            for sta_id in station_ids:
                phase_type_gt = event[sta_id].attrs["phase_type"]
                phase_index_gt = event[sta_id].attrs["phase_index"]
                try:
                    if len(phase_type_gt)==1 or event[sta_id].attrs["phase_status"]=='automatic':
                        continue
                except:
                    pass
                p_picks_gt, s_picks_gt = [], []
                e_ids = event[sta_id].attrs["event_id"]
                for e_id in set(e_ids): # exclude single p event in more than one event case
                    if len(set(event[sta_id].attrs["phase_type"][e_ids==e_id])) < 2:
                        continue
                    p_picks_gt.append(phase_index_gt[np.logical_and(e_ids==e_id, phase_type_gt=='P')][0])
                    s_picks_gt.append(phase_index_gt[np.logical_and(e_ids==e_id, phase_type_gt=='S')][0])
                pick_metric.num_p_picks_label+=len(p_picks_gt)
                pick_metric.num_s_picks_label+=len(s_picks_gt)
                pick_metric.p_trace_stat_label.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(p_picks_gt))
                pick_metric.s_trace_stat_label.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(s_picks_gt))
                pick_metric_thld.num_p_picks_label += len(p_picks_gt) if num_stations>=station_threshold else 0
                pick_metric_thld.num_s_picks_label += len(s_picks_gt) if num_stations>=station_threshold else 0
                pick_metric_thld.p_trace_stat_label.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(p_picks_gt) if num_stations>=station_threshold else [])
                pick_metric_thld.s_trace_stat_label.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(s_picks_gt) if num_stations>=station_threshold else [])
            fn_p_trace.append([f"{event_id}/{sta_id}" for sta_id in station_ids if len(set(event[sta_id].attrs["phase_type"]))>1])
            fn_s_trace.append([f"{event_id}/{sta_id}" for sta_id in station_ids if len(set(event[sta_id].attrs["phase_type"]))>1])
            continue
        
        df = df.sort_values(by=["station_id", "phase_index"], ignore_index=True)
        if 'phase_polarity' not in df.columns:
            df['phase_polarity'] = 0
        # map: polarity_score -> polarity_pred: score<-polarity_thld -> D,  -polarity_thld<score<polarity_thld -> N,  polarity_thld<score -> U
        df['phase_polarity_pred'] = df['phase_polarity'].apply(lambda x: 'D' if x < -args.polarity_threshold else 'U' if x > args.polarity_threshold else 'N')
        
        for sta_id in station_ids:
            phase_type_gt = event[sta_id].attrs["phase_type"]
            phase_index_gt = event[sta_id].attrs["phase_index"]
            phase_polarity_gt = event[sta_id].attrs["phase_polarity"]
            
            try:
                if len(phase_type_gt)==1 or event[sta_id].attrs["phase_status"]=='automatic':
                    continue
            except:
                pass
            
            p_picks_gt, s_picks_gt = [], []
            p_pol_gt, s_pol_gt = [], []
            e_ids = event[sta_id].attrs["event_id"]
            for e_id in set(e_ids): # exclude single p event in more than one event case
                if len(set(event[sta_id].attrs["phase_type"][e_ids==e_id])) < 2:
                    continue
                p_picks_gt.append(phase_index_gt[np.logical_and(e_ids==e_id, phase_type_gt=='P')][0])
                s_picks_gt.append(phase_index_gt[np.logical_and(e_ids==e_id, phase_type_gt=='S')][0])
                p_pol_gt.append(phase_polarity_gt[np.logical_and(e_ids==e_id, phase_type_gt=='P')][0])
                s_pol_gt.append(phase_polarity_gt[np.logical_and(e_ids==e_id, phase_type_gt=='S')][0])
            p_picks_gt = np.array(p_picks_gt)
            s_picks_gt = np.array(s_picks_gt)
            p_pol_gt = np.array(p_pol_gt)
            s_pol_gt = np.array(s_pol_gt)
            assert len(p_picks_gt) == len(p_pol_gt), f"p_picks_gt and p_pol_gt have different length in {event_id} {sta_id}"
            assert len(s_picks_gt) == len(s_pol_gt), f"s_picks_gt and s_pol_gt have different length in {event_id} {sta_id}"
            pick_metric.num_p_picks_label+=len(p_picks_gt)
            pick_metric.num_s_picks_label+=len(s_picks_gt)
            pick_metric.p_trace_stat_label.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(p_picks_gt))
            pick_metric.s_trace_stat_label.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(s_picks_gt))    
            pick_metric_thld.num_p_picks_label += len(p_picks_gt) if num_stations>=station_threshold else 0
            pick_metric_thld.num_s_picks_label += len(s_picks_gt) if num_stations>=station_threshold else 0
            pick_metric_thld.p_trace_stat_label.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(p_picks_gt) if num_stations>=station_threshold else [])
            pick_metric_thld.s_trace_stat_label.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(s_picks_gt) if num_stations>=station_threshold else [])
            if sta_id not in df["station_id"].values:
                continue
            if len(p_picks_gt) == 0 or len(s_picks_gt) == 0:
                continue
            
            df_sta = df[df["station_id"] == sta_id]
            if args.main_event_only:
                duration = np.stack([p_picks_gt-5*sample_rate, p_picks_gt+3*(s_picks_gt-p_picks_gt)], axis=1)
                duration = np.clip(duration, 0, 12000) #NOTE hard code the duration
                mask = np.zeros(len(df_sta), dtype=bool)
                for i, (p, s) in enumerate(duration):
                    mask = np.logical_or(mask, np.logical_and(df_sta["phase_index"]>=p, df_sta["phase_index"]<=s))
                df_sta = df_sta[mask]
            df_sta_p = df_sta[df_sta["phase_type"] == "P"]
            df_sta_s = df_sta[df_sta["phase_type"] == "S"]
            pick_metric.num_p_picks += len(df_sta_p)
            pick_metric.num_s_picks += len(df_sta_s)
            pick_metric.p_trace_stat.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(df_sta_p))
            pick_metric.s_trace_stat.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(df_sta_s))
            pick_metric_thld.num_p_picks += len(df_sta_p) if num_stations>=station_threshold else 0
            pick_metric_thld.num_s_picks += len(df_sta_s) if num_stations>=station_threshold else 0
            pick_metric_thld.p_trace_stat.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(df_sta_p) if num_stations>=station_threshold else [])
            pick_metric_thld.s_trace_stat.extend([{"trace_id": f"{event_id}/{sta_id}", "snr": event[sta_id].attrs["snr"].max(), "magnitude": event.attrs["magnitude"], "distance": event[sta_id].attrs["distance_km"], "azimuth": event[sta_id].attrs["azimuth"]}]*len(df_sta_s) if num_stations>=station_threshold else [])
            if len(df_sta_p) == 0:
                min_res_p = [np.inf] * len(p_picks_gt)
                min_scores_p = [np.inf] * len(p_picks_gt)
                min_polarity_score_p = [0] * len(p_picks_gt)
                min_polarity_pred_p = ['N'] * len(p_picks_gt)
                min_polarity_manual_p = list(p_pol_gt)
                fn_p_trace.append(f"{event_id}/{sta_id}")
            else:
                diff_p = df_sta_p["phase_index"].values[:, None] - p_picks_gt[None, :]
                index_p = np.argmin(np.abs(diff_p), axis=0)
                #assert len(np.unique(index_p)) == len(index_p), f"duplicated matched p picks in {event_id} {sta_id} {phase_type_gt}"
                #if len(np.unique(index_p)) != len(index_p):
                #    print(f"duplicated matched p picks in {event_id} {sta_id} {phase_type_gt} {len(df_sta_p)} {index_p}")
                min_res_p = list(diff_p[index_p, np.arange(len(index_p))])
                if len(index_p[np.abs(min_res_p)<50]) != len(set(index_p[np.abs(min_res_p)<50])):
                    print(f"duplicated matched p picks in {event_id} {sta_id} {phase_type_gt} {p_picks_gt} {df_sta_p['phase_index'].values}")
                min_scores_p = list(df_sta_p["phase_score"].values[index_p])
                min_polarity_score_p = list(df_sta_p["phase_polarity"].values[index_p])
                min_polarity_pred_p = list(df_sta_p["phase_polarity_pred"].values[index_p])
                assert len(index_p) == len(p_pol_gt), f"index_p and p_picks_gt have different length in {event_id} {sta_id}"
                min_polarity_manual_p = list(p_pol_gt)
                
                #if np.abs(min_res_p[list(set(e_ids)).index(event_id)]) > dt_bar:
                if np.abs(min_res_p).max()/sample_rate > dt_bar:
                    fn_p_trace.append(f"{event_id}/{sta_id}")
            if len(df_sta_s) == 0:
                min_res_s = [np.inf] * len(s_picks_gt)
                min_scores_s = [np.inf] * len(s_picks_gt)
                min_polarity_score_s = [0] * len(s_picks_gt)
                min_polarity_pred_s = ['N'] * len(s_picks_gt)
                min_polarity_manual_s = list(s_pol_gt)
                fn_s_trace.append(f"{event_id}/{sta_id}")
            else:
                diff_s = df_sta_s["phase_index"].values[:, None] - s_picks_gt[None, :]
                index_s = np.argmin(np.abs(diff_s), axis=0)
                #assert len(np.unique(index_s)) == len(index_s), f"duplicated matched s picks in {event_id} {sta_id} {phase_type_gt}"
                #if len(np.unique(index_s)) != len(index_s):
                #    print(f"duplicated matched s picks in {event_id} {sta_id} {phase_type_gt} {len(df_sta_s)} {index_s}")
                min_res_s = list(diff_s[index_s, np.arange(len(index_s))])
                min_scores_s = list(df_sta_s["phase_score"].values[index_s])
                min_polarity_score_s = list(df_sta_s["phase_polarity"].values[index_s])
                min_polarity_pred_s = list(df_sta_s["phase_polarity_pred"].values[index_s])
                assert len(index_s) == len(s_pol_gt), f"index_s and s_picks_gt have different length in {event_id} {sta_id}"
                min_polarity_manual_s = list(s_pol_gt)
                #if np.abs(min_res_s[list(set(e_ids)).index(event_id)]) > dt_bar:
                if np.abs(min_res_s).max()/sample_rate > dt_bar:
                    fn_s_trace.append(f"{event_id}/{sta_id}")
                
            snr_sta = event[sta_id].attrs["snr"]
            snr_sta = snr_sta[snr_sta > 0]
            if len(snr_sta) > 0:
                #snr_sta = snr_sta.mean()
                snr_sta = snr_sta.max()
            else:
                snr_sta = 0
            pick_metric.update_matched_picks(min_res_p, min_res_s, min_scores_p, min_scores_s, [event.attrs["magnitude"]], [event[sta_id].attrs["distance_km"]], [snr_sta], [event[sta_id].attrs["azimuth"]], min_polarity_score_p, min_polarity_score_s, min_polarity_pred_p, min_polarity_pred_s, min_polarity_manual_p, min_polarity_manual_s)
            if num_stations>=station_threshold:
                pick_metric_thld.update_matched_picks(min_res_p, min_res_s, min_scores_p, min_scores_s, [event.attrs["magnitude"]], [event[sta_id].attrs["distance_km"]], [snr_sta], [event[sta_id].attrs["azimuth"]], min_polarity_score_p, min_polarity_score_s, min_polarity_pred_p, min_polarity_pred_s, min_polarity_manual_p, min_polarity_manual_s)

pick_metric.list2numpy()
pick_metric_thld.list2numpy()
pick_metric.fn_p_trace = fn_p_trace
pick_metric.fn_s_trace = fn_s_trace

print(f"whole dataset empty files: {pick_metric.empty_files}")
print(f"num_station >= {station_threshold} subset empty files: {pick_metric_thld.empty_files}")

# %%
pick_metric.calc_metrics(dt_bar=dt_bar, distribution_bar=distribution_bar)
pick_metric_thld.calc_metrics(dt_bar=dt_bar, distribution_bar=distribution_bar)

# %%
if args.save_path is not None:
    with open(f"{args.save_path}/pick_metric_score_thld{args.score_threshold}{suffix}.pkl", "wb") as f:
        pickle.dump(pick_metric, f)
    with open(f"{args.save_path}/pick_metric_score_thld{args.score_threshold}_thld{station_threshold}{suffix}.pkl", "wb") as f:
        pickle.dump(pick_metric_thld, f)

print("\nWhole dataset:\n")
pick_metric.plot_residual(save_path=f"./metrics/phase_picking{suffix}.png", info_only=(args.save_path is not None))
if args.save_path is None:
    pick_metric.plot_polarity(save_path=f"./metrics/phase_polarity{suffix}.png")
print(f"\nnum_station >= {station_threshold} subset:\n")
pick_metric_thld.plot_residual(save_path=f"./metrics/phase_picking_threshold{station_threshold}{suffix}.png", info_only=(args.save_path is not None))
if args.save_path is None:
    pick_metric_thld.plot_polarity(save_path=f"./metrics/phase_polarity_threshold{station_threshold}{suffix}.png")
print("\n")

print(f"true_pick_files: {true_pick_files}")
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
        self.event_stat = []
        self.event_stat_label = []
        
        self.time_res = []
        self.x_res = []
        self.y_res = []
        self.z_res = []
        self.x = []
        self.y = []
        self.z = []  
        self.depth = []
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
        assert all([len(self.event_stat)==self.num_event_picks, len(self.event_stat_label)==self.num_event_picks_label]), f"event_stat and num_event_picks have different length in {self.num_event_picks} {self.num_event_picks_label}"
        self.event_stat = pd.DataFrame(self.event_stat)
        self.event_stat_label = pd.DataFrame(self.event_stat_label)
        
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
    
    def plot_residual(self, save_path=None, info_only=False):
        print(f"picks: {self.num_event_picks}, picks_label: {self.num_event_picks_label}")
        print(f"TP: {self.metric_dict['true_positives']}, FP: {self.metric_dict['false_positives']}, FN: {self.metric_dict['false_negatives']}")
        print(f"precision: {self.metric_dict['precision']:.4f}, recall: {self.metric_dict['recall']:.4f}, f1: {self.metric_dict['f1']:.4f}")
        print(f"miu_t: {self.metric_dict['miu_t']:.4f}, miu_x: {self.metric_dict['miu_x']:.4f}, miu_y: {self.metric_dict['miu_y']:.4f}, miu_z: {self.metric_dict['miu_z']:.4f}, miu_m: {self.metric_dict['miu_m']:.4f}, miu_dist: {self.metric_dict['miu_dist']:.4f}, miu_azi: {self.metric_dict['miu_azi']:.4f}")
        print(f"sigma_t: {self.metric_dict['sigma_t']:.4f}, sigma_x: {self.metric_dict['sigma_x']:.4f}, sigma_y: {self.metric_dict['sigma_y']:.4f}, sigma_z: {self.metric_dict['sigma_z']:.4f}, sigma_m: {self.metric_dict['sigma_m']:.4f}, sigma_dist: {self.metric_dict['sigma_dist']:.4f}, sigma_azi: {self.metric_dict['sigma_azi']:.4f}")
        if info_only:
            return

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
fn_event_ids = []
num_event_center_picks = 0

#dbscan = DBSCAN(eps=dbscan_epsilon, min_samples=1)
dbscan = DBSCAN(eps=dbscan_epsilon, min_samples=3)

with h5py.File(h5_file, "r") as fp:
    for pick in tqdm(event_files):
        if args.model == "eqnet":
            event_id = re.sub(r'\.csv$', '', os.path.basename(pick))
        elif args.model == "phasenet_plus":
            event_id = pick.split('/')[-1]
            
        if len(filter_events) > 0:
            if (event_id in filter_events.keys()) != args.include_events:
                continue

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
        df = df[df["event_score"] >= args.score_threshold]
        if args.main_event_only:
            new_df = []
            VAPP = 8.20 # km/s
            for sta_id in df["station_id"].unique():
                phase_type_gt = event[sta_id].attrs["phase_type"]
                phase_index_gt = event[sta_id].attrs["phase_index"]
                #try:
                #    if len(phase_type_gt)==1 or event[sta_id].attrs["phase_status"]=='automatic':
                #        new_df.append(df[df["station_id"]==sta_id])
                #        continue
                #except:
                #    pass
                
                p_picks_gt, s_picks_gt = [], []
                e_ids = event[sta_id].attrs["event_id"]
                for e_id in [event_id]:#set(e_ids): # exclude single p event in more than one event case
                    #if len(set(event[sta_id].attrs["phase_type"][e_ids==e_id])) < 2:
                    #    continue
                    p_picks_gt.append(phase_index_gt[np.logical_and(e_ids==e_id, phase_type_gt=='P')])
                    s_picks_gt.append(phase_index_gt[np.logical_and(e_ids==e_id, phase_type_gt=='S')])
                p_picks_gt = np.concatenate(p_picks_gt)
                s_picks_gt = np.concatenate(s_picks_gt)
                if len(p_picks_gt) == 0 and len(s_picks_gt) == 0:
                    new_df.append(df[df["station_id"]==sta_id])
                    continue
                if len(s_picks_gt) == 0:
                    s_picks_gt = p_picks_gt + event[sta_id].attrs["distance_km"] / VAPP * sample_rate
                if len(p_picks_gt) == 0:
                    p_picks_gt = s_picks_gt - event[sta_id].attrs["distance_km"] / VAPP * sample_rate
                
                df_sta = df[df["station_id"] == sta_id]
                duration = np.stack([p_picks_gt-5*sample_rate, p_picks_gt+3*(s_picks_gt-p_picks_gt)], axis=1) / feature_scale
                duration = np.clip(duration, 0, 12000/feature_scale) #NOTE hard code the duration
                mask = np.zeros(len(df_sta), dtype=bool)
                for i, (p, s) in enumerate(duration):
                    mask = np.logical_or(mask, np.logical_and(df_sta["phase_index"]>=p, df_sta["phase_index"]<=s))
                df_sta = df_sta[mask]
                new_df.append(df_sta)
            df = pd.concat(new_df) if len(new_df) > 0 else pd.DataFrame()
                
        if len(df) == 0:
            event_metric.empty_files += 1
            event_metric_thld.empty_files +=1 if num_stations>=station_threshold else 0
            continue
        assert 'event_center_index' in df.columns, f'event_center_index not in {event_id}.columns'
        assert 'p_index' in df.columns, f'p_index not in {event_id}.columns'
        assert 's_index' in df.columns, f's_index not in {event_id}.columns'
        assert 'event_time' in df.columns, f'event_time not in {event_id}.columns'
        
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
        if len(classes) > 1:
            if classes[0] == -1:
                classes = classes[1:]
                counts = counts[1:]
            #classes = classes[counts>1]
            classes = classes[counts>=3]
            
        event_metric.num_event_picks += len(classes)
        event_metric_thld.num_event_picks += len(classes) if num_stations>=station_threshold else 0
        num_event_center_picks += len(df)
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
        if abs(min_time_diff) > event_dt_bar:
            fn_event_ids.append(event_id)
        #print(f"{event_id}: {min_time_diff:.2f}s, {avg_center}, {label_x:.2f}, {label_y:.2f}, {label_z:.2f}, {x_pred:.2f}, {y_pred:.2f}, {z_pred:.2f}, {x_diff:.2f}, {y_diff:.2f}, {z_diff:.2f}, {magnitude_pr:.2f}")
        event_metric.update_matched_picks([min_time_diff], [x_diff], [y_diff], [z_diff], [x_pred], [y_pred], [z_pred], [event.attrs["magnitude"]], [magnitude_pr], [cluster_picks[:,6]], [dist_diff], [azi_diff])
        if num_stations>=station_threshold:
            event_metric_thld.update_matched_picks([min_time_diff], [x_diff], [y_diff], [z_diff], [x_pred], [y_pred], [z_pred], [event.attrs["magnitude"]], [magnitude_pr], [cluster_picks[:,6]], [dist_diff], [azi_diff])
        
event_metric.list2numpy()
event_metric_thld.list2numpy()
event_metric.fn_event_ids = fn_event_ids

print(f"whole dataset empty files: {event_metric.empty_files}")
print(f"whole dataset no match files: {event_metric.no_match_files}")
print("\n")
print(f"num_station >= {station_threshold} subset empty files: {event_metric_thld.empty_files}")
print(f"num_station >= {station_threshold} subset no match files: {event_metric_thld.no_match_files}")

# %%
event_metric.calc_metrics(event_dt_bar=event_dt_bar)
event_metric_thld.calc_metrics(event_dt_bar=event_dt_bar)

# %%
if args.save_path is not None:
    with open(f"{args.save_path}/event_metric_score_thld{args.score_threshold}{suffix}.pkl", "wb") as f:
        pickle.dump(event_metric, f)
    with open(f"{args.save_path}/event_metric_score_thld{args.score_threshold}_thld{station_threshold}{suffix}.pkl", "wb") as f:
        pickle.dump(event_metric_thld, f)

print("\nWhole dataset:\n")
event_metric.plot_residual(f"./metrics/event_residual{suffix}.png", info_only=(args.save_path is not None))
print(f"azimuth residual: len: {len(event_metric.azimuth_res.flatten())}, abs medium: {np.median(np.abs(event_metric.azimuth_res.flatten()))}, max: {event_metric.azimuth_res.flatten()[np.argmax(np.abs(event_metric.azimuth_res.flatten()))]}, min: {event_metric.azimuth_res.flatten()[np.argmin(np.abs(event_metric.azimuth_res.flatten()))]}")
print(f"\nnum_station >= {station_threshold} subset:\n")
event_metric_thld.plot_residual(f"./metrics/event_residual_threshold{station_threshold}{suffix}.png", info_only=(args.save_path is not None))
print(f"azimuth residual: len: {len(event_metric_thld.azimuth_res.flatten())}, abs medium: {np.median(np.abs(event_metric_thld.azimuth_res.flatten()))}, max: {event_metric_thld.azimuth_res.flatten()[np.argmax(np.abs(event_metric_thld.azimuth_res.flatten()))]}, min: {event_metric_thld.azimuth_res.flatten()[np.argmin(np.abs(event_metric_thld.azimuth_res.flatten()))]}")

print(f"num_event_center_picks: {num_event_center_picks}")