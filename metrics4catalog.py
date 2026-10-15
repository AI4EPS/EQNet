# %%
import os
from functools import partial
from glob import glob
import copy
import datetime
import errno
from collections import OrderedDict, defaultdict, deque
from typing import List, Optional, Tuple, Union
import abc

import argparse
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import obspy
from tqdm import tqdm
import itertools
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min

sns.set_theme()

# %% [markdown]
# ## Metrics for NC dataset

# %% [markdown]
# ### Config

# %%

parser = argparse.ArgumentParser(description='Metrics for earthquake dataset', add_help=True)

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
    
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

mkdir("./metrics") # create the metrics directory

time_tolerance = 0.5 # s
distribution_bar = 0.5 # s
station_threshold = 10
event_dt_bar = 3 # s
score_threshold = 0.3 # min prob for phase_score

dbscan_epsilon = 5
time_space_weight = [1, 0.5, 0.5, 1]

sample_rate = 100
feature_scale = 16
degree2km = 111.32

class EarthquakeMonitoringMetrics:
    def __init__(self, params, attrs=['phase_score'], default_threshold=0.3, sample_rate=100):
        '''Base class for earthquake monitoring metrics
        Args:
            params: list, parameters for metrics, e.g., phase_time, phase_polarity, origin_time, etc.
            attrs: list, attributes for metrics, e.g., magnitude for phase picking, snr for event detection, etc.
            sample_rate: int, sample rate of the data
        '''
        self.params = params
        self.attrs = attrs
        self.default_threshold = default_threshold
        self.sample_rate = sample_rate
        
        self.matched_picks = []
        self.matched_picks_df = None
        
        self.precision = defaultdict(None)
        self.recall = defaultdict(None)
        self.f1 = defaultdict(None)
        
        self._metrics_by_score = {}
        self.precision_list = defaultdict(list)
        self.recall_list = defaultdict(list)
        self.f1_list = defaultdict(list)
        self.tpr_list = defaultdict(list)
        self.fpr_list = defaultdict(list)
        self.ap = defaultdict(None)
        self.max_f1_score = defaultdict(float)
        
    def update_matched_picks(self, matched_picks: Union[pd.DataFrame, list]):
        '''Update matched_picks
        Args:
            matched_picks: pd.DataFrame or list, matched_picks for each event
        '''
        if isinstance(matched_picks, pd.DataFrame):
            self.matched_picks.append(matched_picks)
        elif isinstance(matched_picks, list):
            self.matched_picks.extend(matched_picks)
        else:
            raise ValueError(f"matched_picks should be pd.DataFrame or list, but got {type(matched_picks)}")
        
    def to_dataframe(self):
        '''Concat matched_picks to DataFrame
        '''
        self.matched_picks_df = pd.concat(self.matched_picks, ignore_index=True)
        self.matched_picks_df = self.matched_picks_df.sort_values(by=["phase_score"], ascending=False)
        
    def calc_metrics(self):
        '''Calculate metrics
        '''
        raise NotImplementedError
    
    @property
    def metric_dict(self, score_threshold=None):
        '''Get metric dict
        '''
        if score_threshold is None:
            score_threshold = self.default_threshold
        return self._metrics_by_score[score_threshold]
    
    @property
    def mAP(self):
        '''Mean Average Precision
        '''
        return np.mean([self.ap[phase_type] for phase_type in self.ap.keys()])
    
    def plot_residuals(self):
        '''Plot residuals
        '''
        raise NotImplementedError
    
    def plot_precision_recall(self):
        '''Plot precision-recall curve
        '''
        raise NotImplementedError

class PickMetrics(EarthquakeMonitoringMetrics):
    '''Metrics for phase picking
    P, S arrivals and polarities(optional)
    '''
    def __init__(self, params=['phase_time'], attrs=['magnitude'], phase_types=['P', 'S'], if_polarity=False, default_threshold=0.3, sample_rate=100):
        super(PickMetrics, self).__init__(params, attrs, default_threshold, sample_rate)
        if if_polarity:
            if 'phase_polarity' not in self.params:
                self.params.append('phase_polarity')
        else:
            if 'phase_polarity' in self.params:
                if_polarity = True
        self.if_polarity = if_polarity
        self.phase_types = phase_types
        
        self.empty_files = 0
        self.phase_picks_label = {phase_type: 0 for phase_type in self.phase_types}
        self.phase_picks_pred = {phase_type: 0 for phase_type in self.phase_types}
        self.phase_plority_label = {phase_type: 0 for phase_type in self.phase_types}
        self.phase_plority_pred = {phase_type: 0 for phase_type in self.phase_types}
        
    
    def calc_metrics(self, score_thresholds=Union(None, list, float), time_tolerance=0.5, eval_param="phase_time", eval_keys=None, distribution_width=None, phase_types=None, refresh_df=False):
        '''Calculate metrics according to given evaluation parameter using matched_picks_df
        Args:
            score_threshold: float, threshold for phase_score
            time_tolerance: float, time tolerance for phase_time_residual
            eval_param: str, evaluation parameter, phase_time or phase_polarity
            eval_keys: list(str), key in matched_picks_df for evaluation
            distribution_width: float, width for distribution plot
            phase_types: list, phase types for evaluation
            refresh_df: bool, whether to refresh matched_picks_df
        '''
        if score_thresholds is None:
            score_thresholds = self.default_threshold
        if isinstance(score_thresholds, float):
            score_thresholds = [score_thresholds]
        if self.matched_picks_df is None or refresh_df:
            self.to_dataframe()
        if distribution_width is None:
            distribution_width = time_tolerance
        if phase_types is None:
            phase_types = self.phase_types
        elif not isinstance(phase_types, list):
            phase_types = [phase_types]
        
        column_keys = self.matched_picks_df.columns
        if eval_keys is None:
            eval_keys = ["phase_time_residual"] if eval_param == "phase_time" else ["phase_time_residual", "phase_polarity_pred", "phase_polarity_true"] if eval_param == "phase_polarity" else None
        if eval_param not in self.params or eval_keys is None:
            raise ValueError(f"eval_param should be in {self.params} and must be phase_time or phase_polarity, but got {eval_param}")
        if any([key not in column_keys for key in eval_keys]):
            raise ValueError(f"eval_key should be in {column_keys}, but got {eval_keys}")
        
        
        def check_phase_time(row, key='phase_time_residual', tolerance=time_tolerance):
            return abs(row[key]) <= tolerance

        def check_phase_polarity(row, time_key='phase_time_residual', polarity_key=["phase_polarity_pred", "phase_polarity_true"], tolerance=time_tolerance):
            return (
                abs((row[time_key]) <= tolerance)
                and row[polarity_key[0]] == row[polarity_key[1]]
            )
            
        for score_threshold in score_thresholds:
            df_by_score = self.matched_picks_df[self.matched_picks_df["phase_score"] >= score_threshold]
            df_by_score = df_by_score.sort_values(by=["phase_score"], ascending=False)
            if len(df_by_score) == 0:
                print("No match found!")
                return None

            if eval_param == "phase_time":
                df_by_score["correct"] = df_by_score.apply(partial(check_phase_time, key=eval_keys[0]), axis=1)
            elif eval_param == "phase_polarity":
                df_by_score["correct"] = df_by_score.apply(partial(check_phase_time, key=eval_keys[0]), axis=1)
                df_by_score["polarity_correct"] = df_by_score.apply(partial(check_phase_polarity, time_key=eval_keys[0], polarity_key=eval_keys[1:3]), axis=1)
            else:
                raise ValueError(f"eval_param should be phase_time or phase_polarity, but got {eval_param}")

            metric_dict = {}
            for phase_type in phase_types:
                df_by_type = df_by_score[df_by_score["phase_type"] == phase_type]
                true_positive = df_by_type["correct"].sum()
                false_positive = self.phase_picks_pred[phase_type] - true_positive
                false_negative = self.phase_picks_label[phase_type] - true_positive
                #AP
                tp = np.cumsum(df_by_type["correct"])
                fp = np.cumsum(~df_by_type["correct"])
                df_by_type['precision'] = np.nan_to_num(tp / (tp + fp))
                df_by_type['recall'] = tp / self.phase_picks_label[phase_type]
                
                ap_at_threshold = 0
                for recall_level in np.arange(0, 1.1, 0.1):
                    try:
                        ap_at_threshold += df_by_type[df_by_type['recall'] >= recall_level]['precision'].max() / 11
                    except:
                        pass
                

                self.precision[phase_type] = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
                self.recall[phase_type] = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
                self.f1[phase_type] = 2 * self.precision[phase_type] * self.recall[phase_type] / (self.precision[phase_type] + self.recall[phase_type]) if self.precision[phase_type] + self.recall[phase_type] > 0 else 0
                TPR = true_positive / self.phase_picks_label[phase_type] if self.phase_picks_label[phase_type] > 0 else 0
                FPR = false_positive / self.phase_picks_pred[phase_type] if self.phase_picks_pred[phase_type] > 0 else 0

                miu = df_by_type[df_by_type[eval_keys[0]] <= distribution_width][eval_keys[0]].mean() # time residual mean
                sigma = df_by_type[df_by_type[eval_keys[0]] <= distribution_width][eval_keys[0]].std() # time residual std

                time_metric_dict = {
                    "phase_type": phase_type,
                    "phase_score": score_threshold,
                    "num_picks": self.phase_picks_pred[phase_type],
                    "num_picks_label": self.phase_picks_label[phase_type],
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "precision": self.precision[phase_type],
                    "recall": self.recall[phase_type],
                    "f1": self.f1[phase_type],
                    "TPR": TPR,
                    "FPR": FPR,
                    "miu": miu,
                    "sigma": sigma,
                    "AP": ap_at_threshold # ap at threshold
                }
                metric_dict[phase_type] = time_metric_dict

                if eval_param == "phase_polarity":
                    true_positive_polarity = df_by_type["polarity_correct"].sum()
                    false_positive_polarity = self.phase_plority_pred[phase_type] - true_positive_polarity
                    false_negative_polarity = self.phase_plority_label[phase_type] - true_positive_polarity

                    precision_polarity = true_positive_polarity / (true_positive_polarity + false_positive_polarity) if true_positive_polarity + false_positive_polarity > 0 else 0
                    recall_polarity = true_positive_polarity / (true_positive_polarity + false_negative_polarity) if true_positive_polarity + false_negative_polarity > 0 else 0
                    f1_polarity = 2 * precision_polarity * recall_polarity / (precision_polarity + recall_polarity) if precision_polarity + recall_polarity > 0 else 0
                    TPR_polarity = true_positive_polarity / self.phase_plority_label[phase_type] if self.phase_plority_label[phase_type] > 0 else 0
                    FPR_polarity = false_positive_polarity / self.phase_plority_pred[phase_type] if self.phase_plority_pred[phase_type] > 0 else 0

                    self.precision[phase_type + "_polarity"] = precision_polarity
                    self.recall[phase_type + "_polarity"] = recall_polarity
                    self.f1[phase_type + "_polarity"] = f1_polarity
                    polarity_metric_dict = {
                        "phase_type": phase_type,
                        "phase_score": score_threshold,
                        "num_picks": self.phase_plority_pred[phase_type],
                        "num_picks_label": self.phase_plority_label[phase_type],
                        "true_positive": true_positive_polarity,
                        "false_positive": false_positive_polarity,
                        "false_negative": false_negative_polarity,
                        "precision": precision_polarity,
                        "recall": recall_polarity,
                        "f1": f1_polarity,
                        "TPR": TPR_polarity,
                        "FPR": FPR_polarity
                    }
                    metric_dict[phase_type + "_polarity"] = polarity_metric_dict

            self._metrics_by_score[score_threshold] = metric_dict
        
        return metric_dict
            
    
    def plot_residuals(self, score_threshold=None, phase_types=None, save_path=None):
        '''Plot time residuals distribution
        Args:
            score_threshold: float, threshold for phase_score
            time_tolerance: float, time tolerance for phase_time_residual
            phase_types: list, phase types for evaluation
            distribution_width: float, width for distribution plot
        '''
        if score_threshold is None:
            score_threshold = self.default_threshold
        try:
            self._metrics_by_score[score_threshold]
        except KeyError:
            raise ValueError(f"score_threshold {score_threshold} not found in metrics")
        if distribution_width is None:
            distribution_width = time_tolerance
        if phase_types is None:
            phase_types = self.phase_types
        elif not isinstance(phase_types, list):
            phase_types = [phase_types]
        
        pass
    
    def plot_precision_recall(self, score_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], phase_types=None, eval_params=["phase_time"], save_path=None):
        pass


# %% [markdown]
# ### Event detection

# %%

class EventMetrics(EarthquakeMonitoringMetrics):
    def __init__(self, params=['origin_time'], attrs=['magnitude'], phase_types=['event'], default_threshold=0.3, sample_rate=100):
        super(EventMetrics, self).__init__(params, attrs, default_threshold, sample_rate)
        self.empty_files = 0
        self.phase_types = phase_types
        
        self.event_picks_label = {phase_type: 0 for phase_type in self.phase_types}
        self.event_picks_pred = {phase_type: 0 for phase_type in self.phase_types}
        
    def calc_metrics(self, score_threshold=None, tolerance=[0.5], eval_params=["origin_time"], eval_keys=None, distribution_width=None, phase_types=None, refresh_df=False):
        if score_threshold is None:
            score_threshold = self.default_threshold
        if self.matched_picks_df is None or refresh_df:
            self.to_dataframe()
        
        if not isinstance(eval_params, list):
            eval_params = [eval_params]
        if not isinstance(tolerance, list):
            tolerance = [tolerance]
        if len(tolerance) != len(eval_params):
            raise ValueError("you should provide tolerance for each eval_param")
        if distribution_width is None:
            distribution_width = tolerance
        if phase_types is None:
            phase_types = self.phase_types
        elif not isinstance(phase_types, list):
            phase_types = [phase_types]
        
        df_by_score = self.matched_picks_df[self.matched_picks_df["phase_score"] >= score_threshold]
        if len(df_by_score) == 0:
            print("No match found!")
            return None
        
        def check_all_residual_params(row, keys, tolerance):
            return all([abs(row[key]) <= tol for key, tol in zip(keys, tolerance)])
        
        column_keys = self.matched_picks_df.columns
        if eval_keys is None:
            eval_keys = [param+"_residual" for param in eval_params]
        elif not isinstance(eval_keys, list):
            eval_keys = [eval_keys]
        if len(eval_keys) != len(eval_params):
            raise ValueError("you should provide eval_keys for each eval_param")
        if any([key not in column_keys for key in eval_keys]):
            raise ValueError(f"eval_key should be in {column_keys}, but got {eval_keys}")
        
        df_by_score["correct"] = df_by_score.apply(partial(check_all_residual_params, keys=eval_keys, tolerance=tolerance), axis=1)
        
        metric_dict = {}
        for phase_type in phase_types:
            df_by_type = df_by_score[df_by_score["phase_type"] == phase_type]
            true_positive = df_by_type["correct"].sum()
            false_positive = self.phase_picks_pred[phase_type] - true_positive
            false_negative = self.phase_picks_label[phase_type] - true_positive

            self.precision[phase_type] = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
            self.recall[phase_type] = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
            self.f1[phase_type] = 2 * self.precision[phase_type] * self.recall[phase_type] / (self.precision[phase_type] + self.recall[phase_type]) if self.precision[phase_type] + self.recall[phase_type] > 0 else 0
            TPR = true_positive / self.phase_picks_label[phase_type] if self.phase_picks_label[phase_type] > 0 else 0
            FPR = false_positive / self.phase_picks_pred[phase_type] if self.phase_picks_pred[phase_type] > 0 else 0
            
            df_by_tol = df_by_type[df_by_type['correct']==True]
            miu = {key: df_by_tol[df_by_tol[key] <= distribution_width[i]][key].mean() for i, key in enumerate(eval_keys)}
            sigma = {key: df_by_tol[df_by_tol[key] <= distribution_width[i]][key].std() for i, key in enumerate(eval_keys)}
            
            _metric_dict = {
                "phase_type": phase_type,
                "phase_score": score_threshold,
                "num_picks": self.event_picks_pred[phase_type],
                "num_picks_label": self.event_picks_label[phase_type],
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": self.precision[phase_type],
                "recall": self.recall[phase_type],
                "f1": self.f1[phase_type],
                "TPR": TPR,
                "FPR": FPR,
                "miu": miu,
                "sigma": sigma
            }
            
            metric_dict[phase_type] = _metric_dict
            
        self._metrics_by_score[score_threshold] = metric_dict
        
        return metric_dict
    
    def plot_residuals(self, score_threshold=None, phase_types=None, save_path=None):
        pass
    
    def plot_precision_recall(self, score_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], phase_types=None, save_path=None):
        pass
    

# %% [markdown]
# ### Load data
# %%
