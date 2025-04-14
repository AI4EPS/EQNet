import os
import random
from glob import glob
import pickle
import time
import math
import copy

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import obspy
import fsspec
import json
import logging
import datasets
import torch
from torch.utils.data import Dataset, IterableDataset
from collections import defaultdict, Counter
from datetime import timedelta, datetime
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from ..utils.station_sampler import reorder_keys

def optimal_transport_clustering(points, k):
    # 使用K-means获取初始中心
    kmeans = KMeans(n_clusters=k, random_state=42).fit(points)
    centers = kmeans.cluster_centers_
    
    # 计算成本矩阵(点到中心的距离)
    n_points = len(points)
    cost_matrix = np.zeros((n_points, k))
    for i in range(k):
        cost_matrix[:, i] = np.sum((points - centers[i])**2, axis=1)
    
    # 重复中心以匹配点数量(每个簇大小相近)
    points_per_cluster = n_points // k
    remainder = n_points % k
    repeated_centers = []
    for i in range(k):
        repeated_centers.extend([i] * (points_per_cluster + (1 if i < remainder else 0)))
    
    # 构建扩展成本矩阵
    ext_cost = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if j < len(repeated_centers):
                ext_cost[i, j] = cost_matrix[i, repeated_centers[j]]
            else:
                ext_cost[i, j] = float('inf')
    
    # 使用匈牙利算法求解
    row_ind, col_ind = linear_sum_assignment(ext_cost)
    
    # 获取聚类标签
    labels = np.array([repeated_centers[j] for j in col_ind])
    return labels

def reorder_groups_dp(stations_df, num_groups):
    """使用动态规划找到最短开放路径，时间复杂度O(n²2^n)"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 计算每个组的中心点
    group_centers = stations_df.groupby('group').agg({'longitude': 'mean', 'latitude': 'mean'})
    group_centers.sort_values(by='group', inplace=True)
    
    # 计算距离矩阵
    group_locations = group_centers[['longitude', 'latitude']].values # (num_groups, 2)
    dist = np.sqrt(np.sum((group_locations[:, np.newaxis] - group_locations[np.newaxis, :]) ** 2, axis=-1))  # (num_groups, num_groups)
    group_centers = group_centers.reset_index().values.tolist()  # [(group_id, lon, lat), ...]
    
    # 动态规划表: dp[mask][i] 表示从任意起点出发，访问mask中的所有点并以i结束的最短路径
    n = num_groups
    # 初始化dp表
    dp = np.full((1 << n, n), float('inf'))
    
    # 对于单点路径，距离为0
    for i in range(n):
        dp[1 << i][i] = 0
    
    # 填充dp表
    for mask in range(1, 1 << n):
        for i in range(n):
            # 如果i不在mask中，跳过
            if not (mask & (1 << i)):
                continue
                
            # 去掉i后的子集
            prev = mask & ~(1 << i)
            if prev == 0:  # 如果i是唯一点，已经处理过
                continue
                
            # 检查所有可能的前一个点
            for j in range(n):
                if not (prev & (1 << j)):
                    continue
                dp[mask][i] = min(dp[mask][i], dp[prev][j] + dist[j][i])
    
    # 找到最优路径
    mask = (1 << n) - 1  # 全集
    
    # 找到最后一个点
    last = min(range(n), key=lambda i: dp[mask][i])
    
    # 重建路径
    path = [last]
    mask = mask & ~(1 << last)
    
    while mask:
        best_prev = -1
        best_dist = float('inf')
        
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            current_dist = dp[mask][i] + dist[i][path[0]]
            if current_dist < best_dist:
                best_dist = current_dist
                best_prev = i
        
        path.insert(0, best_prev)  # 插入到路径头部
        mask = mask & ~(1 << best_prev)
    
    # 映射回原来的组ID
    final_path = [group_centers[p][0] for p in path]
    group_mapping = {old: new for new, old in enumerate(final_path)}
    
    # 更新组标签
    stations_df = stations_df.copy()
    stations_df['new_group'] = stations_df['group'].map(group_mapping)
    stations_df.sort_values(by=['new_group', 'longitude', 'latitude'], inplace=True)
    stations_df.reset_index(drop=True, inplace=True)
    stations_df['group'] = stations_df['new_group']
    return stations_df

@np.vectorize
def mapping_phase_type(phase):
    if phase in ["P", "Pn", "Pg", "Pb"]:
        return "P"
    elif phase in ["S", "Sn", "Sg", "Sb"]:
        return "S"
    else:
        return phase

def generate_label(phase_list, label_width=[150, 150], nt=8192):
    if len(label_width) == 1:
        label_width = label_width * len(phase_list)
    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)

    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.05] = 0.0
            target[i + 1, :] += gaussian

    target[1:, :] = np.minimum(1.0, target[1:, :])
    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    return target

def generate_phase_label(
    phase_list,  # [[P1, P2, ...], [S1, S2, ...]]
    label_width=[150],
    nt=8192,
    mask_width=None,
):
    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)
    mask = np.zeros([nt], dtype=np.float32)

    if len(label_width) == 1:
        label_width = label_width * len(phase_list)
    if mask_width is None:
        mask_width = [int(x * 1.5) for x in label_width]
    else:
        # mask_width = [min(int(x * 1.5), mask_width) for x in label_width]
        mask_width = [mask_width] * len(label_width)

    for i, (picks, w, m) in enumerate(zip(phase_list, label_width, mask_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.05] = 0.0
            target[i + 1, :] += gaussian
            mask[int(phase_time) - m : int(phase_time) + m] = True

    target[1:, :] = np.minimum(1.0, target[1:, :])
    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    return target, mask

def generate_event_label(
    reference_center,
    #event_param,
    label_width=[100],
    nt=8192,
    mask_width=None,
):
    target_center = np.zeros(nt, dtype=np.float32)
    #target_param = np.zeros([len(event_param[0]), nt], dtype=np.float32)
    mask = np.zeros(nt, dtype=bool)
    param_mask = np.zeros(nt, dtype=bool)
    
    if len(label_width) == 1:
        label_width = label_width * len(reference_center)
    if mask_width is None:
        mask_width = [int(x * 1.0)//2 for x in label_width] # or 1.5?
    else:
        mask_width = [min(int(x * 1.0), label_width)//2 for x in label_width]

    #for i, (c0, t0, w, m) in enumerate(zip(reference_center, event_param, label_width, mask_width)):
    for i, (c0, w, m) in enumerate(zip(reference_center, label_width, mask_width)):
        t = np.arange(nt) - c0
        gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
        gaussian[gaussian < 0.05] = 0.0
        target_center += gaussian
        mask[max(0, int(c0) - int(w*1.5)) : min(nt, int(c0) + int(w*1.5))] = True
        param_mask[max(0, int(c0) - m) : min(nt, int(c0) + m)] = True
        #t = np.arange(nt) - t0
        #target_param[max(0, int(c0) - m) : min(nt, int(c0) + m)] = t[max(0, int(c0) - m) : min(nt, int(c0) + m)]

    #return target_center, target_param, mask
    return target_center, mask, param_mask

def add_random_negative_samples(mask, random_mask_rate=0.1, random_mask_unit=32):
    nx, nt = mask.shape
    manual_station_mask = np.any(mask, axis=1) # nx
    n_sample_mask = np.random.rand(nx, np.ceil(nt/random_mask_unit).astype(int)) < random_mask_rate
    mask = np.logical_or(mask, np.repeat(n_sample_mask, random_mask_unit, axis=1)[:,:nt]) * manual_station_mask[:, np.newaxis]
    
    return mask.astype(bool)

def cut_data(meta, nt=8192, nt0=2960, feature_scale=16, base_phase_points=80, if_reverse_roll=False):
    tries = 0
    max_tries = 30
    base_phase_points = base_phase_points * meta["num_manual_stations"]
    phase_mask = (meta["phase_pick"] != 0).astype(np.float32)
    event_mask = (meta["event_location_mask"] != 0).astype(np.float32)
    main_event_center_index = meta['main_event_center'][:meta["num_manual_stations"], 0]
    p_start = np.nonzero(phase_mask[:, 1, :].sum(axis=0))[0].min() // feature_scale
    event_start = np.nonzero(event_mask.sum(axis=0))[0].min()
    s_end = np.nonzero(phase_mask[:, 2, :].sum(axis=0))[0].max() // feature_scale
    event_end = np.nonzero(event_mask.sum(axis=0))[0].max()
    main_event_start = np.nonzero(meta["association_mask"].sum(axis=0))[0].min()
    if not if_reverse_roll:
        rand_lower = max(0, min(p_start, event_start)-nt//feature_scale+10, main_event_start-nt//feature_scale+10)
    else:
        rand_lower = -max(nt0//feature_scale, nt//feature_scale-10-max(s_end, event_end))
    #rand_upper = max(1, min(nt0//feature_scale, min(p_start, event_start)-5))
    #raw_rand_upper = max(1, nt0//feature_scale)
    rand_upper = max(1, main_event_start-20)
    #raw_rand_upper = max(min(p_start, event_start)-10, nt0//feature_scale)
    raw_rand_upper = max(min(p_start, event_start), max(s_end, event_end)-nt//feature_scale+10, nt0//feature_scale)
    phase_points = phase_mask[:,1:, :nt].sum()
    event_points = event_mask[:,:nt//feature_scale].sum()
    phase_types = phase_mask.shape[1]-1
    
    for _ in range(max_tries):
        it = random.randint(rand_lower, raw_rand_upper)
        if it >= 0:
            tmp_phase_points = phase_mask[:,1:, it*feature_scale:it*feature_scale+nt].sum()
            tmp_event_points = event_mask[:,it:it+nt//feature_scale].sum()
        else:
            tmp_phase_points = phase_mask[:, 1:, :nt+it*feature_scale].sum() + phase_mask[:, 1:, it*feature_scale:].sum()
            tmp_event_points = event_mask[:, :nt//feature_scale+it].sum() + event_mask[:, it:].sum()
        if all([tmp_phase_points >= base_phase_points * phase_types, tmp_event_points >= base_phase_points // feature_scale]):
            # at least one main event center in the range
            if np.any((main_event_center_index > it) & (main_event_center_index < it + nt//feature_scale)):
                #print(f"it={it}, main_event_center_index={main_event_center_index}")
                break

    if not all([tmp_phase_points>=base_phase_points*phase_types, tmp_event_points>=base_phase_points//feature_scale]):
        rand_lower = max(0, main_event_start-nt//feature_scale+10) if not if_reverse_roll else rand_lower // 2
        print(f"using smaller rand_upper={rand_upper}")
    
        while any([tmp_phase_points < phase_points//2, tmp_event_points < event_points//2]) or \
            not np.any((main_event_center_index > it) & (main_event_center_index < it + nt//feature_scale)):
            it = random.randint(rand_lower, rand_upper)
            if it >= 0:
                tmp_phase_points = phase_mask[:,1:, it*feature_scale:it*feature_scale+nt].sum()
                tmp_event_points = event_mask[:,it:it+nt//feature_scale].sum()
            else:
                tmp_phase_points = phase_mask[:, 1:, :nt+it*feature_scale].sum() + phase_mask[:, 1:, it*feature_scale:].sum()
                tmp_event_points = event_mask[:, :nt//feature_scale+it].sum() + event_mask[:, it:].sum()
            tries += 1
            if tries > max_tries:
                print(f"cut data failed, tries={tries}, it={it}")
                break

    if it>=0 and it*feature_scale+nt<phase_mask.shape[-1]:
        waveforms = meta["waveforms"][:, :, it*feature_scale:it*feature_scale+nt]
        amplitude = meta["amplitude"][:, :, it*feature_scale:it*feature_scale+nt]
        phase_pick = meta["phase_pick"][:, :, it*feature_scale:it*feature_scale+nt]
        phase_mask_ = meta["phase_mask"][:, it*feature_scale:it*feature_scale+nt]
        polarity = meta["polarity"][:, :, it*feature_scale:it*feature_scale+nt]
        polarity_mask = meta["polarity_mask"][:, it*feature_scale:it*feature_scale+nt]
        event_center = meta["event_center"][:, it:it+nt//feature_scale]
        event_mask = meta["event_mask"][:, it:it+nt//feature_scale]
        event_location = meta["event_location"][:, :, it:it+nt//feature_scale]
        event_location_mask = meta["event_location_mask"][:, it:it+nt//feature_scale]
        association_mask = meta["association_mask"][:, it:it+nt//feature_scale]
    else:
        waveforms = np.roll(meta["waveforms"], -it*feature_scale, axis=-1)[:, :, :nt]
        amplitude = np.roll(meta["amplitude"], -it*feature_scale, axis=-1)[:, :, :nt]
        phase_pick = np.roll(meta["phase_pick"], -it*feature_scale, axis=-1)[:, :, :nt]
        phase_mask_ = np.roll(meta["phase_mask"], -it*feature_scale, axis=-1)[:, :nt]
        polarity = np.roll(meta["polarity"], -it*feature_scale, axis=-1)[:, :, :nt]
        polarity_mask = np.roll(meta["polarity_mask"], -it*feature_scale, axis=-1)[:, :nt]
        event_center = np.roll(meta["event_center"], -it, axis=-1)[:, :nt//feature_scale]
        event_mask = np.roll(meta["event_mask"], -it, axis=-1)[:, :nt//feature_scale]
        event_location = np.roll(meta["event_location"], -it, axis=-1)[:, :, :nt//feature_scale]
        event_location_mask = np.roll(meta["event_location_mask"], -it, axis=-1)[:, :nt//feature_scale]
        association_mask = np.roll(meta["association_mask"], -it, axis=-1)[:, :nt//feature_scale]
    main_event_center = meta['main_event_center'][:meta["num_manual_stations"]] - np.array([[it, 0, 0]])
    main_event_center = main_event_center[(main_event_center[:, 0] >= 0) & (main_event_center[:, 0] < nt//feature_scale)]

    return {
        "waveforms": waveforms,
        "amplitude": amplitude,
        "phase_pick": phase_pick,
        "phase_mask": phase_mask_,
        "polarity": polarity,
        "polarity_mask": polarity_mask,
        "event_center": event_center,
        "event_mask": event_mask,
        "event_location": event_location,
        "event_location_mask": event_location_mask,
        "association_mask": association_mask,
        "main_event_center": main_event_center,
    }, it*feature_scale

def sample_by_magnitude(magnitude, critical_magnitude=3, sample_rate=0.7, min_magnitude=0.5):
    '''the events with magnitude larger than critical_magnitude are sampled with sample_rate
    G-R law: log10(N) = a - b*M
    '''
    if magnitude < min_magnitude or magnitude > critical_magnitude:
        return True
    else:
        bar = (1-sample_rate) / (10 ** ((magnitude - min_magnitude)/(critical_magnitude - min_magnitude)))
        return random.random() > bar
    
def sample_by_distance(distances, num_stations, num_manual_stations, critical_distance=50, min_distance=5, sample_rate=0.7):
    if num_manual_stations == 0:
        raise ValueError("num_manual_stations should be greater than 0")
    if num_manual_stations >= num_stations:
        prob = 1 - ((1-sample_rate) / (10 ** ((distances[:num_manual_stations] - min_distance)/(critical_distance - min_distance))))
        prob = prob / prob.sum()
        idxs = np.random.choice(np.arange(num_manual_stations), num_stations, replace=False, p=prob)
        
        return idxs
    else:
        prob_auto = 1 - ((1-sample_rate) / (10 ** ((distances[num_manual_stations:] - min_distance)/(critical_distance - min_distance))))
        prob_auto = prob_auto / prob_auto.sum()
        idxs_auto = np.random.choice(np.arange(num_manual_stations, len(distances)), num_stations-num_manual_stations, replace=False, p=prob_auto)
        
        return np.concatenate([np.arange(num_manual_stations), idxs_auto])

def find_events_with_shared_stations(e2sta, sta2e, min_shared=2, discard_ineligible=True):
    """
    Find events that share at least n stations with each other
    
    Args:
        event_to_stations: dict, event_id -> list[station_id]
        min_shared: int, minimum shared stations
        discard_ineligible: bool, whether to discard events that have less than n stations
        
    Returns:
        dict: event_id -> list[event_id]
    """
    result = {}
    for event_id, stations in e2sta.items():
        thld = min_shared
        if len(stations) < min_shared:
            if discard_ineligible:
                result[event_id] = []
                continue
            else:
                thld = len(stations)
        # the candidate events is the union of all events that share at least one station with the current event
        if thld > 2:
            candidate_events = Counter()
            for station in stations:
                candidate_events.update(sta2e[station])

            shared_events = set([event for event, count in candidate_events.items() if count >= thld])
            result[event_id] = list(shared_events - {event_id})
        elif thld == 1:
            candidate_events = set()
            for station in stations:
                candidate_events.update(sta2e[station])
            result[event_id] = list(candidate_events - {event_id})
        else: # min_shared == 2
            candidate_events = set()
            shared_events = set()
            for station in stations:
                # 交集说明至少有两个台站共享，所以交集满足条件，同时更新并集
                shared_events.update(candidate_events.intersection(sta2e[station]))
                candidate_events.update(sta2e[station])
            result[event_id] = list(shared_events - {event_id})
        
    return result

def sample_event_for_stacking(event_ids, e2sta_dict, sta2e_dict, events_dict, e2e_dict=None, min_shared_sta=2, magnitude_range=1.0, snr_bar=2.5, returns=3, max_tries=30):
    if not isinstance(event_ids, list):
        event_ids = [event_ids]
    event_id0 = event_ids[0]
    #other_event_ids = event_ids[1:]
    event_attrs0 = events_dict[event_id0]
    sta_ids0 = e2sta_dict[event_id0]
    mag0 = event_attrs0['magnitude']
    snr_median0 = event_attrs0['snr_median']
    year0 = event_attrs0['year']
    min_shared_sta = min(len(sta_ids0), min_shared_sta)
    candidate_events = find_events_with_shared_stations({event_id0: sta_ids0}, sta2e_dict, min_shared=min_shared_sta, discard_ineligible=False)[event_id0] if e2e_dict is None else e2e_dict[event_id0]
    #for e_id in event_ids:
    #    if e_id in candidate_events and random.random() < 0.5:
    #        candidate_events.remove(e_id)
    if not candidate_events:
        return []
    
    results = []
    for i in range(max_tries):
        if not candidate_events:
            break
        #sta_id = np.random.choice(sta_ids0)
        #event_id = np.random.choice(sta2e_dict[sta_id])
        event_id = random.choice(candidate_events)
        #for _ in range(10):
        #    #if (event_id in event_ids or (len(sta_ids_2)==1 and len(sta_ids0)>1)) or \
        #    if (i<=max_tries//2 and abs(year0-events_dict[event_id]['year'])>2):
        #        event_id = random.choice(candidate_events)
        #    else:
        #        break
        candidate_events.remove(event_id)
        sta_ids_2 = e2sta_dict[event_id]
        # prefer events in the close years
        # TODO: better solution needed
        # if i<=max_tries//2 and abs(year0-events_dict[event_id]['year'])>1:
        #     continue
        # check if at least given number stations are shared
        if len(set(sta_ids0).intersection(set(sta_ids_2)))<min_shared_sta:
            #print(f"shared sta too few")
            continue
        # check if the magnitude is in the range
        if magnitude_range is not None and abs(events_dict[event_id]['magnitude']-mag0)>magnitude_range:
            #print(f"magnitude too different")
            continue
        # check if the snrs are compatible
        if not (snr_median0>snr_bar and events_dict[event_id]['snr_median']>snr_bar):
            if abs(snr_median0-events_dict[event_id]['snr_median'])>snr_bar:
                continue
        
        results.append(event_id)
        if len(results)>=returns:
            break
    return results

def stack_event(
    meta1,
    meta2,
    max_shift=64 * 4,
    feature_scale=16,
    is_extra=False,
):
    prefix = "extra_" if is_extra else ""
    waveform1 = meta1["waveforms"]
    waveform2 = meta2["waveforms"]
    amplitude1 = meta1["amplitude"]
    amplitude2 = meta2["amplitude"]
    phase_pick1 = meta1["phase_pick"]
    phase_pick2 = meta2["phase_pick"]
    phase_mask1 = meta1["phase_mask"]
    phase_mask2 = meta2["phase_mask"]
    polarity1 = meta1["polarity"]
    polarity2 = meta2["polarity"]
    polarity_mask1 = meta1["polarity_mask"]
    polarity_mask2 = meta2["polarity_mask"]
    event_center1 = meta1["event_center"]
    event_center2 = meta2["event_center"]
    event_mask1 = meta1["event_mask"]
    event_mask2 = meta2["event_mask"]
    event_location_mask1 = meta1["event_location_mask"]
    event_location_mask2 = meta2["event_location_mask"]
    event_location1 = meta1["event_location"]
    event_location2 = meta2["event_location"]
    station_ids1 = meta1["station_ids"]
    station_ids2 = meta2["station_ids"]

    first_arrival1 = meta1["first_arrival"]
    first_arrival2 = meta2["first_arrival"]
    #amp_noise1 = meta1["amp_noise"]
    #amp_noise2 = meta2["amp_noise"]
    #amp_signal1 = meta1["amp_signal"]
    #amp_signal2 = meta2["amp_signal"]
    
    shared_sta_idx = []
    other_sta_idx = []
    for i, sta_id1 in enumerate(station_ids1):
        if sta_id1 in station_ids2:
            for j, sta_id2 in enumerate(station_ids2):
                if sta_id1 == sta_id2:
                    shared_sta_idx.append(j)
                    break
        else:
            other_sta_idx.append(i)
            shared_sta_idx.append(0)
    
    #FIXME: how to deal with first event auto station but second event manual station? or vice versa?
    #FIXME: try to use larger mask and only mask for phase pick and polarity
    waveform2 = waveform2[shared_sta_idx]
    waveform2[other_sta_idx] = 0
    #waveform2[other_sta_idx] = waveform1[other_sta_idx].copy() # to recover the original waveform1 on the other stations
    amplitude2 = amplitude2[shared_sta_idx]
    amplitude2[other_sta_idx] = 0
    #amplitude2[other_sta_idx] = amplitude1[other_sta_idx].copy()
    phase_pick2 = phase_pick2[shared_sta_idx]
    phase_pick2[other_sta_idx] = 0
    phase_mask2 = phase_mask2[shared_sta_idx]
    phase_mask2[other_sta_idx] = 0
    polarity2 = polarity2[shared_sta_idx]
    polarity2[other_sta_idx] = 0
    polarity_mask2 = polarity_mask2[shared_sta_idx]
    polarity_mask2[other_sta_idx] = 0
    event_center2 = event_center2[shared_sta_idx]
    event_center2[other_sta_idx] = 0
    event_mask2 = event_mask2[shared_sta_idx]
    event_mask2[other_sta_idx] = 0
    event_location_mask2 = event_location_mask2[shared_sta_idx]
    event_location_mask2[other_sta_idx] = 0
    event_location2 = event_location2[shared_sta_idx]
    event_location2[other_sta_idx] = 0
    station_ids2 = [station_ids2[i] for i in shared_sta_idx]
    duration2 = meta2["duration"]
    duration2 = [duration2[i] for i in shared_sta_idx]

    nx, nch, nt, = waveform1.shape  # nx, nch, nt
    duration_mask1 = np.zeros([nx, nt])
    duration_mask2 = np.zeros([nx, nt])
    # deal with multi-stacking
    if "duration_mask" in meta1.keys():
        duration_mask1 = meta1["duration_mask"]
    else:
        #ns, ne, nd = meta1["duration"].shape  # stations, events, (t0, t1)
        for i, duration_sta in enumerate(meta1["duration"]):
            if len(duration_sta)==0:
                continue
            for j, duration in enumerate(duration_sta):
                t0, t1 = duration
                duration_mask1[i, t0:t1] = 1

    if "duration_mask" in meta2.keys():
        duration_mask2 = meta2["duration_mask"]
    else:
        #ns, ne, nd = meta2["duration"].shape  # stations, events, (t0, t1)
        for i, duration_sta in enumerate(duration2):
            if len(duration_sta)==0 or i in other_sta_idx:
                continue
            for j, duration in enumerate(duration_sta):
                t0, t1 = duration
                duration_mask2[i, t0:t1] = 1

    max_tries = 40
    # while random.random() < 0.5:
    failed = True
    tries = 0
    for i in range(1 if random.random() < 0.8 else 2 if random.random() < 0.5 else 3):
    #for i in range(1):
        while tries < max_tries:
            if i == 0 and tries > max_tries // 2:
                break
            #min_ratio2 = np.log10(amp_noise1 * 2 / amp_signal2)
            #max_ratio2 = np.log10(amp_signal1 / 2 / amp_noise2)
            #if min_ratio2 > max_ratio2:
            #    # print(f"min_ratio2 > max_ratio2: {min_ratio2} > {max_ratio2}, {amp_noise1 = }, {amp_signal1 = }, {amp_noise2 = }, {amp_signal2 = }")
            #    break

            shift = random.randint(-max_shift, max_shift) + first_arrival1 - first_arrival2
            long_shift = shift * feature_scale
            tmp_mask2 = np.roll(phase_mask2, long_shift, axis=-1)
            if np.max(phase_mask1 + tmp_mask2) >= 2.0:
                tries += 1
                continue
            tmp_mask2 = np.roll(event_mask2, shift, axis=-1)
            if np.max(event_mask1 + tmp_mask2) >= 2.0:
                tries += 1
                continue
            tmp_mask2 = np.roll(duration_mask2, long_shift, axis=-1)
            if np.max(duration_mask1 + tmp_mask2) >= 2.0:
                tries += 1
                continue

            waveform2_ = np.roll(waveform2, long_shift, axis=-1)
            amplitude2_ = np.roll(amplitude2, long_shift, axis=-1)
            phase_pick2_ = np.roll(phase_pick2, long_shift, axis=-1)
            phase_mask2_ = np.roll(phase_mask2, long_shift, axis=-1)
            event_location2_ = np.roll(event_location2, shift, axis=-1)
            event_center2_ = np.roll(event_center2, shift, axis=-1)
            event_mask2_ = np.roll(event_mask2, shift, axis=-1)
            event_location_mask2_ = np.roll(event_location_mask2, shift, axis=-1)
            polarity2_ = np.roll(polarity2, long_shift, axis=-1)
            polarity_mask2_ = np.roll(polarity_mask2, long_shift, axis=-1)
            duration_mask2_ = np.roll(duration_mask2, long_shift, axis=-1)

            #ratio2 = 10 ** (random.uniform(max(-3, min_ratio2), min(3, max_ratio2)))
            #flip = random.choice([-1.0, 1.0])  ## flip waveform2 polarity
            #ratio_alpha = random.uniform(0.1, 0.9)
            #waveform1 = waveform1*ratio_alpha + waveform2_*(1-ratio_alpha) #* ratio2 * flip
            #amplitude1 = amplitude1*ratio_alpha + amplitude2_*(1-ratio_alpha) #* ratio2
            ratio2 = 10 ** (random.uniform(-1.5, 1.5))
            #FIXME: correct the magnitude?
            event_location2_[:, 4, :] = (event_location2_[:, 4, :] + np.log10(ratio2)) * event_mask2_
            waveform1 = waveform1 + waveform2_ * ratio2
            amplitude1 = amplitude1 + amplitude2_ * ratio2
            #waveform1 = waveform1 + waveform2_
            #amplitude1 = amplitude1 + amplitude2_
            #amp_noise1 = amp_noise1 + amp_noise2 * ratio2
            #amp_signal1 = min(amp_signal1, amp_signal2 * ratio2)

            phase_pick = np.zeros_like(phase_pick1)
            phase_pick[:, 1:, :] = phase_pick1[:, 1:, :] + phase_pick2_[:, 1:, :]
            assert np.all(phase_pick[:, 1:, :] <= 1.0), f"{np.max(phase_pick[:, 1:, :]) = }"
            phase_pick[:, 0, :] = np.maximum(0, 1.0 - np.sum(phase_pick[:, 1:, :], axis=1))
            phase_pick1 = phase_pick

            phase_mask1 = np.minimum(1.0, phase_mask1 + phase_mask2_)
            tmp_location = np.zeros_like(event_location1)
            tmp_location_mask = np.repeat((event_location_mask1 >= 1.0)[:, np.newaxis, :], event_location1.shape[1], axis=1)
            tmp_location[tmp_location_mask] = event_location1[tmp_location_mask]
            tmp_location_mask = np.repeat((event_location_mask2_ >= 1.0)[:, np.newaxis, :], event_location2_.shape[1], axis=1)
            tmp_location[tmp_location_mask] = event_location2_[tmp_location_mask]
            event_location1 = tmp_location
            event_center1 = event_center1 + event_center2_
            assert np.all(event_center1 <= 1.0), f"{np.max(event_center1) = }"
            event_mask1 = np.minimum(1.0, event_mask1 + event_mask2_)
            event_location_mask1 = np.minimum(1.0, event_location_mask1 + event_location_mask2_)
            #polarity1 = ((polarity1 - 0.5) + (polarity2_ - 0.5) * flip) + 0.5
            polarity = np.zeros_like(polarity1)
            polarity[:, 1:, :] = polarity1[:, 1:, :] + polarity2_[:, 1:, :]
            assert np.all(polarity[:, 1:, :] <= 1.0), f"{np.max(polarity[:, 1:, :]) = }"
            polarity[:, 0, :] = np.maximum(0, 1.0 - np.sum(polarity[:, 1:, :], axis=1))
            polarity1 = polarity
            polarity_mask1 = np.minimum(1.0, polarity_mask1 + polarity_mask2_)
            duration_mask1 = np.minimum(1.0, duration_mask1 + duration_mask2_)
            failed = False
            break
        
        #if failed:
        #    print(f"{prefix}stack {i}-th event fails after {max_tries} tries")
        # if tries == max_tries:
        #     print(f"stack {i}-th event fails after {max_tries} tries")

    return {
        "waveforms": waveform1,
        "amplitude": amplitude1,
        "phase_pick": phase_pick1,
        "phase_mask": phase_mask1,
        "event_center": event_center1,
        "event_mask": event_mask1,
        "event_location": event_location1,
        "event_location_mask": event_location_mask1,
        "polarity": polarity1,
        "polarity_mask": polarity_mask1,
        "association_mask": meta1["association_mask"],
        "station_location": meta1["station_location"],
        "main_event_center": meta1["main_event_center"],
        "num_stations": meta1["num_stations"],
        "num_manual_stations": meta1["num_manual_stations"],
        "event_id": meta1["event_id"],
        "station_ids": meta1["station_ids"],
        "duration": meta1["duration"],
        "duration_mask": duration_mask1,
        "first_arrival": np.min([meta1["first_arrival"], meta2["first_arrival"]]) if not failed else meta1["first_arrival"],
    }, failed

class SeismicNetworkIterableDataset(IterableDataset):
    degree2km = 111.32
    nt = 4096  ## 8992
    feature_nt = 256  ##560
    raw_nt = 12000 # temporary hard code
    feature_scale = int(nt / feature_nt)
    raw_feature_nt = int(raw_nt / feature_scale)

    def __init__(self, 
        data_path=None,
        data_list=None,
        hdf5_file=None,
        station_file=None, # a json file
        prefix="",
        format="h5",
        dataset="seismic_network",
        training=True,
        phases=["P", "S"],
        sort=False,
        num_stations=None,
        sample_by_magnitude=False,
        sample_by_distance=False,
        stack_event=False,
        drop_channel=False,
        ## for training
        huggingface_cache=None,
        phase_width=[100],
        polarity_width=[20],
        event_width=[10],
        ## for prediction
        is_pad=False,
        sampling_rate=100,
        response_path=None,
        response_xml=None,
        highpass_filter=0.0,
        rank=0,
        world_size=1,
        cut_patch=False,
        nt_cut=1024 * 8,
        nx=32,
        min_nt=1024,
        min_nx=1,
        **kwargs # for compatibility
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size 
        self.hdf5_fp = None
        if hdf5_file is not None:
            tmp_hdf5_keys = f"tmp/{'_'.join(hdf5_file.split('/'))}.txt"
            #tmp_hdf5_keys = f"tmp/debug_2022_train.txt"
            if not os.path.exists(tmp_hdf5_keys) and (rank == 0):
                with h5py.File(hdf5_file, "r", libver="latest", swmr=True) as fp:
                    self.data_list = []
                    for event in tqdm(list(fp.keys()), desc="Caching HDF5 keys"):
                        self.data_list.append(event)
                with open(tmp_hdf5_keys, "w") as f:
                    for x in self.data_list:
                        f.write(x + "\n")
                print(f"Saved {tmp_hdf5_keys}")
            else:
                while not os.path.exists(tmp_hdf5_keys):
                    time.sleep(1)
                print(f"Reading {tmp_hdf5_keys}")
                self.data_list = pd.read_csv(tmp_hdf5_keys, header=None, names=["event_id"])["event_id"].values.tolist()
            if huggingface_cache is not None:
                self.huggingface_cache = True
                data_path = f"{os.path.dirname(hdf5_file)}/waveform_h5"
                code_dir = os.path.dirname(os.path.abspath(__file__))
                script_dir = os.path.join(code_dir, "quakeflow_nc.py")
                self.huggingface_dataset = datasets.load_dataset(script_dir, split="train", name='event_custom', cache_dir=huggingface_cache, num_proc=32, data_files={"train": sorted(glob(f"{data_path}/**", recursive=False))}, trust_remote_code=True)
                #self.huggingface_dataset = datasets.load_from_disk(f'{huggingface_cache}/quakeflow_nc_cache/arrow')
                #self.huggingface_dataset.set_format("numpy")
                self.e2idx_dict = {event_id: idx for idx, event_id in enumerate(self.huggingface_dataset["event_id"])}
            else:
                self.huggingface_cache = False
        elif data_list is not None:
            with open(data_list, "r") as f:
                self.data_list = f.read().splitlines()
        elif data_path is not None:
            if "hour" in format:
                days = glob(f"{data_path}/**/", recursive=False)
                hours = []
                for day in sorted(days):
                    hours += glob(f"{day}**/", recursive=False)
                self.data_list = [
                    f"{x}{prefix}*.{format.split('_')[0]}" for x in sorted(list(hours))
                ]
            else:
                self.data_list = [
                    x for x in sorted(list(glob(os.path.join(data_path, f"{prefix}*.{format}"))))
                ]
        else:
            self.data_list = None
        if self.data_list is not None:
            self.data_list = self.data_list[rank::world_size]
        
        if station_file is not None:
            #self.station_info = json.load(open(station_file, "r"))
            self.station_info = pd.read_csv(station_file)
            self.station_info.set_index("station_id", inplace=True)
            
        self.data_path = data_path
        self.hdf5_file = hdf5_file
        if (hdf5_file is not None) and (not training):
            self.hdf5_fp = h5py.File(hdf5_file, "r", libver="latest", swmr=True)
        self.ill_event_ids = set() if not os.path.exists('tmp/ill_event_ids.pkl') else pickle.load(open('tmp/ill_event_ids.pkl', 'rb'))
        self.phases = phases
        self.response_xml = response_xml
        self.sampling_rate = sampling_rate
        self.highpass_filter = highpass_filter
        self.format = format
        self.dataset = dataset
        self.sort = sort
        self.num_stations = num_stations
        self.at_least_stations = min(10, num_stations)

        ## training
        self.training = training
        self.phase_width = phase_width
        self.polarity_width = polarity_width
        self.event_width = event_width
        self.stack_event = stack_event
        self.drop_channel = drop_channel
        self.sample_by_magnitude = sample_by_magnitude
        self.sample_by_distance = sample_by_distance
        if self.stack_event:
            #TODO: temp
            # base_dir is the directory of hdf5_file
            if self.hdf5_file is not None:
                self.base_dir = os.path.dirname(hdf5_file)
            elif self.data_list is not None:
                self.base_dir = os.path.dirname(self.data_list[0])
            elif self.data_path is not None:
                self.base_dir = data_path
            
            self.picks_dict = pd.read_csv(os.path.join(self.base_dir, "picks_train.csv"), usecols=['event_id', 'station_id', 'snr', 'phase_status'])
            self.events_dict = pd.read_csv(os.path.join(self.base_dir, "events_train.csv"), usecols=['event_id', 'magnitude', 'event_time', 'depth_km'])
            self.events_dict['year'] = self.events_dict['event_time'].apply(lambda x: int(x[:4]))
            self.picks_dict['snr'] = self.picks_dict['snr'].apply(lambda x: np.array([float(number) for number in x[1:-1].split()]))
            temp = self.picks_dict.groupby('event_id')['snr'].apply(lambda x: np.concatenate(x.values)).reset_index()
            temp['snr_mean'] = temp['snr'].apply(lambda x: np.mean(x[x!=0] if len(x[x!=0]) > 0 else 0))
            temp['snr_median'] = temp['snr'].apply(lambda x: np.median(x[x!=0] if len(x[x!=0]) > 0 else 0))
            self.events_dict = self.events_dict.merge(temp[['event_id', 'snr_mean', 'snr_median']], on='event_id', how='left'
                                                      )[['event_id', 'magnitude', 'depth_km', 'year', 'snr_mean', 'snr_median']].set_index('event_id').to_dict('index')
            # only use manual picks
            self.picks_dict = self.picks_dict[self.picks_dict['phase_status']=='manual']
            self.picks_dict = self.picks_dict[['event_id', 'station_id']]
            
            # remove the ill event ids
            self.picks_dict = self.picks_dict[~self.picks_dict['event_id'].isin(self.ill_event_ids)]
            
            self.e2sta_dict = self.picks_dict.groupby('event_id')['station_id'].apply(list).to_dict()
            self.sta2e_dict = self.picks_dict.groupby('station_id')['event_id'].apply(list).to_dict()
            del self.picks_dict

        ## prediction
        self.is_pad = is_pad
        self.cut_patch = cut_patch
        self.nt_cut = nt_cut
        self.nx = nx
        self.min_nt = min_nt
        self.min_nx = min_nx
        
        if self.training:
            print(f"{self.data_path}: {len(self.data_list)} files")
        else:
            print(
                os.path.join(data_path, f".{format}"),
                f": {len(self.data_list)} files",
            )

        self._data_len = self._count()
        self.negative_mask_ratio = 1.0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        data_list = self.data_list[worker_id::num_workers]
        if self.training:
            return iter(self.sample_train(data_list))
        else:
            return iter(self.sample_test(data_list))

    def _count(self):
        if not self.cut_patch:
            return len(self.data_list)
        else:
            if self.format == "h5":
                with fsspec.open(self.data_list[0], "rb") as fs:
                    with h5py.File(fs, "r") as fp:
                        nx, nt = fp["data"].shape
            elif "mseed" in self.format:
                nx = len(self.station_info)
                for fname in self.data_list:
                    try:
                        stream = obspy.read(fname)
                        break
                    except:
                        try:
                            stream = obspy.read(fname.split(',')[0])
                            break
                        except:
                            pass
                begin_time = min([st.stats.starttime for st in stream])
                end_time = max([st.stats.endtime for st in stream])
                nt = int((end_time - begin_time) * self.sampling_rate)
            return len(self.data_list) * ((nt - 1) // self.nt_cut + 1) * ((nx - 1) // self.nx + 1)
            # return len(self.data_list) * ((nt - 1) // self.nt_cut + 1)
    
    def __len__(self):
        if self.training:
            #return self._data_len
            if self._data_len * self.world_size < 100000:
                return self._data_len
            else:
                return self._data_len // 2
        else:
            return self._data_len
    
    
    def _read_training_h5(self, event_id, hdf5_fp):
        if event_id in self.ill_event_ids:
            return None
        if self.huggingface_cache:
            try:
                meta = self.huggingface_dataset[self.e2idx_dict[event_id]]
                if meta["num_stations"] < self.at_least_stations:
                    self.ill_event_ids.add(event_id)
                    return None
                for key in meta.keys():
                    if key in ["waveforms", "phase_pick", "phase_mask", "polarity", "polarity_mask", "event_center", "event_mask", "event_location", "event_location_mask", "station_location", "amplitude", "distance"]:
                        meta[key] = np.array(meta[key])
                assert meta["event_id"] == event_id, f"{meta['event_id']} != {event_id} in huggingface dataset"
                return meta
            except:
                self.ill_event_ids.add(event_id)
                return None
        
        num_stations = self.num_stations
        try:
            event = hdf5_fp[event_id]
        except KeyError:
            self.ill_event_ids.add(event_id)
            return None
        station_ids = list(event.keys())
        if num_stations is not None and len(station_ids) < self.at_least_stations:
            self.ill_event_ids.add(event_id)
            return None  
        #else:
        #    station_ids = np.random.choice(station_ids, num_stations, replace=False)
        event_attrs = event.attrs
        
        # exclude ridgecrest earthquake sequence
        event_time0 = datetime.fromisoformat(event.attrs["event_time"])
        if event_time0 >= datetime(2019,7,4,0,0,0) and event_time0 <= datetime(2019,7,10,0,0,0):
            if event.attrs["latitude"] >= 35.2 and event.attrs["latitude"] <= 36.2 and event.attrs["longitude"] >= -118.0 and event.attrs["longitude"] <= -117.0:
                return None
        #if event_time0 >= datetime(2019,7,4,0,0,0) and event_time0 <= datetime(2019,7,20,0,0,0):
        #    if event.attrs["latitude"] >= 35.0 and event.attrs["latitude"] <= 36.5 and event.attrs["longitude"] >= -118.5 and event.attrs["longitude"] <= -116.5:
        #        return None

        # avoid stations with P arrival equals S arrival
        is_sick = False
        auto_count = 0
        for sta_id in station_ids:
            if len(np.unique(mapping_phase_type(event[sta_id].attrs["phase_type"]))) < 2:
                auto_count += 1
                continue # only one phase type indicates automatic picks
            valid_manual_events = 0
            attrs = event[sta_id].attrs
            sta_phase_type = mapping_phase_type(attrs["phase_type"])
            for e_id in [event_id]:#sorted(np.unique(event[sta_id].attrs["event_id"])):
                if len(np.unique(sta_phase_type[attrs["event_id"]==e_id])) < 2:
                    continue
                
                p_picks = attrs["phase_index"][np.logical_and(sta_phase_type == "P", attrs["event_id"] == e_id)]
                s_picks = attrs["phase_index"][np.logical_and(sta_phase_type == "S", attrs["event_id"] == e_id)]
                if p_picks.min()>self.raw_nt or s_picks.min()>self.raw_nt:
                    continue # phases too late
                # try:
                #     if len(p_picks)!=len(s_picks):
                #         is_sick = True
                #         break
                # except:
                #     pass
                assert len(p_picks)==len(s_picks), f"{event_id}/{sta_id} manual P&S picks num don't match"
                if np.any(p_picks+5>=s_picks):
                    print(f"\n{event_id} when {sta_id} during {e_id} p pick arrives after s pick")
                    is_sick = True
                    break
                valid_manual_events += 1
            if is_sick:
                break    
            if valid_manual_events == 0:
                auto_count += 1
        if auto_count >= len(station_ids):
            is_sick = True
            print(f"\nevent {event_id} all auto picks")
        if is_sick:
            self.ill_event_ids.add(event_id)
            return None
        
        #reference_latitude = 0
        #reference_longitude = 0
        #for sta_id in station_ids:
        #    reference_latitude += event[sta_id].attrs["latitude"]
        #    reference_longitude += event[sta_id].attrs["longitude"]
        #reference_latitude/=len(station_ids)
        #reference_longitude/=len(station_ids)
        
        b, a = butter(4, 1/(self.sampling_rate/2), btype="highpass", analog=False)
        
        waveforms = np.zeros([len(station_ids), 3, self.raw_nt], dtype="float32")
        amplitude = np.zeros_like(waveforms)
        phase_pick = np.zeros_like(waveforms)
        phase_mask = np.zeros([len(station_ids), self.raw_nt])
        polarity = np.zeros_like(waveforms)
        polarity_mask = np.zeros([len(station_ids), self.raw_nt])
        event_center = np.zeros([len(station_ids), self.raw_feature_nt])
        event_mask = np.zeros([len(station_ids), self.raw_feature_nt])
        event_location = np.zeros([len(station_ids), 7, self.raw_feature_nt])
        event_location_mask = np.zeros([len(station_ids), self.raw_feature_nt])
        association_mask = np.zeros([len(station_ids), self.raw_feature_nt])
        main_event_center = np.zeros([len(station_ids), 3])
        station_location = np.zeros([len(station_ids), 3])
        # reference_point = np.array([reference_longitude, reference_latitude])
        main_stas = [] # index of stations with main event
        manual_stas = [] # index of stations with at least one manual pick pair 
        auto_stas = [] # index of stations with only automatic picks
        duration = []
        first_arrival = 1e10

        for i, sta_id in enumerate(station_ids):
            # trace_id = event_id + "/" + sta_id
            waveforms[i, :, :] = event[sta_id][:, :]
            amplitude[i, :, :] = event[sta_id][:, :]
            attrs = event[sta_id].attrs
            sta_phase_type = mapping_phase_type(attrs["phase_type"])
            if attrs["unit"][-6:] == "m/s**2":
                # integrate acceleration to velocity
                amplitude[i] = np.cumsum(amplitude[i]*attrs["dt_s"], axis=-1)
            # time cost too much, so we only use highpass filter
            # for j in range(3): 
            #     spline_i = LSQUnivariateSpline(np.arange(self.nt), amplitude[i, j, :], t=np.arange(self.nt, step=self.nt/2048)[1:], k=3)
            #     amplitude[i, j, :] -= spline_i(np.arange(self.nt))
            amplitude[i] = filtfilt(b, a, amplitude[i], axis=-1)
            
            ## station location
            # cartesian coordinate
            # station_location[i, 0] = round(
            #     (attrs["longitude"] - reference_longitude)
            #     * np.cos(np.radians(reference_latitude))
            #     * self.degree2km,
            #     4,
            # )
            # station_location[i, 1] = round((attrs["latitude"] - reference_latitude)
            #                                * self.degree2km, 4)
            # polar coordinate
            station_location[i, 0] = round(attrs["longitude"], 4)
            station_location[i, 1] = round(attrs["latitude"], 4)

            station_location[i, 2] =  round(-attrs["elevation_m"]/1e3, 2)
            
            ## only consider the station with manual picks for main event
            #if len(set(sta_phase_type[attrs['event_id']==event_id])) < 2:
            if len(set(sta_phase_type)) < 2:
                #print(f"\n{event_id}/{sta_id} only {attrs['event_id']}")
                auto_stas.append(i)
                phase_pick[i, 1:, :] = 0
                phase_pick[i, 0, :] = 1
                phase_mask[i, :] = 0
                event_center[i, :] = 0
                event_mask[i, :] = 0
                event_location[i, :, :] = 0
                event_location_mask[i, :] = 0
                duration.append([])
                continue
            
            manual_event_ids = []
            p_pick_stas = []
            s_pick_stas = []
            up_pick_stas = []
            dn_pick_stas = []
            duration_sta = []   

            event_time0 = datetime.fromisoformat(event_attrs["event_time"])
            for e_id in sorted(np.unique(attrs["event_id"])):
                try: # skip the event_id not in the dataset
                    e_attrs = hdf5_fp[e_id].attrs
                except KeyError:
                    continue
                if len(np.unique(sta_phase_type[attrs["event_id"]==e_id])) < 2:
                    continue
                
                p_picks = attrs["phase_index"][np.logical_and(sta_phase_type == "P", attrs["event_id"] == e_id)]
                s_picks = attrs["phase_index"][np.logical_and(sta_phase_type == "S", attrs["event_id"] == e_id)]
                if "phase_polarity" in attrs:
                    up = attrs["phase_index"][np.logical_and(attrs["phase_polarity"] == "U", attrs["event_id"] == e_id)]
                    dn = attrs["phase_index"][np.logical_and(attrs["phase_polarity"] == "D", attrs["event_id"] == e_id)]
                else:
                    up, dn = [], []
                if p_picks.min()>self.raw_nt or s_picks.min()>self.raw_nt:
                    continue # phases too late
                if len(p_pick_stas)!=0 and np.abs(p_picks-np.array(p_pick_stas)).min() < 30:
                    continue
                if len(s_pick_stas)!=0 and np.abs(s_picks-np.array(s_pick_stas)).min() < 30:
                    continue

                ## TODO: how to deal with multiple phases
                # center = (attrs["phase_index"][::2] + attrs["phase_index"][1::2])/2.0
                ## assuming only one event with both P and S picks
                assert len(p_picks)==len(s_picks), f'{event_id} {sta_id}: p_picks:{p_picks}, s_picks:{s_picks}'
                tmp_min = np.min(p_picks)
                tmp_max = np.max(s_picks)
                duration_sta.append([tmp_min, max(tmp_min + 3, tmp_max + 2 * (tmp_max-tmp_min))])
                first_arrival = min(first_arrival, tmp_min)
                c0 = ((p_picks) + (s_picks)) / 2.0 # phase center
                c0_width = max(((s_picks - p_picks) * self.sampling_rate / 200.0).max(), 10)
                # c0_width = ((s_picks - p_picks) * self.sampling_rate / 200.0).max() # min=160
                assert c0_width>0
                assert len(c0)==1, f"{event_id}/{sta_id} multiple phase centers at {e_id}: {c0}"
                assert c0[0]<self.raw_nt
                c0 = c0/self.feature_scale
                assert c0[0]<self.raw_feature_nt
                c0_width = c0_width/self.feature_scale
                #assert c0_width>=160/self.feature_scale
                c0_int = c0.astype(np.int32)
                assert c0_int[0]<self.raw_feature_nt
                assert abs(c0-c0_int)[0]<1
                # cartesian
                # dx = round(
                #     (e_attrs["longitude"] - attrs["longitude"])
                #     * np.cos(np.radians(reference_latitude))
                #     * self.degree2km,
                #     4,
                # )
                # dy = round(
                #     (e_attrs["latitude"] - attrs["latitude"])
                #     * self.degree2km,
                #     4,
                # )
                dz = round(
                    e_attrs["depth_km"] + attrs["elevation_m"] / 1e3,
                    2,
                )
                # dist = np.sqrt(dx**2 + dy**2)
                # azimuth = np.arctan2(dy, dx)
                dlon = e_attrs["longitude"] - attrs["longitude"]
                dlat = e_attrs["latitude"] - attrs["latitude"]
                a = np.sin(np.radians(dlat) / 2) ** 2 \
                    + np.cos(np.radians(attrs["latitude"])) * np.cos(np.radians(e_attrs["latitude"])) * np.sin(np.radians(dlon) / 2) ** 2
                dist = round(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 6371, 4)
                y = np.sin(np.radians(dlon)) * np.cos(np.radians(e_attrs["latitude"]))
                x = np.cos(np.radians(attrs["latitude"])) * np.sin(np.radians(e_attrs["latitude"])) \
                    - np.sin(np.radians(attrs["latitude"])) * np.cos(np.radians(e_attrs["latitude"])) * np.cos(np.radians(dlon))
                azimuth = round(np.arctan2(y, x), 4)
                                    
                magnitude = round(e_attrs["magnitude"], 2)
            
                begin_time_index_shift = (int(round(
                    (datetime.fromisoformat(e_attrs["event_time"]) - event_time0).total_seconds()
                    / attrs["dt_s"]
                )) if e_id != event_id else 0
                )
                t0 = e_attrs["event_time_index"] + begin_time_index_shift
            
                event_center_eid, mask, param_mask = generate_event_label(
                    # [c0 / self.feature_scale],
                    c0_int,
                    # label_width=[
                    #     c0_width,
                    # ],
                    label_width=self.event_width,
                    nt=self.raw_feature_nt,
                    # nt=self.nt,
                )
                #mask = event_center_eid >= 0.5
                #mask = np.zeros_like(event_center_eid, dtype=bool)
                #mask[max(0, c0_int[0]-self.event_width[0]//2):min(self.raw_feature_nt, c0_int[0]+self.event_width[0]//2)] = True
                #print(f"{event_id}/{sta_id}: {e_id} c0:{c0}, c0_int:{c0_int}, mask:{np.sum(mask)}")
                assert np.max(event_center[i, :]+event_center_eid)<=1, f"{event_id}/{sta_id} event_center overlap when {e_id}: {(event_center[i, :]+event_center_eid).max()}"#{event_center[i, :]+event_center_eid}"
                if not np.all(event_location[i, :, param_mask]==0):
                    print(f"\n{event_id}/{sta_id} event_location overlap when {e_id}: previous {np.nonzero(event_location_mask[i, :])}, current {np.nonzero(param_mask)}")
                    #continue
                
                event_center[i, :] += event_center_eid
                event_mask[i, :] = np.logical_or(event_mask[i, :], mask)
                event_location[i, 0][param_mask] = ((
                    self.feature_scale * np.arange(self.raw_feature_nt) - t0
                )/self.sampling_rate)[param_mask]
                # print(event_location[i, 1:, param_mask].shape, event_location.shape, event_location[i][1:, param_mask].shape)
                event_location[i][1:, param_mask] = np.array([dist, azimuth, dz, magnitude, (c0-c0_int)[0], c0_width])[:, np.newaxis]
                event_location_mask[i, :] = np.logical_or(event_location_mask[i, :], param_mask)
                if e_id == event_id:
                    main_event_center[i, :] = np.array([c0_int[0], round(attrs["longitude"], 4), round(attrs["latitude"], 4)])
                    association_mask[i, max(0, c0_int[0]-self.event_width[0]//2) : min(self.raw_feature_nt, c0_int[0]+self.event_width[0]//2)] = 1.0
                    #association_mask[i, :] = event_center_eid
                    main_stas.append(i)

                manual_event_ids.append(e_id)
                p_pick_stas.append(p_picks)
                s_pick_stas.append(s_picks)
                up_pick_stas.append(up)
                dn_pick_stas.append(dn)
            duration.append(duration_sta)

            if len(manual_event_ids) == 0:
                auto_stas.append(i)
                phase_pick[i, 1:, :] = 0
                phase_pick[i, 0, :] = 1
                phase_mask[i, :] = 0
                polarity[i, 1:, :] = 0
                polarity[i, 0, :] = 1
                polarity_mask[i, :] = 0
                event_center[i, :] = 0
                event_mask[i, :] = 0
                event_location[i, :, :] = 0
                event_location_mask[i, :] = 0
            else:
                manual_stas.append(i)
                phase_pick[i, :, :], phase_mask[i, :] = generate_phase_label([np.concatenate(p_pick_stas), np.concatenate(s_pick_stas)], label_width=self.phase_width, nt=self.raw_nt)
                polarity[i, :, :], polarity_mask[i, :] = generate_phase_label([np.concatenate(up_pick_stas), np.concatenate(dn_pick_stas)], label_width=self.polarity_width, nt=self.raw_nt)

        # std = np.std(waveforms, axis=-1, keepdims=True)
        # std[std == 0] = 1.0
        # waveforms = (waveforms - np.mean(waveforms, axis=-1, keepdims=True)) / std
        # waveforms = waveforms.astype(np.float32)
        if main_stas:
            # first the index of main event, then the index of other manual events
            manual_stas = main_stas + [x for x in manual_stas if x not in set(main_stas)]
        len_manual = len(manual_stas)
        assert len_manual>0, f"event {event_id} no manual picks"
        assert len_manual==np.any(event_location_mask>0, axis=1).sum(), f"{event_id} manual picks num don't match, len_manual:{len_manual}, manual_stas:{manual_stas}, event_location_mask:{np.any(event_location_mask>0, axis=1)}"
        assert first_arrival!=1e10, f"{event_id} first_arrival not set"
        manual_cluster_index = np.array(manual_stas+auto_stas)
        # if abs(it) <= 160:
        #     print(f"\n{event_id} data cut off shift smaller than {160/self.sampling_rate} s: {it/self.sampling_rate} s")
        waveforms = waveforms[manual_cluster_index]
        amplitude = amplitude[manual_cluster_index]
        phase_pick = phase_pick[manual_cluster_index]
        phase_mask = phase_mask[manual_cluster_index]
        polarity = polarity[manual_cluster_index]
        polarity_mask = polarity_mask[manual_cluster_index]
        event_center = event_center[manual_cluster_index]
        event_mask = event_mask[manual_cluster_index]
        event_location = event_location[manual_cluster_index]
        event_location_mask = event_location_mask[manual_cluster_index]
        association_mask = association_mask[manual_cluster_index]
        main_event_center = main_event_center[manual_cluster_index]
        station_location = station_location[manual_cluster_index]
        sta_ids = [station_ids[i] for i in manual_cluster_index]
        duration = [duration[i] for i in manual_cluster_index]

        assert np.any(phase_pick[:, 1:, :]>0), f"{event_id} no manual phase picks"
        assert np.any(event_center>0), f"{event_id} no manual event picks"
        meta = {
            "waveforms": waveforms,
            "phase_pick": phase_pick,
            "phase_mask": phase_mask,
            "polarity": polarity,
            "polarity_mask": polarity_mask,
            "event_center": event_center,
            "event_mask": event_mask,
            "event_location": event_location,
            "event_location_mask": event_location_mask,
            "association_mask": association_mask,
            "station_location": station_location,
            "amplitude": amplitude,
            "main_event_center": main_event_center,
            "num_stations": waveforms.shape[0],
            "num_manual_stations": len_manual,
            "event_id": event_id,
            "station_ids": sta_ids,
            "duration": duration,
            "first_arrival": first_arrival,
            "distance": np.array([event[sta_id].attrs["distance_km"] for sta_id in sta_ids])
        }
        
        return meta


    def sample_train(self, data_list):
        if self.format == "h5":
            hdf5_fp = h5py.File(self.hdf5_file, "r", libver="latest", swmr=True, locking=False)
        random_idx = np.random.permutation(len(data_list))
        random_idx = iter(random_idx)
        while True:
            #event_id = np.random.choice(data_list)
            try:
                event_id = data_list[next(random_idx)]
            except StopIteration:
                random_idx = np.random.permutation(len(data_list))
                random_idx = iter(random_idx)
                event_id = data_list[next(random_idx)]
            
            if self.sample_by_magnitude:
                if not sample_by_magnitude(self.hdf5_fp[event_id].attrs["magnitude"], critical_magnitude=2, sample_rate=0.6, min_magnitude=0.3):
                    continue
            # if True:
            try:
                meta = self._read_training_h5(event_id, hdf5_fp)
            except Exception as e:
                print(f"Error reading {event_id}:\n{e}")
                continue

            if meta is None:
                continue
            
            if self.sample_by_distance:
                idx = sample_by_distance(meta['distance'], num_stations=self.num_stations, num_manual_stations=meta['num_manual_stations'],
                                         critical_distance=50, min_distance=5, sample_rate=0.5)
                assert len(idx) == self.num_stations, f"{event_id} sample_by_distance failed"
                
                meta = {
                    "waveforms": meta['waveforms'][idx],
                    "phase_pick": meta['phase_pick'][idx],
                    "phase_mask": meta['phase_mask'][idx],
                    "polarity": meta['polarity'][idx],
                    "polarity_mask": meta['polarity_mask'][idx],
                    "event_center": meta['event_center'][idx],
                    "event_mask": meta['event_mask'][idx],
                    "event_location": meta['event_location'][idx],
                    "event_location_mask": meta['event_location_mask'][idx],
                    "association_mask": meta['association_mask'][idx],
                    "station_location": meta['station_location'][idx],
                    "amplitude": meta['amplitude'][idx],
                    "main_event_center": meta['main_event_center'][idx],
                    "num_stations": meta['waveforms'].shape[0],
                    "num_manual_stations": min(meta['num_manual_stations'], self.num_stations),
                    "event_id": meta['event_id'],
                    "station_ids": [meta['station_ids'][i] for i in idx],
                    "duration": [meta['duration'][i] for i in idx],
                    "first_arrival": meta['first_arrival'],
                }
            else:
                meta = {
                    "waveforms": meta['waveforms'][:self.num_stations],
                    "phase_pick": meta['phase_pick'][:self.num_stations],
                    "phase_mask": meta['phase_mask'][:self.num_stations],
                    "polarity": meta['polarity'][:self.num_stations],
                    "polarity_mask": meta['polarity_mask'][:self.num_stations],
                    "event_center": meta['event_center'][:self.num_stations],
                    "event_mask": meta['event_mask'][:self.num_stations],
                    "event_location": meta['event_location'][:self.num_stations],
                    "event_location_mask": meta['event_location_mask'][:self.num_stations],
                    "association_mask": meta['association_mask'][:self.num_stations],
                    "station_location": meta['station_location'][:self.num_stations],
                    "amplitude": meta['amplitude'][:self.num_stations],
                    "main_event_center": meta['main_event_center'][:self.num_stations],
                    "num_stations": meta['waveforms'].shape[0],
                    "num_manual_stations": min(meta['num_manual_stations'], self.num_stations),
                    "event_id": meta['event_id'],
                    "station_ids": meta['station_ids'][:self.num_stations],
                    "duration": meta['duration'][:self.num_stations],
                    "first_arrival": meta['first_arrival'],
                }
            
            if self.stack_event and random.random() < 0.9:
                try:
                    #if random.random() < 0.4:
                    #    # copy the meta into meta2
                    #    meta2 = copy.deepcopy(meta)
                    #else:
                    selected_event_ids = sample_event_for_stacking(
                        event_ids = event_id,
                        e2sta_dict = self.e2sta_dict,
                        sta2e_dict = self.sta2e_dict,
                        events_dict = self.events_dict,
                        min_shared_sta=3,
                        magnitude_range=None,
                        returns=7,
                    )
                    meta2 = None
                    stack_times = 0
                    max_stack_times = random.choice([1]*4 + [2]*3 + [3]*2 + [4])#np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])#
                    for e_id in selected_event_ids:
                        meta2 = self._read_training_h5(e_id, hdf5_fp)
                        if meta2 is not None:
                            meta, is_failed = stack_event(meta, meta2, max_shift=64*4, feature_scale=self.feature_scale, is_extra=stack_times>0)
                            stack_times += 1 if not is_failed else 0
                            if stack_times >= max_stack_times:
                                break
                            #print(f"event {event_id} succeed")
                    #else:
                    #    print(f"event {event_id} no valid meta2")
                except Exception as e:
                    print(f"Error stacking {event_id}:\n{e}")
            else:
                stack_times = 0
            #    print(f"event {event_id} not stacked")
            
            negative_mask_ratio = self.negative_mask_ratio.value if not isinstance(self.negative_mask_ratio, float) else self.negative_mask_ratio
            meta['phase_mask'] = add_random_negative_samples(meta['phase_mask'], random_mask_rate=0.1*negative_mask_ratio, random_mask_unit=32)
            meta['event_mask'] = add_random_negative_samples(meta['event_mask'], random_mask_rate=0.2*negative_mask_ratio, random_mask_unit=8)
            if not np.any(meta['association_mask']>0):
                print(f"{event_id} no association mask {meta['main_event_center']=}")
                continue
            meta_cutted, it = cut_data(meta, nt=self.nt, feature_scale=self.feature_scale, if_reverse_roll=False, base_phase_points=int(64*(1+stack_times/8))) #TODO: allow reverse roll?
            
            # random selected one main event center as prompt
            if len(meta_cutted['main_event_center'])==0:
                print(f"{event_id} no main event center")
                continue
            prompt = random.choice(meta_cutted['main_event_center'])
            # add random time shift
            prompt[0] = np.clip(prompt[0]+np.random.randint(-int(self.event_width[0]*0.3), int(self.event_width[0]*0.3)), 0, self.feature_nt-1)
            #prompt = np.concatenate([prompt, np.array([1])], axis=0) # 1 as positive point
            prompt = prompt[np.newaxis, :]

            if self.sort:
                D = np.sqrt(((meta["station_location"][:, np.newaxis, :2] -  meta["station_location"][np.newaxis, :, :2])**2).sum(axis=-1))
                Tcsr = minimum_spanning_tree(D)
                index = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                
                for k in meta_cutted.keys():
                    meta_cutted[k] = meta_cutted[k][index]
                meta["station_location"] = meta["station_location"][index]
            
            if self.num_stations > self.at_least_stations and meta['num_stations'] < self.num_stations:
                lack_num = self.num_stations - meta['num_stations']
                idx = np.random.choice(meta['num_stations'], lack_num, replace=True)
                meta_cutted['waveforms'] = np.concatenate([meta_cutted['waveforms'], meta_cutted['waveforms'][idx]], axis=0)
                meta_cutted['amplitude'] = np.concatenate([meta_cutted['amplitude'], meta_cutted['amplitude'][idx]], axis=0)
                empty_pick = np.zeros((lack_num, *meta_cutted['phase_pick'].shape[1:]))
                empty_pick[:, 0, :] = 1
                meta_cutted['phase_pick'] = np.concatenate([meta_cutted['phase_pick'], empty_pick], axis=0)
                meta_cutted['phase_mask'] = np.concatenate([meta_cutted['phase_mask'], np.zeros((lack_num, *meta_cutted['phase_mask'].shape[1:]))], axis=0)
                meta_cutted['polarity'] = np.concatenate([meta_cutted['polarity'], empty_pick], axis=0)
                meta_cutted['polarity_mask'] = np.concatenate([meta_cutted['polarity_mask'], np.zeros((lack_num, *meta_cutted['polarity_mask'].shape[1:]))], axis=0)
                meta_cutted['event_center'] = np.concatenate([meta_cutted['event_center'], np.zeros((lack_num, *meta_cutted['event_center'].shape[1:]))], axis=0)
                meta_cutted['event_mask'] = np.concatenate([meta_cutted['event_mask'], np.zeros((lack_num, *meta_cutted['event_mask'].shape[1:]))], axis=0)
                meta_cutted['event_location'] = np.concatenate([meta_cutted['event_location'], np.zeros((lack_num, *meta_cutted['event_location'].shape[1:]))], axis=0)
                meta_cutted['event_location_mask'] = np.concatenate([meta_cutted['event_location_mask'], np.zeros((lack_num, *meta_cutted['event_location_mask'].shape[1:]))], axis=0)
                meta_cutted['association_mask'] = np.concatenate([meta_cutted['association_mask'], np.zeros((lack_num, *meta_cutted['association_mask'].shape[1:]))], axis=0)
                meta['station_location'] = np.concatenate([meta['station_location'], meta['station_location'][idx]], axis=0)
                meta['num_stations'] = self.num_stations
            
            meta = reorder_keys({
                "data": torch.from_numpy(meta_cutted["waveforms"]).float(),
                "amplitude": torch.from_numpy(meta_cutted["amplitude"]).float(),
                "phase_pick": torch.from_numpy(meta_cutted["phase_pick"]).float(),
                "phase_mask": torch.from_numpy(meta_cutted["phase_mask"]).float(),
                "polarity": torch.from_numpy(meta_cutted["polarity"]).float(),
                "polarity_mask": torch.from_numpy(meta_cutted["polarity_mask"]).float(),
                "event_center": torch.from_numpy(meta_cutted["event_center"]).float(),
                "event_mask": torch.from_numpy(meta_cutted["event_mask"]).float(),
                "event_location": torch.from_numpy(meta_cutted["event_location"]).float(),
                "event_location_mask": torch.from_numpy(meta_cutted["event_location_mask"]).float(),
                "association_mask": torch.from_numpy(meta_cutted["association_mask"]).unsqueeze(1).float(),
                "station_location": torch.from_numpy(meta["station_location"]).float(),
                "prompt": torch.from_numpy(prompt).float(),
                "num_stations": meta['num_stations'],
                "num_manual_stations": meta['num_manual_stations'],
                #"station_ids": meta['station_ids'],
                "event_id": event_id,
            })
            
            yield meta
            
    
    def read_mseed(self, fname, response_path=None, response_xml=None, highpass_filter=0.0, sampling_rate=100):
        try:
            stream = obspy.Stream()
            for tmp in fname.split(","):
                meta = obspy.read(tmp, format="MSEED")
                if response_path is not None:
                    inv = obspy.read_inventory(os.path.join(response_path, f"{meta[0].id.split('.')[0]}.{meta[0].id.split('.')[1]}") + ".xml")
                    meta = meta.remove_sensitivity(inv)
                stream += meta
                # stream += obspy.read(tmp)
            stream = stream.merge(fill_value="latest")
            if (response_path is None) and (response_xml is not None):
                response = obspy.read_inventory(response_xml)
                stream = stream.remove_sensitivity(response)
        except Exception as e:
            print(f"Error reading {fname}:\n{e}")
            return None

        tmp_stream = obspy.Stream()
        for trace in stream:
            if len(trace.data) < 10:
                continue

            ## interpolate to 100 Hz
            if trace.stats.sampling_rate != sampling_rate:
                logging.warning(f"Resampling {trace.id} from {trace.stats.sampling_rate} to {sampling_rate} Hz")
                try:
                    trace = trace.interpolate(sampling_rate, method="linear")
                except Exception as e:
                    print(f"Error resampling {trace.id}:\n{e}")

            trace = trace.detrend("demean")

            ## detrend
            # try:
            #     trace = trace.detrend("spline", order=2, dspline=5 * trace.stats.sampling_rate)
            # except:
            #     logging.error(f"Error: spline detrend failed at file {fname}")
            #     trace = trace.detrend("demean")

            if highpass_filter > 0:
                trace = trace.filter("highpass", freq=highpass_filter)

            tmp_stream.append(trace)

        if len(tmp_stream) == 0:
            return None
        stream = tmp_stream

        begin_time = min([st.stats.starttime for st in stream])
        end_time = max([st.stats.endtime for st in stream])
        stream = stream.trim(begin_time, end_time, pad=True, fill_value=0)

        comp = ["3", "2", "1", "E", "N", "Z"]
        comp2idx = {"3": 0, "2": 1, "1": 2, "E": 0, "N": 1, "Z": 2}

        station_ids = defaultdict(list)
        for tr in stream:
            station_ids[tr.id[:-1]].append(tr.id[-1])
            if tr.id[-1] not in comp:
                print(f"Unknown component {tr.id[-1]}")

        station_keys = sorted(list(station_ids.keys()))
        station_location = np.zeros([len(station_keys), 3], dtype=np.float32)
        for i, sta in enumerate(station_keys):
            station_location[i, 0] = self.station_info.loc[sta]["longitude"]
            station_location[i, 1] = self.station_info.loc[sta]["latitude"]
            station_location[i, 2] = round(-self.station_info.loc[sta]["elevation_m"]/1e3, 2)

        nx = len(station_ids)
        nt = len(stream[0].data)
        data = np.zeros([3, nt+((32- nt%32)%32), nx], dtype=np.float32) # the length of data should be multiple of 32
        amplitude = np.zeros_like(data)
        for i, sta in enumerate(station_keys):
            for c in station_ids[sta]:
                j = comp2idx[c]

                if len(stream.select(id=sta + c)) == 0:
                    print(f"Empty trace: {sta+c} {begin_time}")
                    continue

                trace = stream.select(id=sta + c)[0]

                ## accerleration to velocity
                if sta[-1] == "N":
                    trace = trace.integrate().filter("highpass", freq=1.0) #TODO: why 1.0hz highpass?
                tmp = trace.data.astype("float32")
                data[j, : len(tmp), i] = tmp[:nt]
                amplitude[j, : len(tmp), i] = tmp[:nt] * 1e6 # unit from m/s to 10e-6 m/s
        
        if self.sort and len(station_keys) > 5:
            if 'group' not in self.station_info.columns:
                #D = np.sqrt(((station_location[:, np.newaxis, :2] -  station_location[np.newaxis, :, :2])**2).sum(axis=-1))
                #Tcsr = minimum_spanning_tree(D)
                #index = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                groups = optimal_transport_clustering(station_location[:, :2], k=3)#math.ceil(nx/self.nx) if nx/self.nx>1 else 2)
                index = np.lexsort([station_location[:, 1], station_location[:, 0], groups])
                #index = np.argsort(groups)
            else:
                # let the stations follow the order in self.stations
                ordered_ids = self.station_info.index.to_list()
                index = []
                for i in ordered_ids:
                    if i in station_keys:
                        index.append(station_keys.index(i))
                index = np.array(index)
            
            data = data[:, :, index]
            amplitude = amplitude[:, :, index]
            station_keys = [station_keys[i] for i in index]
            station_location = station_location[index]

        return {
            "data": torch.from_numpy(data),
            "amplitude": torch.from_numpy(amplitude),
            "station_id": station_keys,
            "station_location": torch.from_numpy(station_location).float(),
            "begin_time": begin_time.datetime.isoformat(timespec="milliseconds"),
            "dt_s": 1 / sampling_rate,
        }
            
    def read_hdf5(self, event_id):
        if self.hdf5_fp is None:
            raise ValueError("hdf5_fp is not opened")
        else:
            hdf5_fp = self.hdf5_fp
            event = hdf5_fp[event_id]
            station_ids = list(event.keys())
            if len(station_ids)==0:
                return None
            event_attrs = event.attrs

            # for sta_id in station_ids:
            #     attrs = event[sta_id].attrs
            #     for e_id in sorted(np.unique(event[sta_id].attrs["event_id"])):
            #         if len(np.unique(attrs["phase_type"][attrs["event_id"]==e_id])) < 2:
            #             continue
            #         
            #         p_picks = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "P", attrs["event_id"] == e_id)]
            #         s_picks = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "S", attrs["event_id"] == e_id)]
            #         assert len(p_picks)==len(s_picks), f"{event_id}/{sta_id} manual P&S picks num don't match"
            #         if np.any(p_picks+5>=s_picks):
            #             print(f"\n{event_id} when {sta_id} during {e_id} p pick arrives after s pick")
            #             return None
            
            begin_time = event_attrs["begin_time"]
            dt_s = event[station_ids[0]].attrs["dt_s"]
            # data shape
            nt = event[station_ids[0]].shape[1]
            nt_feature = 32
            b, a = butter(4, 0.1, btype="highpass", analog=False)
            waveforms = np.zeros([3, nt+((nt_feature-(nt%nt_feature))%nt_feature), len(station_ids)], dtype="float32")
            amplitude = np.zeros_like(waveforms)
            station_location = np.zeros([len(station_ids), 3])
            for i, sta_id in enumerate(station_ids):
                # trace_id = event_id + "/" + sta_id
                waveforms[:, :nt, i] = event[sta_id][:, :]
                amplitude[:, :nt, i] = event[sta_id][:, :]
                attrs = event[sta_id].attrs
                if attrs["unit"][-6:] == "m/s**2":
                    # integrate acceleration to velocity
                    amplitude[:, :, i] = np.cumsum(amplitude[:, :, i]*attrs["dt_s"], axis=1)
                # elif attrs["unit"][-3:] == "m/s": #TODO: temp
                #     amplitude[:, :, i] = amplitude[:, :, i] * 10e4
                # for j in range(3): 
                #     spline_i = LSQUnivariateSpline(np.arange(amplitude.shape[1]), amplitude[j, :, i], t=np.arange(amplitude.shape[1], step=amplitude.shape[1]/2048)[1:], k=3)
                #     amplitude[j, :, i] -= spline_i(np.arange(amplitude.shape[1]))
                amplitude[:, :, i] = filtfilt(b, a, amplitude[:, :, i], axis=1)

                ## station location
                # cartesian coordinate
                # station_location[i, 0] = round(
                #     (attrs["longitude"] - reference_longitude)
                #     * np.cos(np.radians(reference_latitude))
                #     * self.degree2km,
                #     4,
                # )
                # station_location[i, 1] = round((attrs["latitude"] - reference_latitude)
                #                                * self.degree2km, 4)
                # polar coordinate
                station_location[i, 0] = round(attrs["longitude"], 4)
                station_location[i, 1] = round(attrs["latitude"], 4)

                station_location[i, 2] = round(-attrs["elevation_m"]/1e3, 2)

            # std = np.std(waveforms, axis=1, keepdims=True)
            # std[std == 0] = 1.0
            # waveforms = (waveforms - np.mean(waveforms, axis=1, keepdims=True)) / std
            # waveforms = waveforms.astype(np.float32)
            
            ## cut for test
            #if len(station_ids) > 100:
            #    is_manual = ["manual" in hdf5_fp[event_id][sta_id].attrs["phase_status"] for sta_id in station_ids]
            #    # first choose manual picks, then choose auto picks
            #    idx = np.where(is_manual)[0]
            #    if len(idx) > 100:
            #        idx = np.random.choice(idx, 100, replace=False)
            #    else:
            #        lack_num = 100 - len(idx)
            #        idx = np.concatenate([idx, np.random.choice(np.where(~is_manual)[0], lack_num, replace=False)])
            #    waveforms = waveforms[:, :, idx]
            #    amplitude = amplitude[:, :, idx]
            #    station_ids = [station_ids[i] for i in idx]
            #    station_location = station_location[idx]
            
            if self.sort and len(station_ids)>5:
                D = np.sqrt(((station_location[:, np.newaxis, :2] -  station_location[np.newaxis, :, :2])**2).sum(axis=-1))
                Tcsr = minimum_spanning_tree(D)
                index = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                
                waveforms = waveforms[:, :, index]
                amplitude = amplitude[:, :, index]
                station_ids = [station_ids[i] for i in index]
                station_location = station_location[index]
            
            if self.is_pad and len(station_ids) < self.num_stations:
                pad_indices = np.random.choice(len(station_ids), self.num_stations-len(station_ids), replace=True)
                # waveforms = np.concatenate([waveforms, np.zeros_like(waveforms[:, :, pad_indices])], axis=-1)
                # amplitude = np.concatenate([amplitude, np.zeros_like(amplitude[:, :, pad_indices])], axis=-1)
                # station_location = np.concatenate([station_location, np.zeros_like(station_location[pad_indices])], axis=0)
                waveforms = np.concatenate([waveforms, waveforms[:, :, pad_indices]], axis=-1)
                amplitude = np.concatenate([amplitude, amplitude[:, :, pad_indices]], axis=-1)
                station_location = np.concatenate([station_location, station_location[pad_indices]], axis=0)

        return {
            "data": torch.from_numpy(waveforms).float(),
            "amplitude": torch.from_numpy(amplitude).float(),
            "station_id": station_ids,
            "station_location": torch.from_numpy(station_location).float(),
            "begin_time": begin_time,
            "dt_s": dt_s,
        }
        
    def read_das_hdf5(self, fname):
        meta = {}
        with fsspec.open(fname, "rb") as fs:
            try:
                with h5py.File(fs, "r") as fp:
                    # raw_data = fp["data"][()]  # [nt, nx]
                    raw_data = fp["data"][:, :].T  # (nx, nt) -> (nt, nx)
                    raw_data = raw_data - np.mean(raw_data, axis=0, keepdims=True)
                    raw_data = raw_data - np.median(raw_data, axis=1, keepdims=True)
                    std = np.std(raw_data, axis=0, keepdims=True)
                    std[std == 0] = 1.0
                    raw_data = raw_data / std
                    attrs = fp["data"].attrs
                    nt, nx = raw_data.shape
                    data = np.zeros([3, nt, nx], dtype=np.float32)
                    data[-1, :, :] = raw_data[:, :]
                    meta["waveform"] = torch.from_numpy(data)
                    if "station_id" in attrs:
                        station_id = attrs["station_name"]
                    else:
                        station_id = [f"{i}" for i in range(nx)]
                    meta["station_location"] = torch.from_numpy(
                        np.zeros([nx, 3], dtype=np.float32)
                    )
                    meta["station_location"][:, 0] = torch.arange(nx)/1000 # TODO: get real location
                    meta["station_id"] = station_id
                    meta["begin_time"] = attrs["begin_time"]
                    meta["dt_s"] = attrs["dt_s"]
            except Exception as e:
                print(f"Error reading {fname}:\n{e}")
                return None
        return meta
    
    def sample_test(self, data_list):
        for fname in data_list:
            if "mseed" in self.format:
                meta = self.read_mseed(
                    fname,
                    response_xml=self.response_xml,
                    highpass_filter=self.highpass_filter,
                    sampling_rate=self.sampling_rate,
                )
            elif (self.format == "h5") and (self.dataset == "seismic_network"):
                try:
                    meta = self.read_hdf5(fname)
                except Exception as e:
                    print(f"Error reading {fname}:\n{type(e)} {e}")
                    continue
            elif (self.format == "h5") and (self.dataset == "das"):
                meta = self.read_das_hdf5(fname)
            else:
                raise NotImplementedError
            if meta is None:
                continue
            if self.format == "h5":
                meta["file_name"] = fname
            elif "mseed" in self.format:
                if "hour" in self.format:
                    meta["file_name"] = fname.split("/")[-3]+"/"+fname.split("/")[-2]
                else:
                    meta["file_name"] = os.path.basename(fname).split(".")[0]

            if not self.cut_patch:
                yield meta
            else:
                _, nt, nx = meta["data"].shape
                for i in list(range(0, nt, self.nt_cut)):
                    for j in list(range(0, nx, self.nx)):
                        if nt - i <=32:
                            continue # too short
                        assert meta["data"][:, i : i + self.nt_cut, j : j + self.nx].shape[1] > 256, f"Error: {fname} i:{i} j:{j}, nt:{nt}, nt_cut:{self.nt_cut}, too short {meta['data'][:, i : i + self.nt_cut, j : j + self.nx].shape[1]}, please check the data or adjust cut_patch parameters"
                        yield {
                            "data": meta["data"][:, i : i + self.nt_cut, j : j + self.nx],
                            "amplitude": meta["amplitude"][:, i : i + self.nt_cut, j : j + self.nx],
                            "station_location": meta["station_location"][j : j + self.nx, :], # [nx, 3]
                            "station_id": meta["station_id"][j : j + self.nx],
                            "begin_time": (
                                datetime.fromisoformat(meta["begin_time"]) + timedelta(seconds=i * meta["dt_s"])
                            ).isoformat(timespec="milliseconds"),
                            "begin_time_index": i,
                            "dt_s": meta["dt_s"],
                            "file_name": meta["file_name"] + f"_{i:04d}_{j:04d}",
                        }



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SeismicNetworkIterableDataset("/Users/weiqiang/Research/EQNet/datasets/NCEDC/ncedc_event.h5")
    for x in dataset:
        # print(x)
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))
        for i in range(x["data"].shape[-1]):
            axes[0].plot((x["data"][-1, :, i]) / torch.std(x["data"][-1, :, i]) / 10 + i)
            axes[1].plot(x["amplitude"][-1, :, i] / torch.std(x["amplitude"][-1, :, i]) / 10 + i)

            axes[2].plot(x["phase_pick"][1, :, i] + i)
            axes[2].plot(x["phase_pick"][2, :, i] + i)

            axes[3].plot(x["event_center"][:, i] + i - 0.5)
            # axes[2].scatter(x["event_location"][0, :, i], x["event_location"][1, :, i])

            axes[4].plot(x["event_location"][0, :, i] / 10 + i)

            t = np.arange(x["event_location"].shape[1])[x["event_location_mask"][:, i] == 1]
            axes[5].plot(t, x["event_location"][1, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")
            axes[5].plot(t, x["event_location"][2, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")
            axes[5].plot(t, x["event_location"][3, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")

        plt.savefig("test.png")
        plt.show()

        raise
