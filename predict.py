import logging
import os
import time
import math
from contextlib import nullcontext
from glob import glob
import pickle

import numpy as np
from scipy.ndimage import binary_opening, binary_closing
import matplotlib
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data
from datetime import datetime, timedelta
import wandb
from tqdm import tqdm
import networkx as nx
from collections import defaultdict

import eqnet
import utils
from eqnet.data import DASIterableDataset, SeismicTraceIterableDataset, SeismicNetworkIterableDataset
from eqnet.models.unet import moving_normalize
from eqnet.utils import (
    detect_peaks,
    extract_events,
    extract_picks,
    merge_events,
    merge_patch,
    merge_picks,
    plot_das,
    plot_phasenet,
    plot_eqnet,
    plot_phasenet_plus,
)

# mp.set_start_method("spawn", force=True)
matplotlib.use("agg")
logger = logging.getLogger()

# cuda:1
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

def time_morphology_filter(mask, t_kernel_size=10):
    """
    仅在 t 方向进行形态学处理，使用结构元素形状 (t_kernel_size, 1)；
    保证 x 方向不受影响。
    """
    # add support for any dimensions, only apply filter in time dimension
    dims = mask.ndim
    struct = np.ones((t_kernel_size, 1)) if dims == 2 else np.ones([1]*(dims-2) + [t_kernel_size, 1])
    opened_mask = binary_opening(mask, structure=struct)
    closed_mask = binary_closing(opened_mask, structure=struct)
    return closed_mask
    #closed_mask = binary_closing(mask, structure=struct)
    #opened_mask = binary_opening(closed_mask, structure=struct)
    #return opened_mask

def compute_iou(mask1, mask2):
    """
    计算两个 mask 的交并比（IoU）。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def sliding_window(meta, window_size, step_size=None):
    """
    Sliding window for the input data
    """
    if step_size is None:
        step_size = (window_size[0], window_size[1])
    len_time, len_station = meta['data'].shape[2], meta['data'].shape[3]
    assert len_time // window_size[0] >= 2 and len_station // window_size[1] >= 2, f"we recommend that window_size should be smaller than the half of the data size to avoid too much overlap, but now {len_time} {len_station} {window_size[0]} {window_size[1]}"
    
    meta_slice = lambda meta, i, j, status: {
                'data': meta['data'][:, :, i:i+window_size[0], j:j+window_size[1]],
                'amplitude': meta['amplitude'][:, :, i:i+window_size[0], j:j+window_size[1]],
                'station_id': meta['station_id'][j:j+window_size[1]],
                'station_location': meta['station_location'][:, j:j+window_size[1], :],
                'file_name': meta['file_name'],
                'begin_time': [datetime.strptime(meta['begin_time'][0], "%Y-%m-%dT%H:%M:%S.%f")+timedelta(seconds=i * meta['dt_s'][0].item())],
                'begin_time_index': i if 'begin_time_index' not in meta else meta['begin_time_index']+i,
                'dt_s': meta['dt_s'],
                'patch_status': (status, i, j),
            }
    
    for i in range(0, len_time, step_size[0]):
        if i>0 and (len_time-i) < 256:
            break # no too short waveform
        
        for j in range(0, len_station, step_size[1]):
            yield meta_slice(meta, i, j, 'major')
            if j+window_size[1] >= len_station:
                break
        if i+window_size[0] >= len_time:
            break
    
    for i in range(window_size[0]//2, len_time - window_size[0]//2, step_size[0]):
        if len_time-window_size[0]<256 or window_size[1] >= len_station:
            break
        
        if i>0 and (len_time - window_size[0]//2-i) < 256:
            break # no too short waveform
        for j in range(window_size[1]//2, len_station - window_size[1]//2, step_size[1]):
            if j>0 and (len_station - window_size[1]//2-j) < 0:
                break
            yield meta_slice(meta, i, j, 'minor')
            if j+window_size[1] >= len_station - window_size[1]//2:
                break
        if i+window_size[0] >= len_time - window_size[0]//2:
            break
        
    for i in range(window_size[0]//2, len_time - window_size[0]//2, step_size[0]):
        if len_time-window_size[0]<256:
            break
        
        if i>0 and (len_time - window_size[0]//2-i) < 256:
            break # no too short waveform
        for j in [0, len_station - window_size[1]]:
            yield meta_slice(meta, i, j, 'minor')
            if j+window_size[1] >= len_station:
                break
        if i+window_size[0] >= len_time - window_size[0]//2:
            break
        
    tt = [0, len_time - window_size[0]] if len_time%window_size[0] > 256 else [0, max(0, window_size[0]*(len_time//window_size[0]-1))]
    for i in list(set(tt)):
        if window_size[1] >= len_station:
            break
        
        if i>0 and (len_time-i) < 256:
            break # no too short waveform
        for j in range(window_size[1]//2, len_station - window_size[1]//2, step_size[1]):
            yield meta_slice(meta, i, j, 'minor')
            if j+window_size[1] >= len_station - window_size[1]//2:
                break
        if i+window_size[0] >= len_time:
            break
        
        
def sliding_window_indices(total_size, window_size, step_size=None, edge=True):
    """
    Sliding window for the input data
    """
    if step_size is None:
        step_size = (window_size[0], window_size[1])
    len_time, len_station = total_size[0], total_size[1]
    
    for i in range(0, len_time, step_size[0]):
        if i>0 and (len_time-i) < 3:
            break # no too short waveform
        for j in range(0, len_station, step_size[1]):
            yield (i, i+window_size[0], j, j+window_size[1])
            if j+window_size[1] >= len_station:
                break
        if i+window_size[0] >= len_time:
            break
    
    for i in range(window_size[0]//2, len_time - window_size[0]//2, step_size[0]):
        if len_time-window_size[0]<3 or window_size[1] >= len_station:
            break
        
        if i>0 and (len_time - window_size[0]//2-i) < 3:
            break # no too short waveform
        for j in range(window_size[1]//2, len_station - window_size[1]//2, step_size[1]):
            if j>0 and (len_station - window_size[1]//2-j) < 0:
                break
            yield (i, i+window_size[0], j, j+window_size[1])
            if j+window_size[1] >= len_station - window_size[1]//2:
                break
        if i+window_size[0] >= len_time - window_size[0]//2:
            break
    if not edge:
        return
        
    for i in range(window_size[0]//2, len_time - window_size[0]//2, step_size[0]):
        if len_time-window_size[0]<3:
            break
        
        if i>0 and (len_time - window_size[0]//2-i) < 3:
            break # no too short waveform
        for j in [0, len_station - window_size[1]]:
            yield (i, i+window_size[0], j, j+window_size[1])
            if j+window_size[1] >= len_station:
                break
        if i+window_size[0] >= len_time - window_size[0]//2:
            break
        
    tt = [0, len_time - window_size[0]] if len_time%window_size[0] > 256 else [0, max(0, window_size[0]*(len_time//window_size[0]-1))]
    for i in list(set(tt)):
        if window_size[1] >= len_station:
            break
        
        if i>0 and (len_time-i) < 256:
            break # no too short waveform
        for j in range(window_size[1]//2, len_station - window_size[1]//2, step_size[1]):
            yield (i, i+window_size[0], j, j+window_size[1])
            if j+window_size[1] >= len_station - window_size[1]//2:
                break
        if i+window_size[0] >= len_time:
            break

def postprocess(meta, output, polarity_scale=4, event_scale=16):
    assert torch.all(meta["nt"]==meta["nt"][0]), "nt should be the same for all data"
    assert torch.all(meta["nx"]==meta["nx"][0]), "nx should be the same for all data"
    nt, nx = meta["nt"][0], meta["nx"][0]
    data = meta["data"][:, :, :nt, :nx]
    # data = moving_normalize(data)
    meta["data"] = data
    if "phase" in output:
        output["phase"] = output["phase"][:, :, :nt, :nx]
    if "polarity" in output:
        output["polarity"] = output["polarity"][:, :, : (nt - 1) // polarity_scale + 1, :nx]
    if "event_center" in output:
        output["event_center"] = output["event_center"][:, :, : (nt - 1) // event_scale + 1, :nx]
    if "event_time" in output:
        output["event_time"] = output["event_time"][:, :, : (nt - 1) // event_scale + 1, :nx]
    return meta, output



def pred_phasenet(args, model, data_loader, pick_path, figure_path):
    model.eval()
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    with torch.inference_mode():
        for meta in tqdm(data_loader, desc="Predicting", total=len(data_loader)):
            with ctx:
                output = model(meta)
                meta, output = postprocess(meta, output)
            if "phase" in output:
                phase_scores = torch.softmax(output["phase"], dim=1)  # [batch, nch, nt, nsta]
                topk_phase_scores, topk_phase_inds = detect_peaks(phase_scores, vmin=args.min_prob, kernel=128)
                phase_picks_ = extract_picks(
                    topk_phase_inds,
                    topk_phase_scores,
                    file_name=meta["file_name"],
                    station_id=meta["station_id"],
                    begin_time=meta["begin_time"] if "begin_time" in meta else None,
                    begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                    vmin=args.min_prob,
                    phases=args.phases,
                    waveform=meta["data"],
                    window_amp=[10, 5],  # s
                )

            for i in range(len(meta["file_name"])):
                tmp = meta["file_name"][i].split("/")
                parent_dir = "/".join(tmp[-args.folder_depth : -1])
                filename = tmp[-1].replace("*", "").replace("?", "").replace(".mseed", "")

                if not os.path.exists(os.path.join(pick_path, parent_dir)):
                    os.makedirs(os.path.join(pick_path, parent_dir), exist_ok=True)
                if len(phase_picks_[i]) == 0:
                    ## keep an empty file for the file with no picks to make it easier to track processed files
                    with open(os.path.join(pick_path, parent_dir, filename + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(phase_picks_[i])
                picks_df.sort_values(by=["phase_time"], inplace=True)
                picks_df.to_csv(os.path.join(pick_path, parent_dir, filename + ".csv"), index=False)

            if args.plot_figure:
                # meta["waveform_raw"] = meta["waveform"].clone()
                # meta["data"] = moving_normalize(meta["data"])
                plot_phasenet(
                    meta,
                    phase_scores.cpu(),
                    file_name=meta["file_name"],
                    dt=meta["dt_s"] if "dt_s" in meta else torch.tensor(0.01),
                    figure_dir=figure_path,
                )

    ## merge picks
    if args.distributed:
        torch.distributed.barrier()
        if utils.is_main_process():
            merge_picks(pick_path)
    else:
        merge_picks(pick_path)
    return 0


def pred_phasenet_plus(args, model, data_loader, pick_path, event_path, figure_path):
    model.eval()
    ctx = (
        nullcontext()
        if args.device in ["cpu", "mps"]
        else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    )
    with torch.inference_mode():
        for meta in tqdm(data_loader, desc="Predicting", total=len(data_loader)):
            with ctx:
                output = model(meta)
                meta, output = postprocess(meta, output)

            dt = meta["dt_s"] if "dt_s" in meta else [torch.tensor(0.01)] * len(meta["data"])

            if "phase" in output:
                phase_scores = torch.softmax(output["phase"], dim=1)  # [batch, nch, nt, nsta]
                if ("polarity" in output) and (output["polarity"] is not None):
                    polarity_scores = (torch.sigmoid(output["polarity"]) - 0.5) * 2.0
                else:
                    polarity_scores = None
                topk_phase_scores, topk_phase_inds = detect_peaks(
                    phase_scores, vmin=args.min_prob, kernel=128, dt=dt.min().item()
                )
                phase_picks_ = extract_picks(
                    topk_phase_inds,
                    topk_phase_scores,
                    file_name=meta["file_name"],
                    station_id=meta["station_id"],
                    begin_time=meta["begin_time"] if "begin_time" in meta else None,
                    begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    dt=dt,
                    vmin=args.min_prob,
                    phases=args.phases,
                    polarity_score=polarity_scores,
                    waveform=meta["data"],
                    window_amp=[10, 5],  # s
                )

            if ("event_center" in output) and (output["event_center"] is not None):
                event_center = torch.sigmoid(output["event_center"])
                event_time = output["event_time"]
                topk_event_scores, topk_event_inds = detect_peaks(
                    event_center, vmin=args.min_prob, kernel=16, dt=dt.min().item() * 16.0
                )
                event_detects = extract_events(
                    topk_event_inds,
                    topk_event_scores,
                    file_name=meta["file_name"],
                    station_id=meta["station_id"],
                    begin_time=meta["begin_time"] if "begin_time" in meta else None,
                    begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    dt=dt,
                    vmin=args.min_prob,
                    event_time=event_time,
                )

            for i in range(len(meta["file_name"])):
                tmp = meta["file_name"][i].split("/")
                parent_dir = "/".join(tmp[-args.folder_depth : -1])
                filename = tmp[-1].replace("*", "").replace("?", "").replace(".mseed", "")

                if not os.path.exists(os.path.join(pick_path, parent_dir)):
                    os.makedirs(os.path.join(pick_path, parent_dir), exist_ok=True)
                if len(phase_picks_[i]) == 0:
                    ## keep an empty file for the file with no picks to make it easier to track processed files
                    with open(os.path.join(pick_path, parent_dir, filename + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(phase_picks_[i])
                picks_df.sort_values(by=["phase_time"], inplace=True)
                picks_df.to_csv(os.path.join(pick_path, parent_dir, filename + ".csv"), index=False)

                if ("event_center" in output) and ("event_time" in output):
                    if not os.path.exists(os.path.join(event_path, parent_dir)):
                        os.makedirs(os.path.join(event_path, parent_dir), exist_ok=True)
                    if len(event_detects[i]) == 0:
                        with open(os.path.join(event_path, parent_dir, filename + ".csv"), "a"):
                            pass
                        continue
                    events_df = pd.DataFrame(event_detects[i])
                    events_df.sort_values(by=["event_time"], inplace=True)
                    events_df.to_csv(os.path.join(event_path, parent_dir, filename + ".csv"), index=False)

            if args.plot_figure:
                plot_phasenet_plus(
                    meta,
                    phase_scores.cpu(),
                    polarity_scores.cpu() if polarity_scores is not None else None,
                    event_center.cpu() if "event_center" in output else None,
                    event_time.cpu() if "event_time" in output else None,
                    file_name=meta["file_name"],
                    dt=dt,
                    figure_dir=figure_path,
                )

    ## merge picks
    if args.distributed:
        torch.distributed.barrier()
        if utils.is_main_process():
            merge_picks(pick_path)
            merge_events(event_path)
    else:
        merge_picks(pick_path)
        merge_events(event_path)
    return 0


def pred_phasenet_das(args, model, data_loader, pick_path, figure_path):
    model.eval()
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    with torch.inference_mode():
        # for meta in metric_logger.log_every(data_loader, 1, header):
        for meta in tqdm(data_loader, desc="Predicting", total=len(data_loader)):
            with ctx:
                output = model(meta)

            meta, output = postprocess(meta, output)
            scores = torch.softmax(output["phase"], dim=1)  # [batch, nch, nt, nsta]
            topk_scores, topk_inds = detect_peaks(scores, vmin=args.min_prob, kernel=21)

            picks_ = extract_picks(
                topk_inds,
                topk_scores,
                file_name=meta["file_name"],
                begin_time=meta["begin_time"] if "begin_time" in meta else None,
                begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                begin_channel_index=meta["begin_channel_index"] if "begin_channel_index" in meta else None,
                dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                vmin=args.min_prob,
                phases=args.phases,
            )

            for i in range(len(meta["file_name"])):
                tmp = meta["file_name"][i].split("/")
                parent_dir = "/".join(tmp[-args.folder_depth : -1])
                filename = tmp[-1].replace("*", "").replace(f".{args.format}", "")
                if not os.path.exists(os.path.join(pick_path, parent_dir)):
                    os.makedirs(os.path.join(pick_path, parent_dir), exist_ok=True)

                if len(picks_[i]) == 0:
                    ## keep an empty file for the file with no picks to make it easier to track processed files
                    with open(os.path.join(pick_path, parent_dir, filename + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(picks_[i])
                picks_df["channel_index"] = picks_df["station_id"].apply(lambda x: int(x))
                picks_df.sort_values(by=["channel_index", "phase_index"], inplace=True)
                picks_df.to_csv(
                    os.path.join(pick_path, parent_dir, filename + ".csv"),
                    columns=["channel_index", "phase_index", "phase_time", "phase_score", "phase_type"],
                    index=False,
                )

            if args.plot_figure:
                plot_das(
                    meta["data"].cpu().float(),
                    scores.cpu().float(),
                    picks=picks_,
                    phases=args.phases,
                    file_name=meta["file_name"],
                    begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    begin_channel_index=meta["begin_channel_index"] if "begin_channel_index" in meta else None,
                    dt=meta["dt_s"] if "dt_s" in meta else torch.tensor(0.01),
                    dx=meta["dx_m"] if "dx_m" in meta else torch.tensor(10.0),
                    figure_dir=figure_path,
                )

    if args.distributed:
        torch.distributed.barrier()
        if args.cut_patch and utils.is_main_process():
            merge_patch(pick_path, pick_path.rstrip("_patch"), return_single_file=False)
    else:
        if args.cut_patch:
            merge_patch(pick_path, pick_path.rstrip("_patch"), return_single_file=False)

    return 0


def pred_eqnet(args, model, data_loader, pick_path, figure_path, event_path=None):
    feature_scale = 16
    downsample_rate = 256
    encoder_window_size = (args.nt, args.nx)
    window_size = (4096//256, 48)
    mask_thld = 0.4
    iou_thld = 0.40
    MIN_STATIONS = 3
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Predicting:"
    assert encoder_window_size[0] % feature_scale == 0, f"encoder_window_size[0] {args.nt} should be divisible by feature_scale"
    
    try:
        rank = args.rank
    except:
        rank = 0
    assert not (args.cut_patch and args.sliding_window), "cut_patch and sliding_window cannot be used at the same time"
    # ctx = nullcontext() if args.device == "cpu" else torch.cuda.amp.autocast(enabled=args.amp)
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=args.device, dtype=args.ptdtype)
    for isample, meta in enumerate(metric_logger.log_every(data_loader, 1, header)):
        num_stations = len(meta["station_id"])
        if not args.sliding_window:
            try:
                with torch.inference_mode(), ctx:
                    output = model(meta)
            except torch.OutOfMemoryError:
                print(f"Out of memory during processing {meta['file_name']}")
                try:
                    del meta, output
                except:
                    pass
                if args.device != "cpu" and args.device != "mps":
                    torch.cuda.empty_cache()
                continue
        else:
            output_patches = []
            for patch in sliding_window(meta, encoder_window_size, encoder_window_size):
                with torch.inference_mode(), ctx:
                    patch_output = model(patch)
                patch_output["patch_status"] = patch["patch_status"]
                for k in patch_output.keys():
                    try:
                        patch_output[k] = patch_output[k].cpu()
                    except AttributeError:
                        if k == "feature":
                            for f in patch_output[k].keys():
                                patch_output[k][f] = patch_output[k][f].cpu()
                        else:
                            pass
                output_patches.append(patch_output)
            phase = torch.zeros_like(meta["data"]).to(output_patches[0]["phase"].device)
            polarity = torch.zeros_like(meta["data"]).to(phase.device) if "polarity" in output_patches[0] else None
            event = torch.zeros((meta["data"].shape[0], meta["data"].shape[2]//feature_scale, meta["data"].shape[3])).to(phase.device)
            offset = torch.zeros((meta["data"].shape[0], 2, meta["data"].shape[2]//feature_scale, meta["data"].shape[3])).to(phase.device)
            hypocenter = torch.zeros((meta["data"].shape[0], 5, meta["data"].shape[2]//feature_scale, meta["data"].shape[3])).to(phase.device)
            ppm_feat = torch.zeros((output_patches[0]["feature"]['ppm'].shape[0], meta["data"].shape[3], output_patches[0]["feature"]['ppm'].shape[2], meta["data"].shape[2]//downsample_rate)).to(phase.device)
            ppm_dividend = torch.zeros_like(ppm_feat)
            try:
                midlower_feat = torch.zeros((output_patches[0]["feature"]['midlower'].shape[0], meta["data"].shape[3], output_patches[0]["feature"]['midlower'].shape[2], meta["data"].shape[2]//(downsample_rate//4))).to(phase.device)
                midlower_dividend = torch.zeros_like(midlower_feat)
                midupper_feat = torch.zeros((output_patches[0]["feature"]['midupper'].shape[0], meta["data"].shape[3], output_patches[0]["feature"]['midupper'].shape[2], meta["data"].shape[2]//(downsample_rate//16))).to(phase.device)
                midupper_dividend = torch.zeros_like(midupper_feat)
            except KeyError:
                pass

            for patch_output in output_patches:
                status, i, j = patch_output["patch_status"]
                if status=='major':
                    phase[:, :, i:i+encoder_window_size[0], j:j+encoder_window_size[1]] = patch_output["phase"]
                    if polarity is not None:
                        polarity[:, :, i:i+encoder_window_size[0], j:j+encoder_window_size[1]] = patch_output["polarity"]
                    event[:, i//feature_scale:(i+encoder_window_size[0])//feature_scale, j:j+encoder_window_size[1]] = patch_output["event"]
                    offset[:, :, i//feature_scale:(i+encoder_window_size[0])//feature_scale, j:j+encoder_window_size[1]] = patch_output["offset"]
                    hypocenter[:, :, i//feature_scale:(i+encoder_window_size[0])//feature_scale, j:j+encoder_window_size[1]] = patch_output["hypocenter"]
                ppm_t_idx = i//downsample_rate
                ppm_feat[:, j:j+encoder_window_size[1], :, ppm_t_idx:ppm_t_idx+(encoder_window_size[0]//downsample_rate)] += patch_output["feature"]['ppm'][..., :encoder_window_size[0]//downsample_rate]
                ppm_dividend[:, j:j+encoder_window_size[1], :, ppm_t_idx:ppm_t_idx+(encoder_window_size[0]//downsample_rate)] += 1
                try:
                    midlower_t_idx = i//(downsample_rate//4)
                    midlower_feat[:, j:j+encoder_window_size[1], :, midlower_t_idx:midlower_t_idx+(encoder_window_size[0]//(downsample_rate//4))] += patch_output["feature"]['midlower'][..., :encoder_window_size[0]//(downsample_rate//4)]
                    midlower_dividend[:, j:j+encoder_window_size[1], :, midlower_t_idx:midlower_t_idx+(encoder_window_size[0]//(downsample_rate//4))] += 1
                    midupper_t_idx = i//(downsample_rate//16)
                    midupper_feat[:, j:j+encoder_window_size[1], :, midupper_t_idx:midupper_t_idx+(encoder_window_size[0]//(downsample_rate//16))] += patch_output["feature"]['midupper'][..., :encoder_window_size[0]//(downsample_rate//16)]
                    midupper_dividend[:, j:j+encoder_window_size[1], :, midupper_t_idx:midupper_t_idx+(encoder_window_size[0]//(downsample_rate//16))] += 1
                except NameError:
                    pass
            feats = {'ppm': ppm_feat/ppm_dividend}
            try:
                feats['midlower'] = midlower_feat/midlower_dividend
                feats['midupper'] = midupper_feat/midupper_dividend
            except NameError:
                pass
            output = {
                "phase": phase.to(args.device),
                "polarity": polarity.to(args.device),
                "event": event.to(args.device),
                "offset": offset.to(args.device),
                "hypocenter": hypocenter.to(args.device),
                "feature": feats,
            }
            del output_patches, patch_output
        
        if "phase" in output:
            output["phase"] = output["phase"][:, :, :, :num_stations]
        if "event" in output:
            output["event"] = output["event"][:, :, :num_stations]
            output["offset"] = output["offset"][:, :, :, :num_stations]
            output["hypocenter"] = output["hypocenter"][:, :, :, :num_stations]
            
        meta["station_location"] = meta["station_location"][:, :num_stations]
        if args.association:
            association_query = {"feature": {k: output['feature'][k][:, :num_stations].float().cpu() for k in output['feature'].keys()}, "station_location": meta["station_location"],
                                 'station_id': meta["station_id"], 'begin_time': meta["begin_time"], 'dt_s': meta["dt_s"],}
        
        
        dt = meta["dt_s"] if "dt_s" in meta else [torch.tensor(0.01)] * len(meta["data"])
        for k in meta.keys():
            try:
                # if v is a number in fp16, convert it to fp32
                meta[k] = meta[k].float()
            except:
                pass
        for k in output.keys():
            if k=='feature':
                continue
            output[k] = output[k].float()
        
        if "phase" in output:
            phase_scores = F.softmax(output["phase"], dim=1)  # [batch, nch, nt, nsta]
            if ("polarity" in output) and (output["polarity"] is not None):
                if output['polarity'].shape[1] > 1:
                    polarity_scores = torch.softmax(output["polarity"], dim=1)
                else:
                    cut_scale = 20
                    cutpoints = (torch.arange(3-1).float()*cut_scale/(3-2) - cut_scale / 2)[None, :, None, None].detach()
                    sigmoids = torch.sigmoid(cutpoints.to(output["polarity"].device) - output["polarity"])
                    link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
                    link_mat = torch.cat((
                            sigmoids[:, [0]],
                            link_mat,
                            (1 - sigmoids[:, [-1]])
                        ),
                        dim=1
                    )
                    eps = 1e-15
                    polarity_scores = torch.clamp(link_mat, eps, 1 - eps)[:, [1,2,0]] # DNU -> NUD
            else:
                polarity_scores = None
            topk_phase_scores, topk_phase_inds = detect_peaks(phase_scores, vmin=args.min_prob, kernel=128, dt=dt.min().item())
            phase_picks_ = extract_picks(
                topk_phase_inds,
                topk_phase_scores,
                file_name=meta["file_name"],
                station_id=meta["station_id"],
                begin_time=meta["begin_time"] if "begin_time" in meta else None,
                begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                polarity_score=polarity_scores,
                vmin=args.min_prob,
                phases=args.phases,
                waveform=meta["data"],
                window_amp=[10, 5],  # s
            )

        if ("event" in output) and (output["event"] is not None):
            event_scores = torch.sigmoid(output["event"])
            event_scores = event_scores.unsqueeze(1)  # [batch, 1, nt, nsta]
            topk_event_scores, topk_event_inds, event_mask = detect_peaks(event_scores, vmin=args.min_prob, kernel=16, if_mask=True, dt=dt.min().item()*feature_scale)
            event_scores = event_scores.squeeze(1)  # [batch, nt, nsta]
            event_picks_ = extract_events(#)extract_picks(
                topk_event_inds,
                topk_event_scores,
                file_name=meta["file_name"],
                station_id=meta["station_id"],
                begin_time=meta["begin_time"] if "begin_time" in meta else None,
                begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                ## event are picked on downsampled time resolution
                event_scale=feature_scale,
                #dt=meta["dt_s"] * feature_scale if "dt_s" in meta else [0.01 * feature_scale]*meta["data"].shape[0],
                dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                travel_time=output["hypocenter"][:, 0],
                station_location=meta["station_location"],
                relative_location=output["hypocenter"][:, 1:4],
                ps_delta=output["offset"][:, 1],
                magnitude=output["hypocenter"][:, 4],
                waveform=meta["data"],
                vmin=args.min_prob,
                #phases=["event"],
                scores_mask=event_mask,
            )
            
        for i in range(len(meta["file_name"])):
            filename = meta["file_name"][i].split("//")[-1].replace("/", "_")
            events_dfs = {}

            if args.save_score:
                torch.save(phase_scores[i], os.path.join(pick_path.replace('picks', 'scores'), filename + "_phase_score.pt"))
                torch.save(event_scores[i], os.path.join(event_path.replace('events', 'scores'), filename + "_event_score.pt"))
                torch.save(output["offset"][i], os.path.join(event_path.replace('events', 'scores'), filename + "_offset.pt"))
                torch.save(output["hypocenter"][i], os.path.join(event_path.replace('events', 'scores'), filename + "_hypocenter.pt"))
                with open(os.path.join(pick_path.replace('picks', 'scores'), filename + "_station_id.txt"), "w") as f:
                    f.write("\n".join([j[i] for j in meta["station_id"]]))
            if args.save_attn:
                for layer in [(0,0), (0,1), (2,0), (2,1), (4,4), (4,5), (6,0), (6,1)]:
                    torch.save(eval(f'model.backbone.features[{layer[0]}][{layer[1]}].attn').attn[i], os.path.join(pick_path.replace('picks', 'scores'), filename + f"_{layer[0]}_{layer[1]}_attn.pt"))
                    torch.save(eval(f'model.backbone.features[{layer[0]}][{layer[1]}].attn').qk_sim[i], os.path.join(pick_path.replace('picks', 'scores'), filename + f"_{layer[0]}_{layer[1]}_qk_sim.pt"))
                    torch.save(eval(f'model.backbone.features[{layer[0]}][{layer[1]}].attn').rpb[i], os.path.join(pick_path.replace('picks', 'scores'), filename + f"_{layer[0]}_{layer[1]}_rpb.pt"))
                with open(os.path.join(pick_path.replace('picks', 'scores'), filename + "_station_id.txt"), "w") as f:
                    f.write("\n".join([j[i] for j in meta["station_id"]]))
                
            if "phase" in output:
                if len(phase_picks_[i]) == 0:
                    ## keep an empty file for the file with no picks to make it easier to track processed files
                    with open(os.path.join(pick_path, filename + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(phase_picks_[i])
                picks_df.sort_values(by=["phase_time"], inplace=True)
                picks_df.to_csv(os.path.join(pick_path, filename + ".csv"), index=False)
                
            if ("event" in output) and (output["event"] is not None):
                if len(event_picks_[i]) == 0:
                    with open(os.path.join(event_path, filename + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(event_picks_[i])
                picks_df.sort_values(by=["event_time"], inplace=True)
                events_dfs[filename] = picks_df
                picks_df.to_csv(os.path.join(event_path, filename + ".csv"), index=False)

        if args.plot_figure:
            # meta["waveform_raw"] = meta["waveform"].clone()
            # meta["data"] = moving_normalize(meta["data"])
            plot_eqnet(
                meta,
                phase_scores.cpu(),
                event_scores.cpu(),
                phase_picks=phase_picks_,
                event_picks=event_picks_,
                phases=args.phases,
                file_name=meta["file_name"],
                begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                dt=meta["dt_s"] if "dt_s" in meta else torch.tensor(0.01),
                figure_dir=figure_path,
                feature_scale=feature_scale,
            )
            print("saving:", meta["file_name"])
            
        del meta, output, phase_scores, event_scores, polarity_scores
        if args.save_attn:
            for layer in [(0,0), (0,1), (2,0), (2,1), (4,4), (4,5), (6,0), (6,1)]:
                eval(f'model.backbone.features[{layer[0]}][{layer[1]}].attn').attn = []
                eval(f'model.backbone.features[{layer[0]}][{layer[1]}].attn').qk_sim = []
                eval(f'model.backbone.features[{layer[0]}][{layer[1]}].attn').rpb = []
                
        if args.association:
            for i, filename in enumerate(events_dfs.keys()):
                #local_clusters = []
                dt_s = association_query['dt_s'][i].item()
                station_ids = [x[i] for x in association_query['station_id']]
                station_location = association_query['station_location'][i]
                feature = {k: v[i] for k, v in association_query['feature'].items()}
                try:
                    feature['midupper'] = feature['midupper'][..., :feature['ppm'].shape[-1]*16]
                    feature['midlower'] = feature['midlower'][..., :feature['ppm'].shape[-1]*4]
                except KeyError:
                    pass
                begin_time = datetime.strptime(association_query['begin_time'][i], "%Y-%m-%dT%H:%M:%S.%f")
                station_info = pd.DataFrame({'station_id': station_ids, 'sta_lon': station_location[:, 0].numpy(), 'sta_lat': station_location[:, 1].numpy()})
                events_df = events_dfs[filename].merge(station_info, on='station_id', how='left')
                events_df['uid'] = np.array([f'{rank}_{isample}_{i}_{k}' for k in range(len(events_df))])
                events_df['event_id'] = events_df['uid'].apply(lambda x: x)
                if len(events_df) < 2 or feature['ppm'].shape[-1] < 2:
                    events_df.drop(columns=['sta_lon', 'sta_lat'], inplace=True)
                    events_df.to_csv(os.path.join(event_path, filename + ".csv"), index=False)
                    continue
                events_df['event_center_time'] = pd.to_datetime(events_df['event_center_time'], format='%Y-%m-%dT%H:%M:%S.%f')
                events_df['event_time'] = pd.to_datetime(events_df['event_time'], format='%Y-%m-%dT%H:%M:%S.%f')
                events_df['timestamp'] = events_df['event_center_time'].apply(lambda x: (x - begin_time).total_seconds())
                G = nx.Graph() # uid graph
                edges = dict()
                G.add_nodes_from((node, {'weight': score}) for node, score in zip(events_df['uid'].tolist(), events_df['event_score'].tolist()))
                station_step = window_size[1] if len(station_ids) > 2*window_size[1] else max(2, len(station_ids))
                time_step = window_size[0] if feature['ppm'].shape[-1] > 2*window_size[0] else feature['ppm'].shape[-1]
                #for j in range(0, len(station_ids), station_step//2):
                #    if j + station_step//2 > len(station_ids):
                #        continue # the last group is covered by the second last group
                #    for k in range(0, feature.shape[-1], time_step//2):
                #        if k + time_step//2 > feature.shape[-1]:
                #            continue # the last group is covered by the second last group
                for k, _, j, _ in sliding_window_indices(total_size=(feature['ppm'].shape[-1], len(station_ids)), window_size=(time_step, station_step)):
                    ## Prediction
                    t0 = k * downsample_rate*dt_s
                    t1 = t0 + time_step * downsample_rate*dt_s
                    e_df = events_df[(events_df['station_id'].isin(station_ids[j:j+station_step])) & (events_df['timestamp'] >= t0) & (events_df['timestamp'] < t1)]
                    if len(e_df) < 2:
                        continue
                    #if len(station_ids[j:j+station_step]) < MIN_STATIONS:
                    #    continue
                    prompt_index = np.round((e_df['timestamp'].values - t0) / (feature_scale*dt_s), 0)
                    #TEMP
                    #if k==0:
                    #    assert np.allclose(prompt_index, e_df['event_center_index'].values), f"prompt index not match: {prompt_index} {e_df['event_center_index'].values}"
                    prompt = np.stack([prompt_index, e_df['sta_lon'].values, e_df['sta_lat'].values], axis=1)
                    prompt = torch.from_numpy(prompt).unsqueeze(1).float()
                    feat_query = {'ppm': feature['ppm'][j:j+station_step, :, k:k+time_step].unsqueeze(0)}
                    try:
                        feat_query['midupper'] = feature['midupper'][j:j+station_step, :, 16*k:16*(k+time_step)].unsqueeze(0)
                        feat_query['midlower'] = feature['midlower'][j:j+station_step, :, 4*k:4*(k+time_step)].unsqueeze(0)
                    except KeyError:
                        pass
                    query = {'feature': feat_query, 'station_location': station_location[j:j+station_step, :].unsqueeze(0), 'prompt': prompt}
                    with torch.inference_mode(), torch.amp.autocast(device_type=args.device, dtype=args.ptdtype):
                        assocation_mask = model(query, association_only=True)
                    mask = assocation_mask['association'].float()
                    mask = mask.sigmoid().squeeze(1).cpu().numpy() # Nprompt, nt, nsta
                    #np.savez(os.path.join(event_path, filename + f"_{j}_{k}.npz"), mask=mask, iou=assocation_mask['iou'].float().squeeze(1).cpu().numpy(), uid=e_df['uid'].values, station_ids=station_ids[j:j+station_step], begin_time=(begin_time+timedelta(seconds=t0)).isoformat(timespec='microseconds'))
                    mask = mask > mask_thld
                    
                    ## Postprocess
                    uid = e_df['uid'].to_list() #to_numpy()#
                    rt = e_df['event_time'].min()
                    e2ot = e_df['event_time'].apply(lambda x: (x - rt).total_seconds()).to_list() #to_numpy()#
                    e2sta = e_df['station_id'].to_list() #to_numpy()#
                    #e2loc = e_df[['event_location_x', 'event_location_y']].to_numpy()
                    mask = time_morphology_filter(mask, t_kernel_size=5)
                        
                    for m in range(len(uid)):
                        if mask[m].sum() < 3: # skip all zero outputs
                            #assert 0, f"event {e_df[e_df['uid'] == uid[m]][['uid', 'station_id', 'event_score']]} mask {mask[m].sum()}"
                            continue
                        for n in range(m+1, len(uid)):
                            if (mask[n].sum() < 3) or (e2sta[m] == e2sta[n]) or (abs(e2ot[m] - e2ot[n]) >= 3):
                                continue
                            #if np.sqrt(((e2loc[m]-e2loc[n])**2).sum()) > 30:
                            #    continue
                            iou_mn = compute_iou(mask[m], mask[n])
                            ot_diff_weighted_iou = (1 - 0.5*(abs(e2ot[m] - e2ot[n]) - 2)/(3 - 2)) * iou_mn if abs(e2ot[m] - e2ot[n]) > 2 else iou_mn
                            #assert not ((tuple(sorted([uid[m], uid[n]])) in edges)&((uid[m], uid[n]) not in edges)), f'invert order {(uid[m], uid[n])}'
                            old = edges.get((uid[m], uid[n]))
                            edges[(uid[m], uid[n])] = (old + ot_diff_weighted_iou) / 2 if old else ot_diff_weighted_iou
                    del query, assocation_mask
                    #if feature['ppm'].shape[-1] == time_step:
                    #    break
                    #if len(station_ids) == station_step:
                    #    break
                G.add_weighted_edges_from([
                    (m, n, w) for (m, n), w in edges.items() if w >= iou_thld
                ])
                # save G
                nx.write_graphml_lxml(G, os.path.join(event_path, filename + ".graphml"))
                communities = nx.community.louvain_communities(G, weight='weight', resolution=1)
                # update events_df
                events_df.set_index('uid', inplace=True)
                for cluster in communities:
                    if len(cluster) < MIN_STATIONS:
                        continue
                    events_df.loc[list(cluster), 'event_id'] = f"{list(cluster)[0]}"
                events_df.reset_index(drop=False, inplace=True)
                events_df['event_time'] = events_df['event_time'].apply(lambda x: x.isoformat(timespec='microseconds'))
                events_df['event_center_time'] = events_df['event_center_time'].apply(lambda x: x.isoformat(timespec='microseconds'))
                events_df.drop(columns=['sta_lon', 'sta_lat'], inplace=True)
                events_df.to_csv(os.path.join(event_path, filename + ".csv"), index=False)
            del association_query, feat_query
    
        # if cuda, clear cache
        if args.device != "cpu" and args.device != "mps":
            torch.cuda.empty_cache()

    ## merge picks
    if args.distributed:
        torch.distributed.barrier()
        if utils.is_main_process():
            merge_patch(pick_path, pick_path.replace("_patch", ""))
            merge_patch(event_path, event_path.replace("_patch", ""))
    else:
        merge_patch(pick_path, pick_path.replace("_patch", ""))
        merge_patch(event_path, event_path.replace("_patch", ""))
        
    return 0


def main(args):
    result_path = args.result_path
    if args.cut_patch:
        pick_path = os.path.join(result_path, f"picks_{args.model}_patch")
        event_path = os.path.join(result_path, f"events_{args.model}_patch")
        figure_path = os.path.join(result_path, f"figures_{args.model}_patch")
    else:
        pick_path = os.path.join(result_path, f"picks_{args.model}")
        event_path = os.path.join(result_path, f"events_{args.model}")
        figure_path = os.path.join(result_path, f"figures_{args.model}")
    if not os.path.exists(result_path):
        utils.mkdir(result_path)
    if not os.path.exists(pick_path):
        utils.mkdir(pick_path)
    if not os.path.exists(event_path):
        utils.mkdir(event_path)
    if not os.path.exists(figure_path):
        utils.mkdir(figure_path)
    if not os.path.exists(pick_path.replace("picks", "scores")):
        utils.mkdir(pick_path.replace("picks", "scores"))

    utils.init_distributed_mode(args)
    print(args)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
    else:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            rank = 0
            world_size = 1

    device = torch.device(args.device)
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    #TODO: check if bfloat16 decreases the accuracy
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    args.dtype, args.ptdtype = dtype, ptdtype
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    if args.use_deterministic_algorithms:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if args.model in ["phasenet", "phasenet_plus"]:
        dataset = SeismicTraceIterableDataset(
            data_path=args.data_path,
            data_list=args.data_list,
            hdf5_file=args.hdf5_file,
            prefix=args.prefix,
            format=args.format,
            dataset=args.dataset,
            training=False,
            highpass_filter=args.highpass_filter,
            response_path=args.response_path,
            response_xml=args.response_xml,
            cut_patch=args.cut_patch,
            resample_time=args.resample_time,
            system=args.system,
            nx=args.nx,
            nt=args.nt,
            rank=rank,
            world_size=world_size,
        )
        sampler = None
    elif args.model == "phasenet_das":
        dataset = DASIterableDataset(
            data_path=args.data_path,
            data_list=args.data_list,
            format=args.format,
            nx=args.nx,
            nt=args.nt,
            training=False,
            system=args.system,
            cut_patch=args.cut_patch,
            highpass_filter=args.highpass_filter,
            resample_time=args.resample_time,
            resample_space=args.resample_space,
            skip_existing=args.skip_existing,
            pick_path=pick_path,
            folder_depth=args.folder_depth,
            rank=rank,
            world_size=world_size,
        )
        sampler = None
    elif args.model == "eqnet":
        dataset = SeismicNetworkIterableDataset(
        data_path=args.data_path,
        data_list=args.data_list,
        hdf5_file=args.hdf5_file,
        station_file=args.station_file,
        prefix=args.prefix,
        format=args.format,
        dataset=args.dataset,
        training=False,
        phases=["P", "S"],
        sort=args.sort,
        ## for prediction
        sampling_rate=100,
        highpass_filter=args.highpass_filter,
        response_path=args.response_path,
        response_xml=args.response_xml,
        cut_patch=args.cut_patch,
        nx=args.nx,
        nt_cut=args.nt,
        rank=rank,
        world_size=world_size,
        is_pad=args.is_pad,
        num_stations=args.num_stations,
        )
        sampler = None
    else:
        raise ("Unknown model")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=min(args.workers, mp.cpu_count()),
        collate_fn=None,
        drop_last=False,
    )

    model = eqnet.models.__dict__[args.model].build_model(
        backbone=args.backbone,
        in_channels=1,
        out_channels=(len(args.phases) + 1),
        ## eqnet
        head=args.head,
        use_station_location=args.use_station_location,
        prediction=True,
        add_prompt=(not args.no_prompt),
    )
    logger.info("Model:\n{}".format(model))

    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "_orig_mod" in list(checkpoint["model"].keys())[0]:
            for k in list(checkpoint["model"].keys()):
                checkpoint["model"][k.replace("_orig_mod.", "")] = checkpoint["model"].pop(k)
        model.load_state_dict(checkpoint["model"], strict=True)
        print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        #print(f"weight keys: {checkpoint['model'].keys()}")
    else:
        if args.model == "phasenet":
            raise ("No pretrained model for phasenet, please use phasenet_plus instead")
        elif args.model == "phasenet_plus":
            model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-Plus-v1/PhaseNet-Plus-v1.pth"
        elif args.model == "phasenet_das":
            if args.location is None:
                # model_url = "ai4eps/model-registry/PhaseNet-DAS:latest"
                # model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v0/PhaseNet-DAS-v0.pth"
                model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v1/PhaseNet-DAS-v1.pth"
            elif args.location == "forge":
                model_url = (
                    "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-ConvertedPhase/model_99.pth"
                )
            else:
                raise NotImplementedError("Missing pretrained model for this location")
        elif args.model == "eqnet":
            model_url = "ai4eps/model-registry/eqnet-test:latest"
        else:
            raise NotImplementedError("Missing pretrained model for this model")

        if utils.is_main_process():
            ## load model from wandb
            if args.model == "eqnet":
                with wandb.init() as run:
                    artifact = run.use_artifact(model_url, type="model")
                    artifact_dir = artifact.download()
                checkpoint = torch.load(glob(os.path.join(artifact_dir, "*.pth"))[0], map_location="cpu")
                model.load_state_dict(checkpoint["model"], strict=True)
            ## load model from url
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    model_url, model_dir="./", progress=True, check_hash=True, map_location="cpu"
                )
                model.load_state_dict(state_dict["model"], strict=True)

    if args.compile:
        model = torch.compile(model, dynamic=True)

    model_without_ddp = model
    if args.distributed:
        torch.distributed.barrier()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    if args.model == "phasenet":
        pred_phasenet(args, model, data_loader, pick_path, figure_path)

    if args.model == "phasenet_plus":
        pred_phasenet_plus(args, model, data_loader, pick_path, event_path, figure_path)

    if args.model == "phasenet_das":
        pred_phasenet_das(args, model, data_loader, pick_path, figure_path)
    
    if args.model == "eqnet":
        pred_eqnet(args, model, data_loader, pick_path, figure_path, event_path)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="EQNet Model", add_help=add_help)

    # model
    parser.add_argument("--model", default="phasenet_das", type=str, help="model name")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--backbone", default="unet", type=str, help="model backbone")
    parser.add_argument("--head", default="simple", type=str, help="model head")
    parser.add_argument("--use-station-location", action="store_true", help="use station location")
    parser.add_argument("--phases", default=["P", "S"], type=str, nargs="+", help="phases to use")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile model to torchscript",
    )
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    
    # deterministic option
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    
    # Mixed precision training parameters
    parser.add_argument(
        "--use_deterministic_algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--timeout", default=30, type=int, help="timeout for distributed barrier")

    # prediction parameters
    parser.add_argument("--data_path", type=str, default="./", help="path to data directory")
    parser.add_argument("--data_list", type=str, default=None, help="selectecd data list")
    parser.add_argument("--hdf5-file", default=None, type=str, help="hdf5 file for training")
    parser.add_argument("--station_file", default=None, type=str, help="station file")
    parser.add_argument("--prefix", default="", type=str, help="prefix for the file name")
    parser.add_argument("--format", type=str, default="h5", help="data format")
    parser.add_argument("--dataset", type=str, default="das", help="dataset type; seismic_trace, seismic_network, das")
    parser.add_argument("--result_path", type=str, default="results", help="path to result directory")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument("--min_prob", default=0.3, type=float, help="minimum probability for picking")
    parser.add_argument("--association", action="store_true", help="If enable association")
    parser.add_argument("--is_pad", action="store_true", help="If pad the data to at least num_stations for eqnet")
    parser.add_argument("--sort", action="store_true", help="If sort the data by station locations for eqnet")
    parser.add_argument("--num_stations", default=10, type=int, help="number of stations for eqnet")
    parser.add_argument("--save_score", action="store_true", help="If save score for test")
    parser.add_argument("--save_attn", action="store_true", help="If save attention for visualization")

    ## Seismic
    parser.add_argument("--add_polarity", action="store_true", help="If use polarity information")
    parser.add_argument("--add_event", action="store_true", help="If use event information")
    parser.add_argument("--no-prompt", action="store_true", help="If use prompt information")
    parser.add_argument("--highpass_filter", type=float, default=0.0, help="highpass filter; default 0.0 is no filter")
    parser.add_argument("--response_path", default=None, type=str, help="response path")
    parser.add_argument("--response_xml", default=None, type=str, help="response xml file")
    parser.add_argument("--folder_depth", default=0, type=int, help="folder depth for data list")

    ## DAS
    parser.add_argument("--cut_patch", action="store_true", help="If cut patch for continuous data")
    parser.add_argument("--sliding_window", action="store_true", help="If use sliding window for continuous data")
    parser.add_argument("--nt", default=1024 * 20, type=int, help="number of time samples for each patch")
    parser.add_argument("--nx", default=1024 * 5, type=int, help="number of spatial samples for each patch")
    parser.add_argument("--resample_time", action="store_true", help="If resample time for continuous data")
    parser.add_argument("--resample_space", action="store_true", help="If resample space for continuous data")
    parser.add_argument(
        "--system", type=str, default=None, help="The name of system of different system: optasense, eqnet, or None"
    )
    parser.add_argument("--location", type=str, default=None, help="The name of systems at location")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing files")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
