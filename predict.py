import datetime
import logging
import os
import sys
from tkinter import W
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torch.distributed as dist
import multiprocessing

import utils
from eqnet.data import DASDataset, DASIterableDataset
from eqnet.utils import detect_peaks, extract_picks, merge_picks, plot_das
import eqnet

import warnings
warnings.filterwarnings("ignore", ".*Length of IterableDataset.*")

logger = logging.getLogger("EQNet")

def pred_fn(model, data_loader, pick_path, figure_path, args):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Predicting:"
    with torch.inference_mode():
        for meta in metric_logger.log_every(data_loader, 1, header):

            with torch.cuda.amp.autocast(enabled=args.amp):
                scores = torch.softmax(model(meta), dim=1) #batch, nch, nt, nsta
                nt1, ns1 = meta["data"].shape[-2:]
                nt2, ns2 = scores.shape[-2:]
                scores = scores[
                    :,
                    :,
                    abs(nt2 - nt1) // 2 : abs(nt2 - nt1) // 2 + nt1,
                    abs(ns2 - ns1) // 2 : abs(ns2 - ns1) // 2 + ns1,
                ]
                vmin = 0.6
                topk_scores, topk_inds = detect_peaks(scores, vmin=vmin, kernel=21)
            
                picks_ = extract_picks(
                    topk_inds,
                    topk_scores,
                    file_name=meta["file_name"],
                    begin_time=meta["begin_time"] if "begin_time" in meta else None,
                    begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    begin_channel_index=meta["begin_channel_index"] if "begin_channel_index" in meta else None,
                    dt = meta["dt_s"] if "dt_s" in meta else 0.01,
                    dx = meta["dx_m"] if "dx_m" in meta else 0.01,
                    vmin=vmin,
                )

            for i in range(len(meta["file_name"])):
                if len(picks_[i]) == 0:
                    continue
                picks_df = pd.DataFrame(picks_[i])
                picks_df["channel_index"] = picks_df["station_name"].apply(lambda x: int(x))
                picks_df.sort_values(by=["channel_index", "phase_index"], inplace=True)
                picks_df.to_csv(
                    os.path.join(
                        pick_path, meta["file_name"][i] + ".csv"
                    ),
                    columns=["channel_index","phase_index","phase_time","phase_score","phase_type"],
                    index=False,
                )

            if args.plot_figure:
                plot_das(
                    # normalize_local(meta["data"]).cpu().numpy(),
                    meta["data"].cpu().numpy(),
                    scores.cpu().numpy(),
                    picks=picks_,
                    file_name=meta["file_name"],
                    begin_time_index=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    begin_channel_index=meta["begin_channel_index"] if "begin_channel_index" in meta else None,
                    dt = meta["dt_s"] if "dt_s" in meta else torch.tensor(0.01),
                    dx = meta["dx_m"] if "dx_m" in meta else torch.tensor(10.0),
                    figure_dir=figure_path,
                )

    if args.distributed:
        torch.distributed.barrier()
        if utils.is_main_process():
            merge_picks(pick_path)
    else:
        merge_picks(pick_path)


    return 0


def main(args):

    result_path = args.result_path
    if not os.path.exists(result_path):
        utils.mkdir(result_path)
    pick_path = os.path.join(result_path, "picks_phasenet_das")
    if not os.path.exists(pick_path):
        utils.mkdir(pick_path)
    figure_path = os.path.join(result_path, "figures_phasenet_das")
    if not os.path.exists(figure_path):
        utils.mkdir(figure_path)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    model = eqnet.models.__dict__[args.model]()
    logger.info("Model:\n{}".format(model))
    
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    else:
        model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v2/model_99.pth"
        state_dict = torch.hub.load_state_dict_from_url(model_url, model_dir="./", progress=True, check_hash=True, map_location="cpu")
        model_without_ddp.load_state_dict(state_dict["model"], strict=False)

    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
    else:
        rank, world_size = 0, 1
    dataset = DASIterableDataset(
        data_path = args.data_path,
        format = args.format,
        filtering = args.filtering,
        rank = rank, 
        world_size = world_size,
        update_total_number = True,
        training=False)
    sampler = None


    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=min(args.workers, multiprocessing.cpu_count()),
        collate_fn=None,
        drop_last=False,
    )


    pred_fn(model, data_loader, pick_path, figure_path, args)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--model", default="phasenet_das", type=str, help="model name")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-j", "--workers", default=32, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument(
        "--data_path", type=str, default="./", help="path to data directory"
    )    
    parser.add_argument(
        "--result_path", type=str, default=None, help="path to result directory"
    )
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument(
        "--format", type=str, default="h5", help="data format"
    )
    parser.add_argument(
        "--filtering", action="store_true", help="If filter the data"
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")


    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
