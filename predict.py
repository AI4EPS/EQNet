import datetime
import logging
import os
import sys
import time
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

import utils
from eqnet.data import DASDataset, DASIterableDataset
from eqnet.utils import detect_peaks, extract_picks, plot_das
import eqnet

import warnings
warnings.filterwarnings("ignore", ".*Length of IterableDataset.*")

logger = logging.getLogger("EQNet")

def pred_fn(model, data_loader, args):


    if args.result_path is None:
        result_path = args.data_path.rstrip("/").split("/")[-1]
    else:
        result_path = args.result_path
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    picks_path = os.path.join(result_path, "picks_phasenet_das")
    if not os.path.exists(picks_path):
        os.mkdir(picks_path)
    if args.plot_figure:
        figures_path = os.path.join(result_path, "figures_phasenet_das")
        if not os.path.exists(figures_path):
            os.mkdir(figures_path)

    model.eval()
    picks = []
    with torch.inference_mode():
        pbar = tqdm(total=len(data_loader))
        processed = set()
        for meta in data_loader:
            for file_name in meta["file_name"]:
                if file_name.split("_")[0] not in processed:
                    processed.add(file_name.split("_")[0])
                    pbar.update(1)
            pbar.set_description(f'{",".join(meta["file_name"])}')


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
                    file_names=meta["file_name"],
                    begin_times=meta["begin_time"] if "begin_time" in meta else None,
                    begin_time_indexs=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    begin_channel_indexs=meta["begin_channel_index"] if "begin_channel_index" in meta else None,
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
                        picks_path, meta["file_name"][i] + ".csv"
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
                    file_names=meta["file_name"],
                    begin_time_indexs=meta["begin_time_index"] if "begin_time_index" in meta else None,
                    begin_channel_indexs=meta["begin_channel_index"] if "begin_channel_index" in meta else None,
                    dt = meta["dt_s"] if "dt_s" in meta else torch.tensor(0.01),
                    dx = meta["dx_m"] if "dx_m" in meta else torch.tensor(10.0),
                    figure_dir=figures_path,
                )

            for x in picks_:
                picks.extend(x)
        pbar.close()


    picks_df = pd.DataFrame.from_records(picks)
    picks_df["channel_index"] = picks_df["station_name"].apply(lambda x: int(x))
    picks_df.sort_values(by=["channel_index", "phase_index"], inplace=True)
    picks_df.to_csv(os.path.join(result_path, f"picks_phasenet_das.csv"), index=False)
    
    return 0


def main(args):

    utils.init_distributed_mode(args)
    print(args)

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

    dataset = DASIterableDataset(
        data_path = args.data_path,
        format = args.format,
        filtering = args.filtering,
        training=False)
    sampler = None


    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=16,
        collate_fn=None,
        drop_last=False,
    )


    pred_fn(model, data_loader, args)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--model", default="phasenet_das", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-j", "--workers", default=16*2, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )

    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")


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
