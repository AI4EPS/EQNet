import os
from datetime import datetime, timedelta
from glob import glob

import h5py
import numpy as np
import pandas as pd
import random
import scipy
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import matplotlib.pyplot as plt


def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x


def normalize(data):
    """
    data: [Nch, Nt, Nsta] (Nchn: number of channels, Nt: number of time, Nsta: number of stations)
    """
    nch, nt, nsta = data.shape
    data = data.double()
    mean = torch.mean(data, dim=(1), keepdims=True)
    std = torch.std(data, dim=(1), keepdims=True)
    std = std.repeat((1, nt, 1))
    data = data - mean
    mask = (std!=0)
    data[mask] = data[mask] / std[mask]
    return data.float()


def normalize_local_1d(data, window=1024 + 1):
    nch, nt, nsta = data.shape
    data = data.permute((2, 0, 1))
    with torch.no_grad():
        data_ = F.pad(data, (window // 2, window // 2), mode="circular")
        mean = F.avg_pool1d(data_, kernel_size=window, stride=1)
        data -= mean
        data_ = F.pad(data, (window // 2, window // 2), mode="circular")
        std = F.lp_pool1d(data_, norm_type=2, kernel_size=window, stride=1) / (window**0.5)
        data /= std
        data = log_transform(data)
    data = data.permute((1, 2, 0))
    return data


# def normalize_local(data, window=1024 + 1):
#     nch, nt, nsta = data.shape
#     data = data.permute((2, 0, 1))
#     data = data.unsqueeze(0)  # batch, nsta, nch, nt
#     with torch.no_grad():
#         data_ = F.pad(data, (window // 2, window // 2, 0, 0), mode="circular")
#         mean = F.avg_pool2d(data_, kernel_size=(nch, window), stride=(1, 1))
#         data -= mean
#         data_ = F.pad(data, (window // 2, window // 2, 0, 0), mode="circular")
#         std = F.lp_pool2d(data_, norm_type=2, kernel_size=(nch, window), stride=(1, 1)) / ((window * nch) ** 0.5)
#         data /= std
#         data = log_transform(data)
#     data = data.squeeze(0)
#     data = data.permute((1, 2, 0))
#     return data

def generate_label(data, phase_list, label_width=[150, 150], label_shape="gaussian", sta_mask=None, return_phase_mask=False):
    """_summary_

    Args:
        data (tensor): (nch, nt, nsta)
        phase_list (_type_): _description_
        label_width (list, optional): label_width=[150, 200]. Defaults to [150, 200].
        label_shape (str, optional): _description_. Defaults to "gaussian".
        mask (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    nch, nt, nsta = data.shape
    target = np.zeros([len(phase_list) + 1, nt, nsta], dtype=np.float32)
    phase_mask = np.zeros([nt, nsta], dtype=np.float32) ## mask for window near the phase arrival

    if sta_mask is None:
        sta_mask = np.zeros((len(phase_list), nsta), dtype=np.bool)

    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for trace, phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.1] = 0.0
            target[i + 1, :, trace] += gaussian
            sta_mask[i, trace] = True
            phase_mask[int(phase_time) - w: int(phase_time) + w, trace] = 1

    sta_mask = np.all(sta_mask, axis=0)  ## traces with all picks

    target[0:1, :, sta_mask] = np.maximum(0, 1 - np.sum(target[1:, :, sta_mask], axis=0, keepdims=True))
    target[:, :, ~sta_mask] = 0

    # plt.figure(figsize=(12, 12))
    # plt.subplot(3, 1, 1)
    # plt.imshow(data[0,:,:], aspect="auto")
    # plt.subplot(3, 1, 2)
    # plt.imshow(target.transpose(1, 2, 0), aspect="auto")
    # plt.subplot(3, 1, 3)
    # plt.imshow(phase_mask, aspect="auto")
    # plt.savefig("test_label.png")
    # raise  

    if return_phase_mask:
        return target, phase_mask
    else:
        return target


def cut_data(data, targets=None, nt=1024*3, nsta=1024*5):
    nch, w, h = data.shape # w: time, h: station
    w0 = np.random.randint(0, max(1, w - nt))
    h0 = np.random.randint(0, max(1, h - nsta))
    w1 = np.random.randint(0, max(1, nt - w))
    h1 = np.random.randint(0, max(1, nsta - h))

    if targets is not None:
        label_width = 150
        max_tries = 100
        max_w0 = 0
        max_h0 = 0
        max_sum = 0
        tmp_sum = 0
        tries = 0
        while tmp_sum < label_width / 2 * nsta * 0.1:
            w0 = np.random.randint(0, max(1, w - nt))
            h0 = np.random.randint(0, max(1, h - nsta))
            tmp_sum = torch.sum(targets[1:, w0 : w0 + nt, h0 : h0 + nsta])  # nch, nt, nsta
            if tmp_sum > max_sum:
                max_sum = tmp_sum
                max_w0 = w0
                max_h0 = h0
            tries += 1
            if tries >= max_tries:
                w0 = max_w0
                h0 = max_h0
                break
        if tries >= max_tries:
            print(f"cut data failed, tries={tries}")
            return None, None

    data_ = torch.zeros((nch, nt, nsta), dtype=data.dtype, device=data.device)
    tmp = data[:, w0 : w0 + nt, h0 : h0 + nsta]
    data_[:, w1:w1+tmp.shape[-2], h1:h1+tmp.shape[-1]] = tmp[:, :, :]
    if targets is not None:
        targets_ = torch.zeros((targets.shape[0], nt, nsta), dtype=targets.dtype, device=targets.device)
        tmp =  targets[..., w0 : w0 + nt, h0 : h0 + nsta]
        targets_[:, w1:w1+tmp.shape[-2], h1:h1+tmp.shape[-1]] = tmp[:, :, :]
        mask = torch.sum(targets_[1:, :, :], axis=(0, 1))  ## no P/S channels
        targets_[0, :, mask == 0] = 0
        return data_, targets_
    else:
        return data_


def cut_noise(noise, nt=1024*3, nsta=1024*5):
    nch, w, h = noise.shape
    w0 = np.random.randint(0, max(1, w - nt))
    h0 = np.random.randint(0, max(1, h - nsta))
    return noise[:, w0 : w0 + nt, h0 : h0 + nsta]


def pad_noise(noise, nt=1024*3, nsta=1024*5):
    nch, w, h = noise.shape
    if w < nt:
        with torch.no_grad():
            noise = noise.unsqueeze(0)
            if (nt//w - 1) > 0:
                for i in range(nt//w - 1):
                    noise = F.pad(noise, (0, 0, 0, w-1), mode="reflect")
                noise = F.pad(noise, (0, 0, 0, nt//w - 1), mode="reflect")
            noise = F.pad(noise, (0, 0, 0, nt % w), mode="reflect").squeeze(0)
    if h < nsta:
        with torch.no_grad():
            noise = noise.unsqueeze(0)
            if (nsta//h - 1) > 0:
                for i in range(nsta//h - 1):
                    noise = F.pad(noise, (0, h-1, 0, 0), mode="reflect")
                noise = F.pad(noise, (0, nsta//h - 1, 0, 0), mode="reflect")
            noise = F.pad(noise, (0, nsta % h, 0, 0), mode="reflect").squeeze(0)
    return noise

def pad_signal(signal, target, mask, nt=1024*3, nsta=1024*5):
    nch, w, h = signal.shape
    if h < nsta:
        with torch.no_grad():
            signal = signal.unsqueeze(0)
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)
            if (nsta//h - 1) > 0:
                for i in range(nsta//h - 1):
                    signal = F.pad(signal, (0, h-1, 0, 0), mode="reflect")
                    target = F.pad(target, (0, h-1, 0, 0), mode="reflect")
                    mask = F.pad(mask, (0, h-1, 0, 0), mode="reflect")
                signal = F.pad(signal, (0, nsta//h - 1, 0, 0), mode="reflect")
                target = F.pad(target, (0, nsta//h - 1, 0, 0), mode="reflect")
                mask = F.pad(mask, (0, nsta//h - 1, 0, 0), mode="reflect")
            signal = F.pad(signal, (0, nsta % h, 0, 0), mode="reflect").squeeze(0)
            target = F.pad(target, (0, nsta % h, 0, 0), mode="reflect").squeeze(0)
            mask = F.pad(mask, (0, nsta % h, 0, 0), mode="reflect").squeeze(0)
    return signal, target, mask


def flip_lr(data, targets=None):
    data = data.flip(-1)
    if targets is not None:
        targets = targets.flip(-1)
        return data, targets
    else:
        return data


def roll_by_gather(data, dim, shifts: torch.LongTensor):

    nch, h, w = data.shape

    if dim == 0:
        arange1 = torch.arange(h).view((1, h, 1)).repeat((nch, 1, w))
        arange2 = (arange1 - shifts) % h
        return torch.gather(data, 1, arange2)
    elif dim == 1:
        arange1 = torch.arange(w).view((1, 1, w)).repeat((nch, h, 1))
        arange2 = (arange1 - shifts.unsqueeze(0)) % w
        return torch.gather(data, 2, arange2)
    else:
        raise ValueError("dim must be 0 or 1")


def filter_labels(label_list):
    label_selected = []
    print(f"{len(label_list) = }")
    for label in label_list:
        if os.path.getsize(label) > 50e3:  # bytes
            label_selected.append(label)
    print(f"{len(label_selected) = }")
    return label_selected


def load_segy(infile, nTrace=1250):
    filesize = os.path.getsize(infile)
    nSample = int(((filesize - 3600) / nTrace - 240) / 4)
    data = np.zeros((nTrace, nSample), dtype=np.float32)
    fid = open(infile, "rb")
    fid.seek(3600)
    for i in range(nTrace):
        fid.seek(240, 1)
        data[i, :] = np.fromfile(fid, dtype=np.float32, count=nSample)
    return data


def add_moveout(data, targets=None, vmin=2.0, vmax=6.0, dt=0.01, dx=0.01, shift_range=1000):
    """_summary_

    Args:
        data (_type_): _description_
        targets (_type_, optional): _description_. Defaults to None.
        vmin (float, optional): _description_. Defaults to 2.0.
        vmax (float, optional): _description_. Defaults to 8.0.
        dt (float, optional): time sample spacing. Defaults to 0.01 s.
        dx (float, optional): channel spacing. Defaults to 0.01 km.
        shift_range (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    ###
    nch, h, w = data.shape
    iw = torch.randint(low=0, high=w, size=(1,))
    shift = (
        (torch.arange(w) - iw).abs()
        * dx
        / (
            vmin
            + torch.rand(
                1,
            )
            * (vmax - vmin)
        )
        / dt
    ).int()
    # shift = ((torch.arange(w) - iw).abs() /w * shift_range * torch.rand((1,))).int()
    data = roll_by_gather(data, dim=0, shifts=shift)
    if targets is not None:
        targets = roll_by_gather(targets, dim=0, shifts=shift)
        return data, targets
    else:
        return data


def calc_snr(data, picks, noise_window=200, signal_window=200):

    SNR = []
    for trace, phase_time in picks:
        phase_time = int(phase_time)
        noise = torch.std(data[:, max(0, phase_time - noise_window) : phase_time, trace])
        signal = torch.std(data[:, phase_time : phase_time + signal_window, trace])
        SNR.append(signal / noise)
    return np.median(SNR)


def stack_noise(data, noise, snr):
    ## approximately after normalization, noise=1, signal=snr, so signal ~ noise * snr
    return data + noise * max(0, snr - 2) * torch.rand(1)
    # return data + noise * max(0, snr - 1.0) * torch.rand(1)
    # return data + noise * max(0, snr - 1.0) * torch.rand(1) * 1.5
    # return data + noise * max(0, snr - 1.0) * torch.rand(1) #* 2.0
    # return data + noise * snr * 0.5


def mask_data(data, target, nt=256, nsta=256):
    # nch, nt, nsta = data.shape
    nt_ = random.randint(1, nt)
    nsta_ = random.randint(1, nsta)
    data_ = torch.clone(data)
    target_ = torch.clone(target)
    if np.random.rand() > 0.5:
        data_[:, :nt_, :] = 0.0
        target_[0, :nt_, :] = 1.0
        target_[1:, :nt_, :] = 0.0
        data_[:, :, :nsta_] = 0.0
        target_[0, :, :nsta_] = 1.0
        target_[1:, :, :nsta_] = 0.0
    else:
        data_[:, -nt_:, :] = 0.0
        target_[0:1, -nt_:, :] = 1.0
        target_[1:, -nt_:, :] = 0.0
        data_[:, :, -nsta_:] = 0.0
        target_[0:1, :, -nsta_:] = 1.0
        target_[1:, :, -nsta_:] = 0.0
    return data_, target_

def resample_space(data, target, factor=1):
    nch, nt, nsta = data.shape
    # if axis == "time":
    #     scale_factor = (random.uniform(min(1, factor), max(1, factor)), 1)
    # elif axis == "space":
    #     scale_factor = (1, random.uniform(min(1, factor), max(1, factor)))
    # else:
    #     raise ValueError("axis must be time or space")
    scale_factor = random.uniform(min(1, factor), max(1, factor))
    with torch.no_grad():
        data_ = F.interpolate(data, scale_factor=scale_factor, mode="nearest")
        target_ = F.interpolate(target, scale_factor=scale_factor, mode="nearest")
    return data_, target_

def resample_time(data, picks, factor=1):
    """_summary_

    Args:
        data (_type_): _description_
        picks (_type_): [[p_channel, p_index],[s_channel, s_index]]
        factor (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    nch, nt, nsta = data.shape
    scale_factor = random.uniform(min(1, factor), max(1, factor))
    with torch.no_grad():
        data_ = F.interpolate(data.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
    picks_ = []
    for phase in picks:
        tmp = []
        for p in phase:
            tmp.append([p[0], p[1]*scale_factor])
        picks_.append(tmp)
    return data_, picks_

def stack_event(data1, target1, data2, target2, snr1=1, snr2=1, mask1=None, mask2=None, min_shift=0, max_shift=1024*3):

    tries = 0
    max_tries = 100
    nch, nt, nsta = data2.shape
    success = False
    while tries < max_tries:
        # shift = random.randint(-nt, nt)
        shift = random.randint(-max_shift, max_shift)
        if mask2 is not None:
            mask2_ = torch.clone(mask2)
            mask2_ = torch.roll(mask2_, shift, dims=-2)
            if torch.max(mask1 + mask2_) >= 2:
                tries += 1
                continue

        data2_ = torch.clone(data2)
        data2_ = torch.roll(data2_, shift, dims=-2)
        target2_ = torch.clone(target2)
        target2_ = torch.roll(target2_, shift, dims=-2)

        ## approximately after normalization, noise=1, signal=snr, so signal ~ noise * snr 
        # data = data1 + data2_ * (1 + max(0, snr1 - 1.0) * torch.rand(1) * 0.5)
        data = data1 * (1 + torch.rand(1) * 2) + data2_ * (1 + torch.rand(1) * 2)
        target = torch.zeros_like(target1)
        target[1:, :, :] = target1[1:, :, :] + target2_[1:, :, :]
        tmp = torch.sum(target[1:, :, :], axis=0)
        target[0, :, :] = torch.maximum(torch.tensor(0.0), 1.0 - tmp)
        success = True
        break

    if tries >= max_tries:
        data = data1
        target = target1
        print(f"stack event failed, tries={tries}")

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(data[0], vmin=-0.5, vmax=0.5, interpolation=None, aspect="auto", cmap='seismic')
    # plt.subplot(122)
    # plt.imshow(torch.permute(target, [1,2,0]), interpolation=None, aspect="auto", cmap='hot')
    # plt.savefig("test.png", dpi=300)
    # print("update test.png")
    # raise

    return data, target, success


class DASIterableDataset(IterableDataset):
    def __init__(
        self,
        data_path="./",
        data_list=None,
        format="h5",
        prefix="",
        suffix="",
        ## training
        training=False,
        picks=["p_picks", "s_picks"],
        noise_path=None,
        label_path=None,
        stack_noise=True,
        add_moveout=True,
        stack_event=True,
        resample_time=True,
        resample_space=True,
        filtering=False,
        filter_params={"freqmin": 0.1, "freqmax": 10.0, "corners": 4, "zerophase": True},
        ## continuous data
        dataset=None,  # "eqnet" or "mammoth" or None
        cut_patch=False,
        skip_files=None,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.data_path = data_path
        self.format = format
        self.prefix = prefix
        self.suffix = suffix
        if data_list is not None:
            self.data_list = np.loadtxt(data_list, dtype=str).tolist()
        else:
            self.data_list = [os.path.basename(x) for x in sorted(list(glob(os.path.join(data_path, f"{prefix}*{suffix}.{format}"))))]
        if skip_files is not None:
            self.data_list = self.filt_list(self.data_list, kwargs["skip_files"])
        
        self.data_list = self.data_list[rank::world_size]

        ## continuous data
        self.dataset = dataset
        self.cut_patch = cut_patch
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.01  # s
        self.dx = kwargs["dx"] if "dx" in kwargs else 10.0  # m
        self.nt = kwargs["nt"] if "nt" in kwargs else 1024 * 3
        self.nx = kwargs["nx"] if "nx" in kwargs else 1024 * 3

        ## training and data augmentation
        self.training = training
        self.picks = picks
        self.noise_path = noise_path
        self.label_path = label_path
        if label_path is not None:
            if type(label_path) is list:
                self.label_list = []
                for i in range(len(label_path)):
                    self.label_list += list(sorted(glob(os.path.join(label_path[i], f"{prefix}*{suffix}.csv"))))
            else:
                self.label_list = sorted(glob(os.path.join(label_path, f"{prefix}*{suffix}.csv")))
            self.label_list = self.label_list[rank::world_size]
        else:
            self.label_list = None
        self.min_picks = kwargs["min_picks"] if "min_picks" in kwargs else 500
        if self.noise_path is not None:
            self.noise_list = glob(os.path.join(noise_path, f"*.{format}"))
        self.stack_noise = stack_noise
        self.stack_event = stack_event
        self.resample_space = resample_space
        self.resample_time = resample_time
        self.add_moveout = add_moveout
        self.filtering = filtering

        if self.training:
            print(f"{label_path}: {len(self.label_list)} files")
        else:
            print(os.path.join(data_path, f"{prefix}*{suffix}.{format}"), f": {len(self.data_list)} files")


    def filt_list(self, data_list, skip_files):
        skip_data_list = [
            os.path.splitext(x.split("/")[-1])[0] for x in sorted(list(glob(skip_files)))
        ]  ## merged picks
        skip_data_list += [
            "_".join(os.path.splitext(x.split("/")[-1])[0].split("_")[:-2]) for x in sorted(list(glob(skip_files)))
        ]  ## raw picks
        print("Total number of files:", len(data_list))
        data_list = [x for x in data_list if os.path.splitext(x.split("/")[-1])[0] not in skip_data_list]
        print("Remaining number for processing:", len(data_list))
        return data_list

    def __len__(self):

        if self.training:
            return len(self.label_list)

        if not self.cut_patch:
            return len(self.data_list)
        else:
            if self.dataset is None:
                nt, nch = h5py.File(os.path.join(self.data_path, self.data_list[0]), "r")["data"].shape
            elif self.dataset == "mammoth":
                nch, nt = h5py.File(os.path.join(self.data_path, self.data_list[0]), "r")["Data"].shape
            else:
                raise ValueError("Unknown dataset")
            return len(self.data_list) * ((nt - 1) // self.nt + 1) * ((nch - 1) // self.nx + 1)


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # print(f"{worker_info = }")
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if self.training:
            return iter(self.sample_training(self.label_list[worker_id::num_workers]))
        else:
            return iter(self.sample(self.data_list[worker_id::num_workers]))


    def sample_training(self, file_list):

        # torch.manual_seed(self.rank)
        # random.seed(self.rank)
        # np.random.seed(self.rank)

        while True:
            
            ## load data and picks
            file = file_list[np.random.randint(0, len(file_list))]
            picks = pd.read_csv(file)
            meta = {}
            commmon_channel_index = []
            for event_index in picks["event_index"].unique():
                for phase_type in picks["phase_type"].unique():
                    commmon_channel_index.append(picks[(picks["event_index"] == event_index) & (picks["phase_type"] == phase_type)]["channel_index"].tolist())
            commmon_channel_index = set.intersection(*map(set, commmon_channel_index))
            picks = picks[picks["channel_index"].isin(commmon_channel_index)]

            meta["p_picks"] = picks[picks["phase_type"] == "P"][["channel_index", "phase_index"]].to_numpy()
            meta["s_picks"] = picks[picks["phase_type"] == "S"][["channel_index", "phase_index"]].to_numpy()
            if (len(meta["p_picks"]) < 500) or (len(meta["s_picks"]) < 500):
                continue
            tmp = file.split("/")
            tmp[-2] = "data"
            tmp[-1] = tmp[-1][:-4] + ".h5"  ## remove .csv
            try:
                with h5py.File("/".join(tmp), "r") as f:
                    data = f["data"][()]
                data = data[np.newaxis, :, :]  # nchn, nt, nsta
                data = torch.from_numpy(data.astype(np.float32))
            except:
                print(f"Failed to load signal: {file}")
                continue

            # load noise
            if self.stack_noise and (not self.noise_path):
                tries = 0
                max_tries = 10
                while tries < max_tries:
                    tmp_file = file_list[np.random.randint(0, len(file_list))]
                    tmp_picks = pd.read_csv(tmp_file)
                    if tmp_picks["phase_index"].min() < 3000:
                        tries += 1
                        continue
                    tmp = tmp_file.split("/")
                    tmp[-2] = "data"
                    tmp[-1] = tmp[-1][:-4] + ".h5"  ## remove .csv
                    try:
                        with h5py.File("/".join(tmp), "r") as f:
                            noise = f["data"][()]
                        ## The first 30s are noise in the training data
                        noise = noise[np.newaxis, :2800, :]  # nchn, nt, nsta
                        noise = torch.from_numpy(noise.astype(np.float32))
                    except:
                        print(f"Failed to load noise: {tmp_file}")
                        noise = torch.zeros_like(data)
                    noise = noise - torch.mean(noise, dim=1, keepdim=True)
                    noise = noise - torch.median(noise, dim=2, keepdims=True)[0]
                    noise = normalize(noise)
                    # noise = noise / torch.std(noise)
                    noise = pad_noise(noise)
                    break
                if tries >= max_tries:
                    print(f"Failed to find noise file for {file}")
                    noise = torch.zeros_like(data)

            ## normalize
            data = data - torch.mean(data, dim=1, keepdim=True)
            data = data - torch.median(data, dim=2, keepdims=True)[0]
            data = normalize(data)
            # data = data / torch.std(data)

            ## snr
            snr = calc_snr(data, meta["p_picks"])

            ## generate labels
            picks = [meta[x] for x in self.picks]

            if self.resample_time:
                tmp = np.random.rand()
                if tmp < 0.2:
                    data, picks = resample_time(data, picks, 5)
                elif tmp < 0.4:
                    data, picks = resample_time(data, picks, 0.5)

            targets, phase_mask = generate_label(data, picks, return_phase_mask=True)
            targets = torch.from_numpy(targets)
            phase_mask = torch.from_numpy(phase_mask)
            data, targets, phase_mask = pad_signal(data, targets, phase_mask)

            if self.resample_space:
                tmp = np.random.rand()
                if tmp < 0.2:
                    data, targets = resample_space(data, targets, 5)
                elif (tmp < 0.4) and (data.shape[-1] > 2000): 
                    data, targets = resample_space(data, targets, 0.5)
            
            status_stack_event = False
            if self.stack_event and (snr > 3) and (np.random.rand() < 0.5) :
                data, targets, status_stack_event = stack_event(data, targets, data, targets, snr, snr, phase_mask, phase_mask)

            # for ii in range(sum([len(x) for x in picks]) // self.min_picks):
            for ii in range(5):
                data_, targets_ = cut_data(data, targets)
                if data_ is None:
                    continue
                # if (np.random.rand() < 0.5) and self.add_moveout:
                #     data_, targets_ = add_moveout(data_, targets_)
                # data_ = data_[:, pre_nt:, :]
                # targets_ = targets_[:, pre_nt:, :]
                # if (snr > 10) and (np.random.rand() < 0.5):
                if not status_stack_event and self.stack_noise:
                    noise_ = cut_noise(noise)
                    data_ = stack_noise(data_, noise_, snr)

                if np.random.rand() < 0.5:
                    data_, targets_ = flip_lr(data_, targets_)
                
                if np.random.rand() < 0.2:
                    data_, targets_ = mask_data(data_, targets_)

                # data_ = data_ - torch.median(data_, dim=2, keepdims=True)[0]
                data_ = normalize(data_)

                yield {
                    "data": torch.nan_to_num(data_),
                    "targets": targets_,
                    "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{ii:02d}",
                    "height": data_.shape[-2],
                    "width": data_.shape[-1],
                }

    def sample(self, file_list):

        for file in file_list:

            sample = {}

            if self.format == "npz": 
                meta = np.load(os.path.join(self.data_path, file))
                data = meta["data"][np.newaxis, :, :]
                data = torch.from_numpy(data.astype(np.float32))

            elif self.format == "npy":
                data = np.load(os.path.join(self.data_path, file)) # (nsta, nt)
                data = data.T[np.newaxis, :, :]# (nch, nt, nsta)
                data = torch.from_numpy(data.astype(np.float32))
                sample["begin_time"] = datetime.fromisoformat("1970-01-01 00:00:00")
                sample["dt_s"] = 0.01
                sample["dx_m"] = 10.0

            elif self.format == "h5" and (self.dataset is None):
                with h5py.File(os.path.join(self.data_path, file), "r") as fp:
                    data = fp["data"][:]  # nt x nsta
                    if self.filtering:
                        b, a = scipy.signal.butter(2, 1.0, "hp", fs=100)
                        data = scipy.signal.filtfilt(b, a, data, axis=0)
                    if "begin_time" in fp["data"].attrs:
                        sample["begin_time"] = datetime.fromisoformat(fp["data"].attrs["begin_time"].rstrip("Z"))
                    if "dt_s" in fp["data"].attrs:
                        sample["dt_s"] = fp["data"].attrs["dt_s"]
                        # if self.dt / sample["dt_s"] >= 2:
                        #     downsample_ratio = int(self.dt / sample["dt_s"])
                        #     sample["dt_s"] *= downsample_ratio
                        #     data = data[::downsample_ratio, :]
                    else:
                        sample["dt_s"] = self.dt
                    if "dx_m" in fp["data"].attrs:
                        sample["dx_m"] = fp["data"].attrs["dx_m"]
                        # if self.dx / sample["dx_m"] >= 2:
                        #     downsample_ratio = int(self.dx / sample["dx_m"])
                        #     sample["dx_m"] *= downsample_ratio
                        #     data = data[:, ::downsample_ratio]
                    else:
                        sample["dx_m"] = self.dx

                    data = data[np.newaxis, :, :]
                    data = torch.from_numpy(data.astype(np.float32))

            elif (self.format == "h5") and (self.dataset == "mammoth"):
                with h5py.File(os.path.join(self.data_path, file), "r") as fp:
                    data = fp["Data"][:]
                    if "startTime" in data.attrs:
                        sample["begin_time"] = datetime.fromisoformat(data.attrs["startTime"].rstrip("Z"))
                    if "dt" in data.attrs:
                        sample["dt_s"] = data.attrs["dt"]
                    if "dCh" in data.attrs:
                        sample["dx_m"] = data.attrs["dCh"]
                    if "nt" in data.attrs:
                        sample["nt"] = data.attrs["nt"]
                    if "nCh" in data.attrs:
                        sample["nx"] = data.attrs["nCh"]
                    data = data[()].T
                data = data[np.newaxis, :, :]  # nchn, nt, nsta
                data = torch.from_numpy(data.astype(np.float32))
                data = torch.diff(data, n=1, dim=1)

            elif self.format == "segy":
                meta = {}
                data = self.load_segy(os.path.join(self.data_path, file), nTrace=self.nTrace)
                data = torch.from_numpy(data)
                with torch.no_grad():
                    data = torch.diff(data, n=1, dim=-1)
                    data = F.interpolate(
                        data.unsqueeze(dim=0),
                        scale_factor=self.raw_dt / self.dt,
                        mode="linear",
                        align_corners=False,
                    )
                    data = data.permute(0, 2, 1)
            else:
                raise (f"Unsupported format: {self.format}")

            data = data - torch.mean(data, dim=1, keepdim=True)
            data = data - torch.median(data, dim=2, keepdims=True)[0]

            if not self.cut_patch:
                yield {
                    "data": data,
                    "file_name": os.path.splitext(file.split("/")[-1])[0],
                    "begin_time": sample["begin_time"].isoformat(timespec="milliseconds"),
                    "begin_time_index": 0,
                    "begin_channel_index": 0,
                    "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                    "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                }
            else:
                _, nt, nx = data.shape
                data = F.pad(data, (0, nx % self.nx, 0, nt % self.nt), mode="constant", value=0)
                for i in list(range(0, nt, self.nt)):
                    for j in list(range(0, nx, self.nx)):
                        yield {
                            "data": data[:, i : i + self.nt, j : j + self.nx],
                            "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}",
                            "begin_time": (sample["begin_time"] + timedelta(seconds=i * sample["dt_s"])).isoformat(
                                timespec="milliseconds"
                            ),
                            "begin_time_index": i,
                            "begin_channel_index": j,
                            "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                            "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                        }


class AutoEncoderIterableDataset(DASIterableDataset):
    def __init__(
        self,
        data_path="./",
        noise_path=None,
        format="npz",
        prefix="",
        suffix="",
        training=False,
        stack_noise=False,
        filtering=False,
        **kwargs,
    ):
        super().__init__(data_path, noise_path, format=format, training=training)

    def sample(self, file_list):
        sample = {}
        # for file in file_list:
        idx = 0
        while True:
            if self.training:
                file = file_list[np.random.randint(0, len(file_list))]
            else:
                if idx >= len(file_list):
                    break
                file = file_list[idx]
                idx += 1

            if self.training and (self.format == "h5"):
                with h5py.File(file, "r") as f:
                    data = f["data"][()]
                    data = data[np.newaxis, :, :]  # nchn, nt, nsta
                    data = torch.from_numpy(data.astype(np.float32))
            else:
                raise (f"Unsupported format: {self.format}")

            data = data - np.median(data, axis=2, keepdims=True)
            data = normalize(data)  # nch, nt, nsta

            if self.training:
                for ii in range(10):
                    pre_nt = 255
                    data_ = cut_data(data, pre_nt=pre_nt)
                    if data_ is None:
                        continue
                    if np.random.rand() < 0.5:
                        data_ = add_moveout(data_)
                    data_ = data_[:, pre_nt:, :]
                    if np.random.rand() < 0.5:
                        data_ = flip_lr(data_)
                    data_ = data_ - np.median(data_, axis=2, keepdims=True)

                    yield {
                        "data": data_,
                        "targets": data_,
                        "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{ii:02d}",
                        "height": data_.shape[-2],
                        "width": data_.shape[-1],
                    }
            else:
                sample["data"] = data
                if self.nt is None:
                    self.nt = data.shape[1]
                if self.nx is None:
                    self.nx = data.shape[2]
                for i in list(range(0, data.shape[1], self.nt)):
                    if self.nt + i + 512 >= data.shape[1]:
                        tn = data.shape[1]
                    else:
                        tn = i + self.nt
                    for j in list(range(0, data.shape[2], self.nx)):
                        if self.nx + j + 512 >= data.shape[2]:
                            xn = data.shape[2]
                        else:
                            xn = j + self.nx
                        yield {
                            "data": data[:, i:tn, j:xn],
                            "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}",
                            "begin_time": (sample["begin_time"] + timedelta(i * sample["dt_s"])).isoformat(
                                timespec="milliseconds"
                            ),
                            "begin_time_index": i,
                            "begin_channel_index": j,
                            "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                            "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                        }
                        if xn == data.shape[2]:
                            break
                    if tn == data.shape[1]:
                        break


class DASDataset(Dataset):
    def __init__(
        self,
        data_path,
        noise_path=None,
        label_path=None,
        format="npz",
        prefix="",
        suffix="",
        training=True,
        stack_noise=True,
        picks=["p_picks", "s_picks"],
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.noise_path = noise_path
        self.label_path = label_path
        self.format = format
        self.training = training
        self.prefix = prefix
        self.suffix = suffix
        self.picks = picks
        self.data_list = sorted(glob(os.path.join(data_path, f"{prefix}*{suffix}.{format}")))
        if label_path is not None:
            if type(label_path) is list:
                self.label_list = []
                for i in range(len(label_path)):
                    self.label_list += list(sorted(glob(os.path.join(label_path[i], f"{prefix}*{suffix}.csv"))))
            else:
                self.label_list = sorted(glob(os.path.join(label_path, f"{prefix}*{suffix}.csv")))
        print(os.path.join(data_path, f"{prefix}*{suffix}.{format}"), len(self.data_list))
        if self.noise_path is not None:
            self.noise_list = glob(os.path.join(noise_path, f"*.{format}"))
        self.num_data = len(self.data_list)
        self.min_picks = kwargs["min_picks"] if "min_picks" in kwargs else 500
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.01
        self.dx = kwargs["dx"] if "dx" in kwargs else 10.0  # m

    def __len__(self):
        if self.label_path is not None:
            return len(self.label_list)
        return len(self.data_list)

    def __getitem__(self, idx):

        sample = {}
        if self.training and (self.format == "npz"):
            meta = np.load(self.data_list[idx])
            data = meta["data"]
            data = data[np.newaxis, :, :]
            data = torch.from_numpy(data.astype(np.float32))

        elif self.training and (self.format == "h5"):

            file = self.label_list[idx]
            picks = pd.read_csv(file)
            meta = {}
            meta["p_picks"] = picks[picks["phase_type"] == "P"][["channel_index", "phase_index"]].to_numpy()
            meta["s_picks"] = picks[picks["phase_type"] == "S"][["channel_index", "phase_index"]].to_numpy()
            # if (len(meta["p_picks"]) < 500) or (len(meta["s_picks"]) < 500):
            #     continue
            tmp = file.split("/")
            tmp[-2] = "data"
            tmp[-1] = tmp[-1][:-4] + ".h5"  ## remove .csv
            with h5py.File("/".join(tmp), "r") as f:
                data = f["data"][()]
                data = data[np.newaxis, :, :]  # nchn, nt, nsta
                data = torch.from_numpy(data.astype(np.float32))

            if self.stack_noise and (not self.noise_path):
                tries = 0
                max_tries = 10
                while tries < max_tries:
                    tmp_file = self.label_list[np.random.randint(0, len(self.label_list))]
                    tmp_picks = pd.read_csv(tmp_file)
                    if tmp_picks["phase_index"].min() < 3000:
                        tries += 1
                        continue
                    tmp = tmp_file.split("/")
                    tmp[-2] = "data"
                    tmp[-1] = tmp[-1][:-4] + ".h5"  ## remove .csv
                    with h5py.File("/".join(tmp), "r") as f:
                        noise = f["data"][()]
                        noise = noise[np.newaxis, :, :]  # nchn, nt, nsta
                        noise = torch.from_numpy(noise.astype(np.float32))
                    break
                if tries >= max_tries:
                    print(f"Failed to find noise file for {file}")
                    noise = torch.zeros_like(data)

        elif self.format == "npz":
            meta = np.load(self.data_list[idx])
            data = meta["data"]
            data = data[np.newaxis, :, :]
            # data = np.diff(data, axis=-2)
            # b, a = scipy.signal.butter(2, 4, 'hp', fs=100)
            # b, a = scipy.signal.butter(2, [0.5, 2.5], 'bandpass', fs=100)
            # data = scipy.signal.filtfilt(b, a, data, axis=-2)
            data = torch.from_numpy(data.astype(np.float32))

        elif self.format == "h5":
            begin_time_index = 0
            begin_channel_index = 0
            with h5py.File(self.data_list[idx], "r") as f:
                data = f["data"][()]
                # data = data[np.newaxis, :, :]
                data = data[np.newaxis, begin_time_index:, begin_channel_index:]
                if "begin_time" in f["data"].attrs:
                    if begin_time_index == 0:
                        sample["begin_time"] = datetime.fromisoformat(
                            f["data"].attrs["begin_time"].rstrip("Z")
                        ).isoformat(timespec="milliseconds")
                    else:
                        sample["begin_time_index"] = begin_time_index
                        sample["begin_time"] = (
                            datetime.fromisoformat(f["data"].attrs["begin_time"].rstrip("Z"))
                            + timedelta(seconds=begin_time_index * f["data"].attrs["dt_s"])
                        ).isoformat(timespec="milliseconds")
                if "dt_s" in f["data"].attrs:
                    sample["dt_s"] = f["data"].attrs["dt_s"]
                if "dx_m" in f["data"].attrs:
                    sample["dx_m"] = f["data"].attrs["dx_m"]
                data = torch.from_numpy(data.astype(np.float32))

        elif self.format == "segy":
            data = load_segy(os.path.join(self.data_path, self.data_list[idx]), nTrace=self.nTrace)
            data = torch.from_numpy(data)
            with torch.no_grad():
                data = torch.diff(data, n=1, dim=-1)
                data = F.interpolate(
                    data.unsqueeze(dim=0),
                    scale_factor=self.raw_dt / self.dt,
                    mode="linear",
                    align_corners=False,
                )
                data = data.permute(0, 2, 1)
        else:
            raise (f"Unsupported format: {self.format}")

        # data = normalize_local_1d(data)
        data = data - np.median(data, axis=2, keepdims=True)
        data = normalize(data)

        if self.training:
            if self.stack_noise:
                if torch.max(torch.abs(noise)) > 0:
                    noise = normalize(noise)
            picks = [meta[x] for x in self.picks]
            targets = generate_label(data, picks)
            targets = torch.from_numpy(targets)
            snr = calc_snr(data, meta["p_picks"])
            with_event = False
            if (snr > 3) and (np.random.rand() < 0.3):
                data, targets = stack_event(data, targets, data, targets, snr)
                with_event = True
            pre_nt = 255
            data, targets = cut_data(data, targets, pre_nt=pre_nt)
            if np.random.rand() < 0.5:
                data, targets = add_moveout(data, targets)
            data = data[:, pre_nt:, :]
            targets_ = targets[:, pre_nt:, :]
            # if (snr > 10) and (np.random.rand() < 0.5):
            if not with_event:
                noise = cut_noise(noise)
                data = stack_noise(data, noise, snr)
            if np.random.rand() < 0.5:
                data, targets = flip_lr(data, targets)

            data = data - np.median(data, axis=2, keepdims=True)
            sample["targets"] = targets

        sample["data"] = data
        sample["file_name"] = os.path.splitext(self.data_list[idx].split("/")[-1])[0]
        sample["height"], sample["width"] = sample["data"].shape[-2:]

        return sample
