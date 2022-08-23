import os
from datetime import datetime, timedelta
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset


def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x


def normalize(data):
    """
    data: [Nch, Nt, Nsta] (Nchn: number of channels, Nt: number of time, Nsta: number of stations)
    """
    data = data.double()
    mean = torch.mean(data, dim=(1), keepdims=True)
    std = torch.std(data, dim=(1), keepdims=True)
    data = (data - mean) / std
    return data.float()


# def normalize_numpy(data):
#     """
#     data: [Nch, Nt, Nsta] (Nchn: number of channels, Nt: number of time, Nsta: number of stations)
#     """
#     data = data.astype(np.float64)
#     mean = np.mean(data, axis=(1), keepdims=True)
#     std = np.std(data, axis=(1), keepdims=True)
#     data = (data - mean) / std
#     return data.astype(np.float32)


def normalize_local_1d(data, window=1024 + 1):
    nch, nt, nsta = data.shape
    data = data.permute((2, 0, 1))
    with torch.no_grad():
        data_ = F.pad(data, (window // 2, window // 2), mode="circular")
        # mean_pool = F.avg_pool1d(data_, kernel_size=window, stride=window // 2)
        # mean = F.interpolate(mean_pool, size=nt, align_corners=True, mode="linear")
        mean = F.avg_pool1d(data_, kernel_size=window, stride=1)
        data -= mean
        data_ = F.pad(data, (window // 2, window // 2), mode="circular")
        # std_pool = F.lp_pool1d(
        #     data_, norm_type=2, kernel_size=window, stride=window // 2
        # ) / (window ** 0.5)
        # std = F.interpolate(std_pool, size=nt, align_corners=True, mode="linear")
        std = F.lp_pool1d(data_, norm_type=2, kernel_size=window, stride=1) / (window**0.5)
        data /= std
        data = log_transform(data)
    data = data.permute((1, 2, 0))
    return data


def normalize_local(data, window=1024 + 1):
    nch, nt, nsta = data.shape
    data = data.permute((2, 0, 1))
    data = data.unsqueeze(0)  # batch, nsta, nch, nt
    with torch.no_grad():
        data_ = F.pad(data, (window // 2, window // 2, 0, 0), mode="circular")
        # mean_pool = F.avg_pool2d(
        #     data_, kernel_size=(nch, window), stride=(1, window // 2)
        # )
        # mean = F.interpolate(
        #     mean_pool, size=(nch, nt), align_corners=True, mode="bilinear"
        # )
        mean = F.avg_pool2d(data_, kernel_size=(nch, window), stride=(1, 1))
        data -= mean
        data_ = F.pad(data, (window // 2, window // 2, 0, 0), mode="circular")
        # std_pool = F.lp_pool2d(
        #     data_, norm_type=2, kernel_size=(nch, window), stride=(1, window // 2)
        # ) / ((window * nch) ** 0.5)
        # std = F.interpolate(
        #     std_pool, size=(nch, nt), align_corners=True, mode="bilinear"
        # )
        std = F.lp_pool2d(data_, norm_type=2, kernel_size=(nch, window), stride=(1, 1)) / ((window * nch) ** 0.5)
        data /= std
        data = log_transform(data)
    data = data.squeeze(0)
    data = data.permute((1, 2, 0))
    return data


def generate_label(data, phase_list, label_width=[150, 200], label_shape="gaussian", mask=None):

    target = np.zeros([len(phase_list) + 1, *data.shape[1:]], dtype=np.float32)

    label_window = []
    for w in label_width:
        if label_shape == "gaussian":
            tmp = np.exp(-((np.arange(-w // 2, w // 2 + 1)) ** 2) / (2 * (w / 6) ** 2))
            label_window.append(tmp)
        elif label_shape == "triangle":
            tmp = 1 - np.abs(2 / w * (np.arange(-w // 2, w // 2 + 1)))
            label_window.append(tmp)
        else:
            raise (f"Label shape {label_shape} should be guassian or triangle")

    if mask is None:
        mask = np.zeros((len(phase_list), data.shape[2]), dtype=np.bool)
    for i, picks in enumerate(phase_list):
        for trace, phase_time in picks:
            phase_time = int(phase_time)
            if (
                (phase_time - label_width[i] // 2 >= 0)
                and (phase_time + label_width[i] // 2 + 1 <= target.shape[1])
                and (trace < target.shape[2])
            ):

                mask[i, trace] = True
                target[
                    i + 1,
                    phase_time - label_width[i] // 2 : phase_time + label_width[i] // 2 + 1,
                    trace,
                ] = label_window[i]

    mask = np.all(mask, axis=0)  ## mask for traces with all picks

    target[0:1, :, mask] = np.maximum(0, 1 - np.sum(target[1:, :, mask], axis=0, keepdims=True))
    target[:, :, ~mask] = 0

    return target


def cut_data(data, targets=None, nt=3000 // 512 * 512, nsta=512, pre_nt=256):
    h, w = data.shape[-2:]
    # h0 = np.random.randint(0, max(1, h-nt))
    h0 = np.random.randint(pre_nt, max(1, h - nt))
    w0 = np.random.randint(0, max(1, w - nsta))
    if targets is not None:
        label_width = 150
        max_tries = 100
        max_w0 = 0
        max_h0 = 0
        max_sum = 0
        tmp_sum = 0
        tries = 0
        while tmp_sum < label_width / 2 * nsta * 0.1:
            h0 = np.random.randint(pre_nt, max(1, h - nt))
            w0 = np.random.randint(0, max(1, w - nsta))
            tmp_sum = torch.sum(targets[1:, h0 : h0 + nt, w0 : w0 + nsta])  # chn, nt, nsta
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
    data = data[..., h0 - pre_nt : h0 + nt, w0 : w0 + nsta].clone()
    if targets is not None:
        targets = targets[..., h0 - pre_nt : h0 + nt, w0 : w0 + nsta].clone()
        tmp_sum = torch.sum(targets[1:, :, :], axis=(0, 1))  ## no P/S channels
        targets[0, :, tmp_sum == 0] = 0
        return data, targets
    else:
        return data


def cut_noise(noise, nt=3000 // 512 * 512, nsta=512):
    h, w = noise.shape[-2:]
    h0 = np.random.randint(0, max(1, 3000 - nt))
    w0 = np.random.randint(0, max(1, w - nsta))
    noise = noise[..., h0 : h0 + nt, w0 : w0 + nsta].clone()
    return noise


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


def calc_snr(data, picks, pre_width=200, post_width=200):
    signal = torch.zeros(
        [
            data.shape[0],
            post_width,
        ]
    )
    noise = torch.zeros(
        [
            data.shape[0],
            pre_width,
        ]
    )
    SNR = []
    for trace, phase_time in picks:
        phase_time = int(phase_time)
        noise = torch.std(data[:, max(0, phase_time - pre_width) : phase_time, trace])
        signal = torch.std(data[:, phase_time : phase_time + post_width, trace])
        SNR.append(signal / noise)
    return np.median(SNR)
    # if phase_time - pre_width > 0:
    # noise += torch.abs(data[:, phase_time - pre_width : phase_time, trace] - torch.mean(data[:, phase_time - pre_width : phase_time, trace]))
    # if phase_time + post_width < data.shape[1]:
    # signal += torch.abs(data[:, phase_time : phase_time + post_width, trace] - torch.mean(data[:, phase_time : phase_time + post_width, trace]))
    # return torch.std(signal) / torch.std(noise)


def stack_noise(data, noise, snr):
    # return data + noise * max(0, snr - 1.0) * torch.rand(1)
    return data + noise * max(0, snr - 1.0) * torch.rand(1) * 1.5
    # return data + noise * max(0, snr - 1.0) * torch.rand(1) #* 2.0
    # return data + noise * snr * 0.5


def stack_event(data1, target1, data2, target2, snr=1, min_shift=500, max_shift=3000 // 512 * 512 - 500):

    tries = 0
    max_tries = 10
    while tries < max_tries:
        shift = torch.randint(low=min_shift, high=max_shift, size=(1,))

        data2_ = torch.zeros_like(data2)
        data2_[:, shift:, :] = data2[:, :-shift, :]
        data2_[:, :shift, :] = data2[:, -shift:, :]

        target2_ = torch.zeros_like(target2)
        target2_[:, shift:, :] = target2[:, :-shift, :]
        target2_[:, :shift, :] = target2[:, -shift:, :]

        data = data1 + data2_ * (1 + max(0, snr - 1.0) * torch.rand(1))
        target = torch.zeros_like(target1)
        target[1:, :, :] = target1[1:, :, :] + target2_[1:, :, :]
        tmp = torch.sum(target[1:, :, :], axis=0)
        if (torch.max(tmp) <= torch.max(torch.sum(target1[1:, :, :], axis=0))) or (
            torch.max(tmp) <= torch.max(torch.sum(target2[1:, :, :], axis=0))
        ):
            target[0, :, :] = torch.maximum(torch.tensor(0.0), 1.0 - tmp)
            break
        tries += 1

    if tries >= max_tries:
        data = data1
        target = target1

        # print(f"stack event failed, tries={tries}")
        # print(shift)
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(data[0], vmin=-0.5, vmax=0.5, interpolation=None, aspect="auto", cmap='seismic')
        # plt.subplot(122)
        # plt.imshow(torch.permute(target, [1,2,0]), interpolation=None, aspect="auto", cmap='hot')
        # plt.savefig("test.png")
        # print("update test.png")

    return data, target


class DASIterableDataset(IterableDataset):
    def __init__(
        self,
        data_path="./",
        noise_path=None,
        label_path=None,
        format="npz",
        prefix="",
        suffix="",
        training=False,
        stack_noise=True,
        picks=["p_picks", "s_picks"],
        filtering=False,
        # filter_params={"freqmin": 0.1, "freqmax": 10.0, "corners": 4, "zerophase": True},
        update_total_number=False,
        dataset="",
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
        self.stack_noise = stack_noise
        self.data_list = sorted(list(glob(os.path.join(data_path, f"{prefix}*{suffix}.{format}"))))
        if ("rank" in kwargs) and ("world_size" in kwargs):
            self.data_list = self.data_list[kwargs["rank"] :: kwargs["world_size"]]
        if label_path is not None:
            if type(label_path) is list:
                self.label_list = []
                for i in range(len(label_path)):
                    self.label_list += list(sorted(glob(os.path.join(label_path[i], f"{prefix}*{suffix}.csv"))))
            else:
                self.label_list = sorted(glob(os.path.join(label_path, f"{prefix}*{suffix}.csv")))
        else:
            self.label_list = None
        print(os.path.join(data_path, f"{prefix}*{suffix}.{format}"), len(self.data_list))
        if self.noise_path is not None:
            self.noise_list = glob(os.path.join(noise_path, f"*.{format}"))
        self.num_data = len(self.data_list)
        self.min_picks = kwargs["min_picks"] if "min_picks" in kwargs else 500
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.01
        self.dx = kwargs["dx"] if "dx" in kwargs else 10.0  # m
        # self.nt = kwargs["nt"] if "nt" in kwargs else 2048
        self.nt = kwargs["nt"] if "nt" in kwargs else 12000
        self.nx = kwargs["nx"] if "nx" in kwargs else 1024
        self.filtering = filtering
        self.dataset = dataset
        if self.label_path is not None:
            self.total_number = len(self.label_list)
        else:
            self.total_number = len(self.data_list)
        if update_total_number:
            self.total_number = self.update_total_number()

    def update_total_number(self):
        if self.format == "h5":
            if self.dataset == "mammoth":
                shape = h5py.File(self.data_list[0], "r")["Data"].shape
            else:
                shape = h5py.File(self.data_list[0], "r")["data"].shape
            total_number = len(self.data_list) * (shape[0] + 512) // self.nt * (shape[1] + 512) // self.nx
        else:
            total_number = len(self.data_list)
        return total_number

    def __len__(self):
        return self.total_number

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            if self.label_list is None:
                data_list = self.data_list
            else:
                data_list = self.label_list
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # num_workers = comm.get_world_size()
            # worker_id = comm.get_local_rank()
            if self.label_list is None:
                data_list = self.data_list[worker_id::num_workers]
            else:
                data_list = self.label_list[worker_id::num_workers]
        return iter(self.sample(data_list))

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
                picks = pd.read_csv(file)
                meta = {}
                meta["p_picks"] = picks[picks["phase_type"] == "P"][["channel_index", "phase_index"]].to_numpy()
                meta["s_picks"] = picks[picks["phase_type"] == "S"][["channel_index", "phase_index"]].to_numpy()
                if (len(meta["p_picks"]) < 500) or (len(meta["s_picks"]) < 500):
                    continue
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
                        tmp_file = file_list[np.random.randint(0, len(file_list))]
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
                meta = np.load(os.path.join(self.data_path, file))
                data = meta["data"][np.newaxis, :, :]
                data = torch.from_numpy(data.astype(np.float32))

            elif self.format == "h5" and (not self.dataset):
                with h5py.File(file, "r") as fp:
                    data = fp["data"][()]  # nt x nsta
                    if self.filtering:
                        # raise
                        # b, a = scipy.signal.butter(2, 4, 'hp', fs=100)
                        b, a = scipy.signal.butter(2, 1.0, "hp", fs=100)
                        data = scipy.signal.filtfilt(b, a, data, axis=0)
                    data = data[np.newaxis, :, :]
                    data = torch.from_numpy(data.astype(np.float32))
                    if "begin_time" in fp["data"].attrs:
                        sample["begin_time"] = datetime.fromisoformat(fp["data"].attrs["begin_time"].rstrip("Z"))
                    if "dt_s" in fp["data"].attrs:
                        sample["dt_s"] = fp["data"].attrs["dt_s"]
                    else:
                        sample["dt_s"] = self.dt
                    if "dx_m" in fp["data"].attrs:
                        sample["dx_m"] = fp["data"].attrs["dx_m"]
                    else:
                        sample["dx_m"] = self.dx

                    # b, a = scipy.signal.butter(2, 4, 'hp', fs=100)
                    # data = scipy.signal.filtfilt(b, a, data, axis=-2)
                    # data = torch.from_numpy(data.astype(np.float32))
                    # data = torch.diff(data, n=1, dim=-1)
                    # data = data.permute(0, 2, 1)  # nch, nt, nsta
                    # data = data - data.median(dim=2, keepdim=True).values

            elif (self.format == "h5") and (self.dataset == "mammoth"):
                with h5py.File(file, "r") as fp:
                    ## For DAS continuous data
                    data = fp["Data"]
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

                # b, a = scipy.signal.butter(2, 4, 'hp', fs=100)
                # data = scipy.signal.filtfilt(b, a, data, axis=-2)
                # data = torch.from_numpy(data.astype(np.float32))
                data = torch.diff(data, n=1, dim=1)
                data = data - data.median(dim=2, keepdim=True).values

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

            data = data - torch.median(data, dim=2, keepdims=True)[0]
            # data = normalize(data)  # nch, nt, nsta

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
                for ii in range(sum([len(x) for x in picks]) // self.min_picks * 10):
                    pre_nt = 255
                    data_, targets_ = cut_data(data, targets, pre_nt=pre_nt)
                    if data_ is None:
                        continue
                    if np.random.rand() < 0.5:
                        data_, targets_ = add_moveout(data_, targets_)
                    data_ = data_[:, pre_nt:, :]
                    targets_ = targets_[:, pre_nt:, :]
                    # if (snr > 10) and (np.random.rand() < 0.5):
                    if not with_event:
                        noise_ = cut_noise(noise)
                        data_ = stack_noise(data_, noise_, snr)
                    if np.random.rand() < 0.5:
                        data_, targets_ = flip_lr(data_, targets_)

                    data_ = data_ - np.median(data_, axis=2, keepdims=True)

                    yield {
                        "data": data_,
                        "targets": targets_,
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
                            "begin_time": (sample["begin_time"] + timedelta(seconds=i * sample["dt_s"])).isoformat(
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
