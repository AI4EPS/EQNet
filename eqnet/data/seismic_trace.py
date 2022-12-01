import os
import random
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
import logging
from collections import defaultdict
from scipy import signal

# import warnings
# warnings.filterwarnings("error")
# import numpy
# numpy.seterr(all='raise')


def normalize(data):
    """
    data: [3, nt, nsta] or [3, nt]
    """
    data = data - data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0.0] = 1.0
    data = data / std
    return data


def generate_label(
    phase_list,
    label_width=[100],
    nt=8192,
    mask_width=None,
    return_mask=True,
):

    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)
    mask = np.zeros([nt], dtype=np.float32)

    if len(label_width) == 1:
        label_width = label_width * len(phase_list)
    if mask_width is None:
        mask_width = [int(x * 1.5) for x in label_width]

    for i, (picks, w, m) in enumerate(zip(phase_list, label_width, mask_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.05] = 0.0
            target[i + 1, :] += gaussian
            mask[int(phase_time) - m : int(phase_time) + m] = 1.0

    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    if return_mask:
        return target, mask
    else:
        return target


def stack_event(
    meta1,
    meta2,
    max_shift=2048,
):

    waveform1 = meta1["waveform"].copy()
    waveform2 = meta2["waveform"].copy()
    phase_pick1 = meta1["phase_pick"].copy()
    phase_pick2 = meta2["phase_pick"].copy()
    phase_mask1 = meta1["phase_mask"].copy()
    phase_mask2 = meta2["phase_mask"].copy()
    event_center1 = meta1["event_center"].copy()
    event_center2 = meta2["event_center"].copy()
    event_location1 = meta1["event_location"].copy()
    event_location2 = meta2["event_location"].copy()
    event_mask1 = meta1["event_mask"].copy()
    event_mask2 = meta2["event_mask"].copy()
    polarity1 = meta1["polarity"].copy()
    polarity2 = meta2["polarity"].copy()
    polarity_mask1 = meta1["polarity_mask"].copy()
    polarity_mask2 = meta2["polarity_mask"].copy()

    first_arrival1 = meta1["first_arrival"].copy()
    first_arrival2 = meta2["first_arrival"].copy()
    amp_noise1 = meta1["amp_noise"].copy()
    amp_noise2 = meta2["amp_noise"].copy()
    amp_signal1 = meta1["amp_signal"].copy()
    amp_signal2 = meta2["amp_signal"].copy()

    _, nt, nx = waveform1.shape  # nch, nt, nx
    duration_mask1 = np.zeros([nt, nx])
    duration_mask2 = np.zeros([nt, nx])
    for i, x in enumerate(meta1["duration"]):
        duration_mask1[x[0] : x[1], i] = 1
    for i, x in enumerate(meta2["duration"]):
        duration_mask2[x[0] : x[1], i] = 1

    max_tries = 30
    # while random.random() < 0.5:
    for i in range(random.randint(1, 6)):
        tries = 0
        while tries < max_tries:

            min_ratio2 = np.log10(amp_noise1 * 2 / amp_signal2)
            max_ratio2 = np.log10(amp_signal1 / 2 / amp_noise2)
            if min_ratio2 > max_ratio2:
                # print(f"min_ratio2 > max_ratio2: {min_ratio2} > {max_ratio2}, {amp_noise1 = }, {amp_signal1 = }, {amp_noise2 = }, {amp_signal2 = }")
                break

            shift = random.randint(-max_shift, max_shift) + first_arrival1 - first_arrival2
            tmp_mask2 = np.roll(phase_mask2, shift, axis=0)
            if np.max(phase_mask1 + tmp_mask2) >= 2.0:
                tries += 1
                continue
            tmp_mask2 = np.roll(event_mask2, shift, axis=0)
            if np.max(event_mask1 + tmp_mask2) >= 2.0:
                tries += 1
                continue
            tmp_mask2 = np.roll(duration_mask2, shift, axis=0)
            if np.max(duration_mask1 + tmp_mask2) >= 2.0:
                tries += 1
                continue

            waveform2_ = np.roll(waveform2, shift, axis=1)
            phase_pick2_ = np.roll(phase_pick2, shift, axis=1)
            phase_mask2_ = np.roll(phase_mask2, shift, axis=0)
            event_center2_ = np.roll(event_center2, shift, axis=0)
            event_mask2_ = np.roll(event_mask2, shift, axis=0)
            polarity2_ = np.roll(polarity2, shift, axis=0)
            polarity_mask2_ = np.roll(polarity_mask2, shift, axis=0)
            duration_mask2_ = np.roll(duration_mask2, shift, axis=0)

            ratio2 = 10 ** (random.uniform(max(-3, min_ratio2), min(3, max_ratio2)))
            flip = random.choice([-1.0, 1.0])  ## flip waveform2 polarity
            waveform1 = waveform1 + waveform2_ * ratio2 * flip
            amp_noise1 = amp_noise1 + amp_noise2 * ratio2
            amp_signal1 = min(amp_signal1, amp_signal2 * ratio2)

            phase_pick = np.zeros_like(phase_pick1)
            phase_pick[1:, :] = phase_pick1[1:, :, :] + phase_pick2_[1:, :, :]
            phase_pick[0, :] = np.maximum(0, 1.0 - np.sum(phase_pick[1:, :, :], axis=0, keepdims=True))
            phase_pick1 = phase_pick

            phase_mask1 = np.minimum(1.0, phase_mask1 + phase_mask2_)
            event_location = np.zeros_like(event_location1)
            event_location[:, event_mask1 >= 1.0] = event_location1[:, event_mask1 >= 1.0]
            event_location[:, event_mask2_ >= 1.0] = event_location2[:, event_mask2_ >= 1.0]
            event_location1 = event_location
            event_center1 = event_center1 + event_center2_
            event_mask1 = np.minimum(1.0, event_mask1 + event_mask2_)
            polarity1 = ((polarity1 - 0.5) + (polarity2_ - 0.5) * flip) + 0.5
            polarity_mask1 = np.minimum(1.0, polarity_mask1 + polarity_mask2_)
            duration_mask1 = np.minimum(1.0, duration_mask1 + duration_mask2_)
            break

        # if tries == max_tries:
        #     print(f"stack {i}-th event fails after {max_tries} tries")

    return {
        "waveform": waveform1,
        "phase_pick": phase_pick1,
        "phase_mask": phase_mask1,
        "event_center": event_center1,
        "event_location": event_location1,
        "event_mask": event_mask1,
        "polarity": polarity1,
        "polarity_mask": polarity_mask1,
        "station_location": meta1["station_location"],
    }


def cut_data(meta, nt=1024 * 4, min_point=200):

    nch0, nt0, nx0 = meta["waveform"].shape  # [3, nt, nsta]

    tries = 0
    max_tries = 100
    phase_mask = meta["phase_mask"].copy()
    it = np.random.randint(0, nt0)
    tmp = np.roll(phase_mask, -it, axis=0)
    while tmp[:nt, :].sum() < min_point:
        it = np.random.randint(0, nt0)
        tmp = np.roll(phase_mask, -it, axis=0)
        tries += 1
        if tries > max_tries:
            print(f"cut data failed, tries={tries}, it={it}")
            break

    waveform = np.roll(meta["waveform"], -it, axis=1)[:, :nt, :]
    phase_pick = np.roll(meta["phase_pick"], -it, axis=1)[:, :nt, :]
    phase_mask = np.roll(meta["phase_mask"], -it, axis=0)[:nt, :]
    event_center = np.roll(meta["event_center"], -it, axis=0)[:nt, :]
    event_location = np.roll(meta["event_location"], -it, axis=1)[:, :nt, :]
    event_mask = np.roll(meta["event_mask"], -it, axis=0)[:nt, :]
    polarity = np.roll(meta["polarity"], -it, axis=0)[:nt, :]
    polarity_mask = np.roll(meta["polarity_mask"], -it, axis=0)[:nt, :]

    return {
        "waveform": waveform,
        "phase_pick": phase_pick,
        "phase_mask": phase_mask,
        "event_center": event_center,
        "event_location": event_location,
        "event_mask": event_mask,
        "polarity": polarity,
        "polarity_mask": polarity_mask,
        "station_location": meta["station_location"],
    }


def flip_polarity(meta):
    meta["waveform"] *= -1
    meta["polarity"] = 1 - meta["polarity"]
    return meta


class SeismicTraceIterableDataset(IterableDataset):

    degree2km = 111.32
    nt = 4096  ## 8992
    feature_scale = 16
    feature_nt = nt // feature_scale

    def __init__(
        self,
        data_path=None,
        data_list=None,
        hdf5_file=None,
        format="h5",
        phases=["P", "S"],
        training=False,
        ## for training
        phase_width=[50],
        polarity_width=[50],
        event_width=[100],
        min_snr=3.0,
        stack_event=False,
        flip_polarity=False,
        ## for prediction
        sampling_rate=100,
        response_xml=None,
        highpass_filter=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.hdf5_fp = None
        if hdf5_file is not None:
            fp = h5py.File(hdf5_file, "r")
            self.hdf5_fp = fp
            tmp_hdf5_keys = f"/tmp/{hdf5_file.split('/')[-1]}.txt"
            if not os.path.exists(tmp_hdf5_keys):
                self.data_list = [event + "/" + station for event in fp.keys() for station in list(fp[event].keys())]
                with open(tmp_hdf5_keys, "w") as f:
                    for x in self.data_list:
                        f.write(x + "\n")
            else:
                self.data_list = pd.read_csv(tmp_hdf5_keys, header=None, names=["trace_id"])["trace_id"].values.tolist()
        elif data_list is not None:
            with open(data_list, "r") as f:
                self.data_list = f.read().splitlines()
        elif data_path is not None:
            self.data_list = [os.path.basename(x) for x in sorted(list(glob(os.path.join(data_path, f"*.{format}"))))]
        else:
            self.data_list = None
        if self.data_list is not None:
            self.data_list = self.data_list[rank::world_size]

        self.data_path = data_path
        self.hdf5_file = hdf5_file
        self.phases = phases
        self.response_xml = response_xml
        self.sampling_rate = sampling_rate
        self.highpass_filter = highpass_filter

        ## training
        self.training = training
        self.phase_width = phase_width
        self.polarity_width = polarity_width
        self.event_width = event_width
        self.stack_event = stack_event
        self.flip_polarity = flip_polarity
        self.min_snr = min_snr

        if self.training:
            print(f"{self.data_path}: {len(self.data_list)} files")
        else:
            print(
                os.path.join(data_path, f".{format}"),
                f": {len(self.data_list)} files",
            )

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
            return iter(self.sample(data_list))

    def __len__(self):
        return len(self.data_list)

    def calc_snr(self, waveform, picks, noise_window=300, signal_window=300, gap_window=50):

        noises = []
        signals = []
        snr = []
        for i in range(waveform.shape[0]):
            for j in picks:
                if j + gap_window < waveform.shape[1]:
                    # noise = np.std(waveform[i, j - noise_window : j - gap_window])
                    # signal = np.std(waveform[i, j + gap_window : j + signal_window])
                    noise = np.max(np.abs(waveform[i, j - noise_window : j - gap_window]))
                    signal = np.max(np.abs(waveform[i, j + gap_window : j + signal_window]))
                    if (noise > 0) and (signal > 0):
                        signals.append(signal)
                        noises.append(noise)
                        snr.append(signal / noise)
                    else:
                        signals.append(0)
                        noises.append(0)
                        snr.append(0)

        if len(snr) == 0:
            return 0.0, 0.0, 0.0
        else:
            return snr[-1], signals[-1], noises[-1]
        # else:
        # idx = np.argmax(snr).item()
        # return snr[idx], signals[idx], noises[idx]

    # def resample_time(self, waveform, picks, factor=1.0):
    #     nch, nt = waveform.shape
    #     scale_factor = random.uniform(min(1, factor), max(1, factor))
    #     with torch.no_grad():
    #         data_ = F.interpolate(data.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
    #         if noise is not None:
    #             noise_ = F.interpolate(noise.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
    #         else:
    #             noise_ = None
    #     picks_ = []
    #     for phase in picks:
    #         tmp = []
    #         for p in phase:
    #             tmp.append([p[0], p[1] * scale_factor])
    #         picks_.append(tmp)
    #     return data_, picks_, noise_

    def _read_training_h5(self, trace_id):

        if self.hdf5_fp is None:
            hdf5_fp = h5py.File(os.path.join(self.data_path, trace_id), "r")
            event_id = "data"
            sta_ids = list(hdf5_fp["data"].keys())
            np.random.shuffle(sta_ids)
            for sta_id in sta_ids:
                trace_id = event_id + "/" + sta_id
                waveform = hdf5_fp[trace_id][:, :].T  # [3, Nt]
                tmp_max = np.max(np.abs(waveform), axis=1)
                if np.all(tmp_max > 0):  ## three component data
                    break
        else:
            hdf5_fp = self.hdf5_fp
            event_id, sta_id = trace_id.split("/")
            waveform = hdf5_fp[trace_id][:, :].T  # [3, Nt]

        # waveform = hdf5_fp[trace_id][:, :].T  # [3, Nt]
        waveform = normalize(waveform)
        nch, nt = waveform.shape

        ## phase picks
        attrs = hdf5_fp[trace_id].attrs
        meta = {}
        for phase in self.phases:
            meta[phase] = attrs["phase_index"][attrs["phase_type"] == phase]
        picks = [meta[x] for x in self.phases]

        # waveform, picks = self.resample_time(waveform, picks, factor=1.0)

        ## calc snr
        snr, amp_signal, amp_noise = self.calc_snr(waveform, meta["P"])
        if snr < self.min_snr:
            return None

        ## phase arrival labels
        phase_pick, phase_mask = generate_label(picks, nt=nt, label_width=self.phase_width, return_mask=True)

        ## phase polarity
        up = attrs["phase_index"][attrs["phase_polarity"] == "U"]
        dn = attrs["phase_index"][attrs["phase_polarity"] == "D"]
        phase_up, mask_up = generate_label([up], nt=nt, label_width=self.polarity_width, return_mask=True)
        phase_dn, mask_dn = generate_label([dn], nt=nt, label_width=self.polarity_width, return_mask=True)
        polarity = ((phase_up[1, :] - phase_dn[1, :]) + 1.0) / 2.0
        polarity_mask = mask_up + mask_dn
        # polarity_mask = phase_mask

        ## P/S center time
        event_ids = set(attrs["event_id"])
        c0 = []
        duration = []
        for e in event_ids:
            c0.append(np.mean(attrs["phase_index"][attrs["event_id"] == e]).item())
            # duration.append([np.min(attrs["phase_index"][attrs["event_id"] == e]).item(), np.max(attrs["phase_index"][attrs["event_id"] == e]).item()])
            tmp_min = np.min(attrs["phase_index"][attrs["event_id"] == e]).item()
            tmp_max = np.max(attrs["phase_index"][attrs["event_id"] == e]).item()
            duration.append([tmp_min, tmp_max + 2 * (tmp_max - tmp_min)])
        event_center, event_mask = generate_label([c0], nt=nt, label_width=self.event_width, return_mask=True)
        event_center = event_center[1, :]

        ## station location
        station_location = np.array(
            [
                round(
                    attrs["longitude"] * np.cos(np.radians(attrs["latitude"])) * self.degree2km,
                    2,
                ),
                round(attrs["latitude"] * self.degree2km, 2),
                round(attrs["elevation_m"] / 1e3, 2),
            ]
        )

        ## event location
        dx = round(
            (hdf5_fp[event_id].attrs["longitude"] - attrs["longitude"])
            * np.cos(np.radians(hdf5_fp[event_id].attrs["latitude"]))
            * self.degree2km,
            2,
        )
        dy = round(
            (hdf5_fp[event_id].attrs["latitude"] - attrs["latitude"]) * self.degree2km,
            2,
        )
        dz = round(hdf5_fp[event_id].attrs["depth_km"] + attrs["elevation_m"] / 1e3, 2)
        event_location = np.zeros([4, nt], dtype=np.float32)
        event_location[0, :] = np.arange(nt) - hdf5_fp[event_id].attrs["event_time_index"]
        event_location[1:, event_mask >= 1.0] = np.array([dx, dy, dz])[:, np.newaxis]

        if self.hdf5_fp is None:
            hdf5_fp.close()

        return {
            "waveform": waveform[:, :, np.newaxis],
            "phase_pick": phase_pick[:, :, np.newaxis],
            "phase_mask": phase_mask[:, np.newaxis],
            "event_center": event_center[:, np.newaxis],
            "event_location": event_location[:, :, np.newaxis],
            "event_mask": event_mask[:, np.newaxis],
            "station_location": station_location[:, np.newaxis],
            "polarity": polarity[:, np.newaxis],
            "polarity_mask": polarity_mask[:, np.newaxis],
            ## used for stack events
            "snr": snr,
            "amp_signal": amp_signal,
            "amp_noise": amp_noise,
            "first_arrival": np.min(meta["P"]),
            "duration": duration,
        }

    def sample_train(self, data_list):

        while True:

            trace_id = np.random.choice(data_list)
            # if True:
            try:
                meta = self._read_training_h5(trace_id)
            except Exception as e:
                print(f"Error reading {trace_id}:\n{e}")
                continue
            if meta is None:
                continue

            if self.stack_event and (random.random() < 0.6):
                # if True:
                try:
                    trace_id2 = np.random.choice(self.data_list)
                    meta2 = self._read_training_h5(trace_id2)
                    if meta2 is not None:
                        meta = stack_event(meta, meta2)
                except Exception as e:
                    print(f"Error reading {trace_id2}:\n{e}")

            meta = cut_data(meta, min_point=self.phase_width[0] * 2)
            if self.flip_polarity and (random.random() < 0.5):
                meta = flip_polarity(meta)

            waveform = meta["waveform"]
            # waveform = normalize(waveform)
            phase_pick = meta["phase_pick"]
            phase_mask = meta["phase_mask"][np.newaxis, ::]
            event_center = meta["event_center"][np.newaxis, :: self.feature_scale]
            polarity = meta["polarity"][np.newaxis, ::]
            polarity_mask = meta["polarity_mask"][np.newaxis, ::]
            event_location = meta["event_location"][:, :: self.feature_scale]
            event_mask = meta["event_mask"][np.newaxis, :: self.feature_scale]
            station_location = meta["station_location"]

            yield {
                "waveform": torch.from_numpy(waveform).float(),
                "phase_pick": torch.from_numpy(phase_pick).float(),
                "phase_mask": torch.from_numpy(phase_mask).float(),
                "event_center": torch.from_numpy(event_center).float(),
                "event_location": torch.from_numpy(event_location).float(),
                "event_mask": torch.from_numpy(event_mask).float(),
                "station_location": torch.from_numpy(station_location).float(),
                "polarity": torch.from_numpy(polarity).float(),
                "polarity_mask": torch.from_numpy(polarity_mask).float(),
            }

    def taper(stream):
        for tr in stream:
            tr.taper(max_percentage=0.05, type="cosine")
        return stream

    def read_mseed(self, fname, response_xml=None, highpass_filter=False, sampling_rate=100):

        try:
            stream = obspy.read(fname)
            stream = stream.merge(fill_value="latest")
            if response_xml is not None:
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

            ## highpass filtering > 1Hz
            if highpass_filter:
                trace = trace.filter("highpass", freq=1.0)

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

        nx = len(station_ids)
        nt = len(stream[0].data)
        data = np.zeros([3, nt, nx], dtype=np.float32)
        for i, sta in enumerate(sorted(station_ids)):

            for c in station_ids[sta]:
                j = comp2idx[c]

                if len(stream.select(id=sta + c)) == 0:
                    print(f"Empty trace: {sta+c} {begin_time}")
                    continue

                trace = stream.select(id=sta + c)[0]

                ## accerleration to velocity
                if sta[-1] == "N":
                    trace = trace.integrate().filter("highpass", freq=1.0)

                tmp = trace.data.astype("float32")
                data[j, : len(tmp), i] = tmp[:nt]

        return {
            "waveform": torch.from_numpy(data),
            "station_id": list(station_ids.keys()),
            "begin_time": begin_time.datetime.isoformat(timespec="milliseconds"),
            "dt_s": 1 / sampling_rate,
        }

    def sample(self, data_list):

        for f in data_list:
            meta = self.read_mseed(
                os.path.join(self.data_path, f),
                response_xml=self.response_xml,
                highpass_filter=self.highpass_filter,
                sampling_rate=self.sampling_rate,
            )
            if meta is None:
                continue
            meta["file_name"] = os.path.splitext(f)[0]
            yield meta


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # dataset = SeismicNetworkIterableDataset("../../datasets/NCEDC/ncedc_seismic_dataset_3.h5")
    dataset = SeismicTraceIterableDataset("/atomic-data/poggiali/test.hdf5")
    for x in dataset:
        # print(x)
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        for i in range(x["waveform"].shape[-1]):

            axes[0].plot((x["waveform"][-1, :, i]) / torch.std(x["waveform"][-1, :, i]) / 10 + i)

            axes[1].plot(x["phase_pick"][1, :, i] + i)
            axes[1].plot(x["phase_pick"][2, :, i] + i)

            axes[2].plot(x["center_heatmap"][:, i] + i - 0.5)
            # axes[2].scatter(x["event_location"][0, :, i], x["event_location"][1, :, i])

            axes[3].plot(x["event_location"][0, :, i] / 10 + i)

            t = np.arange(x["event_location"].shape[1])[x["event_location_mask"][:, i] == 1]
            axes[4].plot(
                t,
                x["event_location"][1, x["event_location_mask"][:, i] == 1, i] / 10 + i,
                color=f"C{i}",
            )
            axes[4].plot(
                t,
                x["event_location"][2, x["event_location_mask"][:, i] == 1, i] / 10 + i,
                color=f"C{i}",
            )
            axes[4].plot(
                t,
                x["event_location"][3, x["event_location_mask"][:, i] == 1, i] / 10 + i,
                color=f"C{i}",
            )

        plt.savefig("test.png")
        plt.show()

        raise
