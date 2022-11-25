import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from glob import glob


def generate_label(
    phase_list,
    label_width=[
        100,
    ],
    nt=8192,
    mask_width=[
        200,
    ],
    return_mask=True,
):

    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)
    mask = np.zeros(
        [
            nt,
        ],
        dtype=np.float32,
    )

    if len(label_width) == 1:
        label_width = label_width * len(phase_list)
    if len(mask_width) == 1:
        mask_width = mask_width * len(phase_list)

    for i, (picks, w, m) in enumerate(zip(phase_list, label_width, mask_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.1] = 0.0
            target[i + 1, :] += gaussian
            mask[int(phase_time) - m : int(phase_time) + m] = 1.0

    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    if return_mask:
        return target, mask
    else:
        return target


def cut_data(data, nt=4096):

    nch0, nt0, nsta0 = data["waveform"].shape  # [3, nt, nsta]
    it0 = np.random.randint(0, max(1, nt0 - nt))

    tries = 0
    max_tries = 10
    while data["phase_mask"][it0 : it0 + nt, ...].sum() == 0:
        it0 = np.random.randint(0, max(1, nt0 - nt))
        tries += 1
        if tries > max_tries:
            print(f"cut data failed, tries={tries}")
            it0 = 0
            break

    return {
        "waveform": data["waveform"][:, it0 : it0 + nt, :].copy(),
        "phase_pick": data["phase_pick"][:, it0 : it0 + nt, :].copy(),
        "phase_mask": data["phase_mask"][it0 : it0 + nt, :].copy(),
        "event_center": data["event_center"][it0 : it0 + nt, :].copy(),
        "event_location": data["event_location"][:, it0 : it0 + nt, :].copy(),
        "event_mask": data["event_mask"][it0 : it0 + nt, :].copy(),
        "station_location": data["station_location"],
        "polarity": data["polarity"][it0 : it0 + nt, :].copy(),
        "polarity_mask": data["polarity_mask"][it0 : it0 + nt, :].copy(),
    }


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
        sampling_rate=100,
        training=False,
        phase_width=[100],
        polarity_width=[100],
        event_width=[200],
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
                    for x in self.trace_ids:
                        f.write(x + "\n")
            else:
                self.data_list = pd.read_csv(tmp_hdf5_keys, header=None, names=["trace_id"])["trace_id"].values.tolist()
        elif data_list is not None:
            self.data_list = np.loadtxt(data_list, dtype=str).tolist()
        elif data_path is not None:
            self.data_list = [os.path.basename(x) for x in sorted(list(glob(os.path.join(data_path, f"*.{format}"))))]
        else:
            self.data_list = None
        if self.data_list is not None:
            self.data_list = self.data_list[rank::world_size]

        self.data_path = data_path
        self.hdf5_file = hdf5_file
        self.phases = phases
        self.sampling_rate = sampling_rate

        ## training
        self.training = training
        self.phase_width = phase_width
        self.polarity_width = polarity_width
        self.event_width = event_width

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
        return iter(self.sample(data_list))

    def __len__(self):
        return len(self.data_list)

    def _read_training_h5(self, trace_id):

        if self.hdf5_fp is None:
            hdf5_fp = h5py.File(os.path.join(self.data_path, trace_id), "r")
            event_id = "data"
            sta_id = np.random.choice(list(hdf5_fp["data"].keys()))
            trace_id = event_id + "/" + sta_id
        else:
            event_id, sta_id = trace_id.split("/")

        waveform = hdf5_fp[trace_id][:, :].T  # [3, Nt]
        nt = waveform.shape[1]

        ## phase picks
        attrs = hdf5_fp[trace_id].attrs
        meta = {}
        for phase in self.phases:
            meta[phase] = attrs["phase_index"][attrs["phase_type"] == phase]
        picks = [meta[x] for x in self.phases]
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
        for e in event_ids:
            c0.append(np.mean(attrs["phase_index"][attrs["event_id"] == e]).item())
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
        }

    def sample(self, data_list):

        while True:

            trace_id = np.random.choice(data_list)
            try:
                data = self._read_training_h5(trace_id)
            except:
                print(f"Error reading {trace_id}")
                continue

            data = cut_data(data)

            waveform = data["waveform"]
            std = np.std(waveform, axis=1, keepdims=True)
            std[std == 0] = 1.0
            waveform = (waveform - np.mean(waveform, axis=1, keepdims=True)) / std
            phase_pick = data["phase_pick"]
            phase_mask = data["phase_mask"][np.newaxis, ::]
            event_center = data["event_center"][np.newaxis, :: self.feature_scale]
            polarity = data["polarity"][np.newaxis, ::]
            polarity_mask = data["polarity_mask"][np.newaxis, ::]
            event_location = data["event_location"][:, :: self.feature_scale]
            event_mask = data["event_mask"][np.newaxis, :: self.feature_scale]
            station_location = data["station_location"]

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
