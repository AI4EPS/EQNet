import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


def generate_label(phase_list, label_width=[150, 150], nt=8192):

    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)

    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.1] = 0.0
            target[i + 1, :] += gaussian

    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    return target


class SeismicNetworkIterableDataset(IterableDataset):

    degree2km = 111.32
    nt = 8192  ## 8992
    feature_nt = 512  ##560

    def __init__(self, hdf5_file="ncedc_event.h5", sampling_rate=100.0):
        super().__init__()
        self.hdf5_fp = h5py.File(hdf5_file, "r")
        self.event_ids = list(self.hdf5_fp.keys())
        self.sampling_rate = sampling_rate

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            data_list = self.event_ids
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            data_list = self.event_ids[worker_id::num_workers]
        return iter(self.sample(data_list))

    def __len__(self):
        return len(self.event_ids)

    def sample(self, event_ids):

        num_station = 10
        while True:

            idx = np.random.randint(0, len(event_ids))
            event_id = event_ids[idx]
            station_ids = list(self.hdf5_fp[event_id].keys())
            if len(station_ids) < num_station:
                continue
            else:
                station_ids = np.random.choice(station_ids, num_station, replace=False)

            waveforms = np.zeros([3, self.nt, len(station_ids)])
            phase_pick = np.zeros([3, self.nt, len(station_ids)])
            attrs = self.hdf5_fp[event_id].attrs
            event_location = [attrs["longitude"], attrs["latitude"], attrs["depth_km"], attrs["event_time_index"]]
            station_location = []

            for i, sta_id in enumerate(station_ids):

                trace_id = event_id + "/" + sta_id

                waveforms[:, :, i] = self.hdf5_fp[trace_id][: self.nt, :].T
                attrs = self.hdf5_fp[trace_id].attrs
                p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
                s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
                phase_pick[:, :, i] = generate_label([p_picks, s_picks], nt=self.nt)

                station_location.append([attrs["longitude"], attrs["latitude"], -attrs["elevation_m"]/1e3])

            std = np.std(waveforms, axis=1, keepdims=True)
            std[std == 0] = 1.0
            waveforms = (waveforms - np.mean(waveforms, axis=1, keepdims=True)) / std
            waveforms = waveforms.astype(np.float32)

            yield {
                "waveform": torch.from_numpy(waveforms).float(),
                "phase_pick": torch.from_numpy(phase_pick).float(),
                "event_location": event_location,
                "station_location": station_location,
            }


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dataset = SeismicNetworkIterableDataset("../../datasets/NCEDC/ncedc_event_dataset_3c.h5")
    
    for x in dataset:

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(x["waveform"].shape[-1]):

            axes[0].plot((x["waveform"][-1, :, i]) / torch.std(x["waveform"][-1, :, i]) / 10 + i)

            axes[1].plot(x["phase_pick"][1, :, i] + i)
            axes[1].plot(x["phase_pick"][2, :, i] + i)

            axes[2].plot(x["event_location"][0], x["event_location"][1], "*")

            for sta in x["station_location"]:
                axes[2].plot(sta[0], sta[1], "^")

        plt.savefig("test.png")
        plt.show()

        raise
