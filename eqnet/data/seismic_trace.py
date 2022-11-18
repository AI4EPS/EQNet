import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset


def generate_label(phase_list, label_width=[100, 100], nt=8192):

    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)

    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.1] = 0.0
            target[i + 1, :] += gaussian

    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    return target


def cut_data(data, nt=4096):

    nch0, nt0, nsta0 = data["waveform"].shape  # [3, nt, nsta]
    it0 = np.random.randint(0, max(1, nt0 - nt))

    tries = 0
    max_tries = 10
    while data["waveform_mask"][it0 : it0 + nt, ...].sum() == 0:
        it0 = np.random.randint(0, max(1, nt0 - nt))
        tries += 1
        if tries > max_tries:
            print(f"cut data failed, tries={tries}")
            it0 = 0
            break

    return {
        "waveform": data["waveform"][:, it0 : it0 + nt, :].copy(),
        "waveform_mask": data["waveform_mask"][it0 : it0 + nt, :].copy(),
        "phase_pick": data["phase_pick"][:, it0 : it0 + nt, :].copy(),
        "center_heatmap": data["center_heatmap"][it0 : it0 + nt, :].copy(),
        "station_location": data["station_location"],
        "event_location": data["event_location"][:, it0 : it0 + nt, :].copy(),
        "event_location_mask": data["event_location_mask"][it0 : it0 + nt, :].copy(),
        "polarity": data["polarity"][it0 : it0 + nt, :].copy(),
        "polarity_mask": data["polarity_mask"][it0 : it0 + nt, :].copy(),
    }


class SeismicTraceIterableDataset(IterableDataset):

    degree2km = 111.32
    nt = 4096  ## 8992
    feature_scale = 16
    feature_nt = nt // feature_scale

    def __init__(self, hdf5_file="ncedc_event.h5", trace_ids=None, sampling_rate=100):
        super().__init__()
        fp = h5py.File(hdf5_file, "r")
        self.hdf5_fp = fp
        tmp_hdf5_keys = f"/tmp/{hdf5_file.split('/')[-1]}.txt"
        if not os.path.exists(tmp_hdf5_keys):
            self.trace_ids = [event + "/" + station for event in fp.keys() for station in list(fp[event].keys())]
            with open(tmp_hdf5_keys, "w") as f:
                for x in self.trace_ids:
                    f.write(x + "\n")
        else:
            self.trace_ids = pd.read_csv(tmp_hdf5_keys, header=None, names=["trace_id"])["trace_id"].values.tolist()
        self.sampling_rate = sampling_rate

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            data_list = self.trace_ids
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            data_list = self.trace_ids[worker_id::num_workers]
        return iter(self.sample(data_list))

    def __len__(self):
        return len(self.trace_ids)

    # def _read_training_h5(self, trace_id):

    #     event_id, sta_id = trace_id.split("/")

    #     waveform = self.hdf5_fp[trace_id][:, :].T  # [3, Nt]
    #     nt = waveform.shape[1]

    #     ## P/S picks
    #     attrs = self.hdf5_fp[trace_id].attrs
    #     p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
    #     s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
    #     if len(p_picks) != len(s_picks):
    #         print(f"picks not match, {trace_id}: {len(p_picks)}, {len(s_picks)}")

    #     ## TODO: remove this part
    #     if len(p_picks) == 0:
    #         p_picks = [0]
    #     if len(s_picks) == 0:
    #         s_picks = [0]

    #     phase_pick = generate_label(
    #         [p_picks, s_picks],
    #         # label_width=[100, 100],
    #         nt=nt,
    #     )
    #     waveform_mask = np.zeros(nt, dtype=np.int32)
    #     waveform_mask[min(p_picks) - 1 * self.sampling_rate : max(s_picks) + 1 * self.sampling_rate] = 1

    #     ## P/S center time
    #     ## TODO: how to deal with multiple phases
    #     ## center = (self.hdf5_fp[event_id+'/'+sta_id].attrs["phase_index"][::2] + self.hdf5_fp[event_id+'/'+sta_id].attrs["phase_index"][1::2])/2.0
    #     ## assuming only one event with both P and S picks
    #     c0 = np.mean(self.hdf5_fp[trace_id].attrs["phase_index"]).item()
    #     center_heatmap = generate_label(
    #         [
    #             [c0],
    #         ],
    #         # label_width=[
    #         #     # 10 * self.feature_scale,
    #         #     100
    #         # ],
    #         nt=nt,
    #     )[1, :]

    #     ## station location
    #     station_location = np.array(
    #         [
    #             round(
    #                 self.hdf5_fp[trace_id].attrs["station_longitude"]
    #                 * np.cos(np.radians(self.hdf5_fp[trace_id].attrs["station_latitude"]))
    #                 * self.degree2km,
    #                 2,
    #             ),
    #             round(self.hdf5_fp[trace_id].attrs["station_latitude"] * self.degree2km, 2),
    #             round(self.hdf5_fp[trace_id].attrs["station_elevation_m"], 2),
    #         ]
    #     )

    #     ## event location
    #     dx = round(
    #         (self.hdf5_fp[event_id].attrs["longitude"] - self.hdf5_fp[trace_id].attrs["longitude"])
    #         * np.cos(np.radians(self.hdf5_fp[event_id].attrs["latitude"]))
    #         * self.degree2km,
    #         2,
    #     )
    #     dy = round(
    #         (self.hdf5_fp[event_id].attrs["latitude"] - self.hdf5_fp[trace_id].attrs["latitude"]) * self.degree2km,
    #         2,
    #     )
    #     dz = round(
    #         self.hdf5_fp[event_id].attrs["depth_km"] + self.hdf5_fp[trace_id].attrs["elevation_m"] / 1e3,
    #         2,
    #     )
    #     mask = center_heatmap >= 0.5
    #     event_location = np.zeros([4, nt], dtype=np.float32)
    #     # event_location[0, :] = (np.arange(nt) - self.hdf5_fp[event_id].attrs["event_time_index"])
    #     event_location[0, :] = np.arange(nt) - 3000
    #     event_location[1:, mask] = np.array([dx, dy, dz])[:, np.newaxis]
    #     event_location_mask = mask

    #     # return {
    #     #     "waveform": torch.from_numpy(waveform).float(),
    #     #     "waveform_mask": torch.from_numpy(waveform_mask).float(),
    #     #     "phase_pick": torch.from_numpy(phase_pick).float(),
    #     #     "center_heatmap": torch.from_numpy(center_heatmap).float(),
    #     #     "station_location": torch.from_numpy(station_location).float(),
    #     #     "event_location": torch.from_numpy(event_location).float(),
    #     #     "waveform_mask": torch.from_numpy(waveform_mask).float(),
    #     #     "event_location_mask": torch.from_numpy(event_location_mask).float(),
    #     # }

    #     return {
    #         "waveform": waveform[:, :, np.newaxis],
    #         "waveform_mask": waveform_mask[:, np.newaxis],
    #         "phase_pick": phase_pick[:, :, np.newaxis],
    #         "center_heatmap": center_heatmap[:, np.newaxis],
    #         "station_location": station_location[:, np.newaxis],
    #         "event_location": event_location[:, :, np.newaxis],
    #         "waveform_mask": waveform_mask[:, np.newaxis],
    #         "event_location_mask": event_location_mask[:, np.newaxis],
    #     }

    def _read_training_h5(self, trace_id):

        event_id, sta_id = trace_id.split("/")

        waveform = self.hdf5_fp[trace_id][:, :].T  # [3, Nt]
        nt = waveform.shape[1]

        ## P/S picks
        attrs = self.hdf5_fp[trace_id].attrs
        p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
        s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
        phase_pick = generate_label(
            [p_picks, s_picks],
            # label_width=[100, 100],
            nt=nt,
        )

        ## polarity
        first_motion = attrs["first_motion"]
        polarity = generate_label(
            [p_picks],
            label_width=[200],
            nt=nt,
        )[1, :]
        if first_motion == "U":
            polarity[polarity < 0.5] = 0.5
        elif first_motion == "D":
            polarity = 1.0 - polarity
            polarity[polarity > 0.5] = 0.5
        polarity_mask = np.zeros_like(polarity)
        i = np.random.randint(0, len(polarity_mask))
        polarity_mask[i : i + 5 * self.sampling_rate] = 1.0
        for p in p_picks:
            polarity_mask[p - 1 * self.sampling_rate : p + 1 * self.sampling_rate] = 1.0
        # # polarity_mask[polarity != 0.5] = 1.0
        # for p, s in zip(p_picks, s_picks):
        #     try:
        #         assert s > p
        #     except:
        #         print(p_picks, s_picks)
        #         # plt.figure()
        #         # plt.plot(waveform[-1, :])
        #         # plt.savefig("debug.png")
        #     width = min(s - p, 1 * self.sampling_rate)
        #     polarity_mask[p - width : p + width] = 1.0

        waveform_mask = np.zeros(nt, dtype=np.int32)
        waveform_mask[min(p_picks) - 1 * self.sampling_rate : max(s_picks) + 1 * self.sampling_rate] = 1

        ## P/S center time
        ## TODO: how to deal with multiple phases
        ## center = (self.hdf5_fp[event_id+'/'+sta_id].attrs["phase_index"][::2] + self.hdf5_fp[event_id+'/'+sta_id].attrs["phase_index"][1::2])/2.0
        ## assuming only one event with both P and S picks
        # c0 = np.mean(self.hdf5_fp[trace_id].attrs["phase_index"]).item()
        c0 = (
            (self.hdf5_fp[trace_id].attrs["phase_index"][attrs["phase_type"] == "P"])
            + (self.hdf5_fp[trace_id].attrs["phase_index"][attrs["phase_type"] == "S"])
        ) / 2.0
        center_heatmap = generate_label(
            [c0],
            # label_width=[
            #     # 10 * self.feature_scale,
            #     300
            # ],
            nt=nt,
        )[1, :]

        ## station location
        station_location = np.array(
            [
                round(
                    self.hdf5_fp[trace_id].attrs["longitude"]
                    * np.cos(np.radians(self.hdf5_fp[trace_id].attrs["latitude"]))
                    * self.degree2km,
                    2,
                ),
                round(self.hdf5_fp[trace_id].attrs["latitude"] * self.degree2km, 2),
                round(self.hdf5_fp[trace_id].attrs["elevation_m"], 2),
            ]
        )

        ## event location
        dx = round(
            (self.hdf5_fp[event_id].attrs["longitude"] - self.hdf5_fp[trace_id].attrs["longitude"])
            * np.cos(np.radians(self.hdf5_fp[event_id].attrs["latitude"]))
            * self.degree2km,
            2,
        )
        dy = round(
            (self.hdf5_fp[event_id].attrs["latitude"] - self.hdf5_fp[trace_id].attrs["latitude"]) * self.degree2km,
            2,
        )
        dz = round(
            self.hdf5_fp[event_id].attrs["depth_km"] + self.hdf5_fp[trace_id].attrs["elevation_m"] / 1e3,
            2,
        )
        mask = center_heatmap >= 0.5
        event_location = np.zeros([4, nt], dtype=np.float32)
        event_location[0, :] = np.arange(nt) - self.hdf5_fp[event_id].attrs["time_index"]
        # event_location[0, :] = np.arange(nt) - 3000
        event_location[1:, mask] = np.array([dx, dy, dz])[:, np.newaxis]
        event_location_mask = mask

        # return {
        #     "waveform": torch.from_numpy(waveform).float(),
        #     "waveform_mask": torch.from_numpy(waveform_mask).float(),
        #     "phase_pick": torch.from_numpy(phase_pick).float(),
        #     "center_heatmap": torch.from_numpy(center_heatmap).float(),
        #     "station_location": torch.from_numpy(station_location).float(),
        #     "event_location": torch.from_numpy(event_location).float(),
        #     "waveform_mask": torch.from_numpy(waveform_mask).float(),
        #     "event_location_mask": torch.from_numpy(event_location_mask).float(),
        # }

        return {
            "waveform": waveform[:, :, np.newaxis],
            "waveform_mask": waveform_mask[:, np.newaxis],
            "phase_pick": phase_pick[:, :, np.newaxis],
            "center_heatmap": center_heatmap[:, np.newaxis],
            "station_location": station_location[:, np.newaxis],
            "event_location": event_location[:, :, np.newaxis],
            "event_location_mask": event_location_mask[:, np.newaxis],
            "polarity": polarity[:, np.newaxis],
            "polarity_mask": polarity_mask[:, np.newaxis],
        }

    def sample(self, trace_ids):

        while True:

            trace_id = np.random.choice(trace_ids)
            # print(trace_id)

            data = self._read_training_h5(trace_id)

            # print(data["waveform"].shape)
            # if data["waveform"].shape[1] != 9001:
            #     print(data["waveform"].shape)
            #     continue

            # print(data["waveform"].shape)
            data = cut_data(data)
            # data = stack_noise(data)
            # data = stack_event(data)

            # waveforms = np.zeros([3, self.nt, num_station])
            # phase_pick = np.zeros([3, self.nt, num_station])
            # center_heatmap = np.zeros([self.feature_nt, num_station])
            # event_location = np.zeros([4, self.feature_nt, num_station])
            # event_location_mask = np.zeros([self.feature_nt, num_station])
            # station_location = np.zeros([num_station, 2])

            # for x in data:
            #     print(x, data[x].shape)

            waveform = data["waveform"]
            std = np.std(waveform, axis=1, keepdims=True)
            std[std == 0] = 1.0
            waveform = (waveform - np.mean(waveform, axis=1, keepdims=True)) / std
            phase_pick = data["phase_pick"]
            center_heatmap = data["center_heatmap"][np.newaxis, :: self.feature_scale]
            polarity = data["polarity"][np.newaxis, ::]
            polarity_mask = data["polarity_mask"][np.newaxis, ::]
            event_location = data["event_location"][:, :: self.feature_scale]
            event_location_mask = data["event_location_mask"][:: self.feature_scale]
            station_location = data["station_location"]

            yield {
                "waveform": torch.from_numpy(waveform).float(),
                "phase_pick": torch.from_numpy(phase_pick).float(),
                "center_heatmap": torch.from_numpy(center_heatmap).float(),
                "event_location": torch.from_numpy(event_location).float(),
                "event_location_mask": torch.from_numpy(event_location_mask).float(),
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
            axes[4].plot(t, x["event_location"][1, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")
            axes[4].plot(t, x["event_location"][2, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")
            axes[4].plot(t, x["event_location"][3, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")

        plt.savefig("test.png")
        plt.show()

        raise
