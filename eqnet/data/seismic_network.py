import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from .seismic_trace import generate_phase_label, generate_event_label


class SeismicNetworkIterableDataset(IterableDataset):
    degree2km = 111.32
    nt0 = 12000
    nt = 4096  ## 8992
    event_feature_scale = 16
    polarity_feature_scale = 1
    event_feature_nt = nt // event_feature_scale
    polarity_feature_nt = nt // polarity_feature_scale

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
        num_station = 8

        ### FIXME: DUMMY DATA
        data_old = np.zeros([3, self.nt, num_station])
        phase_pick_old = np.zeros([3, self.nt, num_station])
        phase_mask_old = np.zeros([self.nt, num_station])
        polarity_old = np.zeros([self.nt, num_station])
        polarity_mask_old = np.zeros([self.nt, num_station])
        event_center_old = np.zeros([self.nt, num_station])
        event_time_old = np.zeros([self.nt, num_station])
        event_mask_old = np.zeros([self.nt, num_station])
        station_location_old = np.zeros([3, num_station])
        prompt_center_old = np.zeros([self.nt, num_station])
        prompt_mask_old = np.zeros([self.nt, num_station])
        prompt_old = np.zeros([3])
        position_old = np.zeros([self.nt, num_station, 3])
        ###

        while True:
            idx = np.random.randint(0, len(event_ids))
            event_id = event_ids[idx]
            station_ids = list(self.hdf5_fp[event_id].keys())
            event_time_index = self.hdf5_fp[event_id].attrs["event_time_index"]
            # if len(station_ids) < num_station:
            #     continue
            # else:
            station_ids = np.random.choice(station_ids, num_station, replace=True)

            data = np.zeros([3, self.nt0, len(station_ids)])
            phase_pick = np.zeros([3, self.nt0, len(station_ids)])
            phase_mask = np.zeros([self.nt0, len(station_ids)])
            polarity = np.zeros([self.nt0, len(station_ids)])
            polarity_mask = np.zeros([self.nt0, len(station_ids)])
            event_center = np.zeros([self.nt0, len(station_ids)])
            event_time = np.zeros([self.nt0, len(station_ids)])
            event_location = np.zeros([3])
            event_mask = np.zeros([self.nt0, len(station_ids)])
            station_location = np.zeros([3, len(station_ids)])
            prompt_center = np.zeros([self.nt0, len(station_ids)])
            prompt_mask = np.zeros([self.nt0, len(station_ids)])

            position = []
            for i, sta_id in enumerate(station_ids):
                trace_id = event_id + "/" + sta_id

                # if self.hdf5_fp[trace_id][()].shape != (9000, 3):
                #     continue

                data[:, :, i] = self.hdf5_fp[trace_id][:, :]
                attrs = self.hdf5_fp[trace_id].attrs
                p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
                s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
                phase_pick[:, :, i], phase_mask[:, i] = generate_phase_label([p_picks, s_picks], nt=self.nt0)

                up = attrs["phase_index"][attrs["phase_polarity"] == "U"]
                dn = attrs["phase_index"][attrs["phase_polarity"] == "D"]


                phase_up, mask_up = generate_phase_label([up], nt=self.nt0)
                phase_dn, mask_dn = generate_phase_label([dn], nt=self.nt0)
                polarity[:, i] = ((phase_up[1, :] - phase_dn[1, :]) + 1.0) / 2.0
                polarity_mask[:, i] = mask_up + mask_dn

                ## TODO: how to deal with multiple phases
                # center = (self.hdf5_fp[trace_id].attrs["phase_index"][::2] + self.hdf5_fp[trace_id].attrs["phase_index"][1::2])/2.0
                ## assuming only one event with both P and S picks
                c0 = (
                    (self.hdf5_fp[trace_id].attrs["p_phase_index"])
                    + (self.hdf5_fp[trace_id].attrs["s_phase_index"])
                ) / 2.0
                t0 = event_time_index
                c0_width = (
                    (
                        (self.hdf5_fp[trace_id].attrs["s_phase_index"])
                        - (self.hdf5_fp[trace_id].attrs["p_phase_index"])
                    )
                    * self.sampling_rate
                    / 200.0
                ).max()
                dx = round(
                    (self.hdf5_fp[event_id].attrs["longitude"] - self.hdf5_fp[trace_id].attrs["longitude"])
                    * np.cos(np.radians(self.hdf5_fp[event_id].attrs["latitude"]))
                    * self.degree2km,
                    2,
                )
                dy = round(
                    (self.hdf5_fp[event_id].attrs["latitude"] - self.hdf5_fp[trace_id].attrs["latitude"])
                    * self.degree2km,
                    2,
                )
                dz = round(
                    self.hdf5_fp[event_id].attrs["depth_km"] + self.hdf5_fp[trace_id].attrs["elevation_m"] / 1e3,
                    2,
                )
                # dt = (c0 - self.hdf5_fp[event_id].attrs["event_time_index"]) / self.sampling_rate
                # dt = (c0 - 3000) / self.sampling_rate

                # event_center[int(c0//self.feature_scale), i] = 1
                # print(c0_width)

                event_center[:, i], event_time[:, i], event_mask[:, i] = generate_event_label([c0], [t0], nt=self.nt0)
                event_location[:] = np.array([dx, dy, dz])

                prompt_center[:, i], _, prompt_mask[:, i] = generate_event_label([c0], [t0], nt=self.nt0)

                ## station location
                station_location[0, i] = round(
                    self.hdf5_fp[trace_id].attrs["longitude"]
                    * np.cos(np.radians(self.hdf5_fp[trace_id].attrs["latitude"]))
                    * self.degree2km,
                    2,
                )
                station_location[1, i] = round(self.hdf5_fp[trace_id].attrs["latitude"] * self.degree2km, 2)
                station_location[2, i] = round(-self.hdf5_fp[trace_id].attrs["elevation_m"] / 1e3, 2)

                if i == 0:
                    prompt = np.array([c0, dx, dy]) # t, x, y
                    prompt_location = np.array([self.hdf5_fp[trace_id].attrs["longitude"], self.hdf5_fp[trace_id].attrs["latitude"]])
                # position.append([dx, dy])
                dx = round(
                    (prompt_location[0] - self.hdf5_fp[trace_id].attrs["longitude"])
                    * np.cos(np.radians(prompt_location[1]))
                    * self.degree2km,
                    2,
                )
                dy = round(
                    (prompt_location[1] - self.hdf5_fp[trace_id].attrs["latitude"])
                    * self.degree2km,
                    2,
                )
                position.append([dx, dy])

            std = np.std(data, axis=1, keepdims=True)
            std[std == 0] = 1.0
            data = (data - np.mean(data, axis=1, keepdims=True)) / std
            data = data.astype(np.float32)

            # ii = np.random.randint(0, self.nt0 - self.nt)
            ii = np.random.randint(0, 3000)
            data = data[:, ii : ii + self.nt, :]
            phase_pick = phase_pick[:, ii : ii + self.nt, :]
            phase_mask = phase_mask[np.newaxis, ii : ii + self.nt, :]
            event_center = event_center[np.newaxis, ii : ii + self.nt, :]
            event_time = event_time[np.newaxis, ii : ii + self.nt, :]
            # event_location = event_location[:, ii : ii + self.nt, :]
            event_mask = event_mask[np.newaxis, ii : ii + self.nt, :]
            polarity = polarity[np.newaxis, ii : ii + self.nt, :]
            polarity_mask = polarity_mask[np.newaxis, ii : ii + self.nt, :]

            prompt_center = prompt_center[np.newaxis, ii : ii + self.nt, :]
            prompt_mask = prompt_mask[np.newaxis, ii : ii + self.nt, :]
            
            prompt[0] -= ii #[3,] ## FIXME: Double check if the prompt and position time is the same
            t = np.arange(self.nt)[::self.event_feature_scale] #[nt]
            position = np.array(position) #[nsta, 2]
            position = np.stack([np.tile(t[:, None], (1, position.shape[0])), 
                               np.tile(position[:, 0], (len(t), 1)),
                               np.tile(position[:, 1], (len(t), 1))], axis=-1) #[nt, nsta, 3]
            
            prompt[0] = prompt[0] / self.nt # [0 - 1]
            prompt[1] = prompt[1] / 100 # scale by 100 km
            prompt[2] = prompt[2] / 100 # scale by 100 km
            position[:, :, 0] = position[:, :, 0] / self.nt # [0 - 1]
            position[:, :, 0] = position[:, :, 0] - prompt[0]
            prompt[0] = 0
            position[:, :, 1] = position[:, :, 1] / 100 # scale by 100 km
            position[:, :, 2] = position[:, :, 2] / 100 # scale by 100 km


            ## FIXME: DUMMY DATA
            data_new = data.copy()
            phase_pick_new = phase_pick.copy()
            phase_mask_new = phase_mask.copy()
            polarity_new = polarity.copy()
            polarity_mask_new = polarity_mask.copy()
            event_center_new = event_center.copy()
            event_time_new = event_time.copy()
            
            data += data_old
            phase_pick[1, :, :] = np.clip(phase_pick[1, :, :] + phase_pick_old[1, :, :], 0, 1)
            phase_pick[2, :, :] = np.clip(phase_pick[2, :, :] + phase_pick_old[2, :, :], 0, 1)
            phase_pick[0, :, :] = 1 - np.clip(phase_pick[1, :, :] + phase_pick[2, :, :], 0, 1)
            phase_mask = np.clip(phase_mask + phase_mask_old, 0, 1)
            polarity = (polarity - 0.5) + (polarity_old - 0.5) + 0.5
            polarity_mask = np.clip(polarity_mask + polarity_mask_old, 0, 1)
            event_center = np.clip(event_center + event_center_old, 0, 1)
            event_time += event_time_old
            event_mask = np.clip(event_mask + event_mask_old, 0, 1)

            data_old = data_new
            phase_pick_old = phase_pick_new
            phase_mask_old = phase_mask_new
            polarity_old = polarity_new
            polarity_mask_old = polarity_mask_new
            event_center_old = event_center_new
            event_time_old = event_time_new
            ###


            yield {
                "data": torch.from_numpy(data).float(),
                "phase_pick": torch.from_numpy(phase_pick).float(),
                "phase_mask": torch.from_numpy(phase_mask).float(),
                "polarity": torch.from_numpy(polarity[:, ::self.polarity_feature_scale]).float(),
                "polarity_mask": torch.from_numpy(polarity_mask[:, ::self.polarity_feature_scale]).float(),
                "event_center": torch.from_numpy(event_center[:, ::self.event_feature_scale]).float(),
                "event_time": torch.from_numpy(event_time[:, ::self.event_feature_scale]).float(),
                "event_mask": torch.from_numpy(event_mask[:, ::self.event_feature_scale]).float(),
                "station_location": torch.from_numpy(station_location).float(),
                "prompt_center": torch.from_numpy(prompt_center[:, ::self.event_feature_scale]).float(),
                "prompt_mask": torch.from_numpy(prompt_mask[:, ::self.event_feature_scale]).float(),
                "prompt": torch.tensor(prompt),
                "position": torch.tensor(position),
            }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SeismicNetworkIterableDataset("/global/home/users/zhuwq0/scratch/CEED/quakeflow_nc/waveform_test.h5")
    for i, x in enumerate(dataset):
        if i < 3:
            continue
        # print(x)
        fig, axes = plt.subplots(2, 4, figsize=(15, 5))
        for i in range(x["data"].shape[-1]):
            axes[0, 0].plot((x["data"][-1, :, i]) / torch.std(x["data"][-1, :, i]) / 10 + i)

            axes[0, 1].plot(x["phase_pick"][1, :, i] + i)
            axes[0, 1].plot(x["phase_pick"][2, :, i] + i)

            axes[0, 2].plot(x["polarity"][0, :, i] + i)
            axes[0, 2].plot(x["polarity_mask"][0, :, i] + i, linestyle="--", color="k")

            axes[0, 3].plot(x["event_center"][0, :, i] + i - 0.5)
            axes[0, 3].plot(x["event_mask"][0, :, i] + i - 0.5, linestyle="--", color="k")
            # axes[2].scatter(x["event_location"][0, :, i], x["event_location"][1, :, i])

            event_time = x["event_time"][0, :, i] * x["event_mask"][0, :, i]
            event_time = event_time / torch.max(event_time)
            axes[1, 0].plot(event_time + i)
            axes[1, 0].plot(x["event_mask"][0, :, i] + i - 0.5, linestyle="--", color="k")

            axes[1, 1].plot(x["position"][:, i, 0])
            axes[1, 1].axhline(x["prompt"][0], color="k", linestyle="--")
            axes[1, 2].scatter(x["position"][:, :, 1].flatten(), x["position"][:, :, 2].flatten(), marker="o", color="k")
            axes[1, 2].plot(x["prompt"][1], x["prompt"][2], marker="x", color="r")

            axes[1, 3].plot(x["prompt_center"][0, :, i] + i - 0.5)
            axes[1, 3].plot(x["prompt_mask"][0, :, i] + i - 0.5, linestyle="--", color="k")

        plt.savefig("test.png")
        plt.show()

        raise
