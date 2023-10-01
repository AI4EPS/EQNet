import os
import random
from glob import glob

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import obspy
import fsspec
import json
import logging
import torch
from torch.utils.data import Dataset, IterableDataset
from collections import defaultdict
from datetime import timedelta, datetime
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import butter, filtfilt

from ..utils.station_sampler import reorder_keys
from .utils import random_shift


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
    feature_scale = int(nt / feature_nt)

    def __init__(self, 
        data_path=None,
        data_list=None,
        hdf5_file=None,
        station_file=None, # a json file
        prefix="",
        format="h5",
        dataset="seismic_network",
        training=True,
        phases=["P", "S"],
        sort=False,
        num_stations=None,
        ## for prediction
        sampling_rate=100,
        response_xml=None,
        highpass_filter=False,
        rank=0,
        world_size=1,
        cut_patch=False,
        nt_cut=1024 * 8,
        nx=30,
        min_nt=1024,
        min_nx=1,
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
                self.data_list = list(fp.keys())
                with open(tmp_hdf5_keys, "w") as f:
                    for x in self.data_list:
                        f.write(x + "\n")
            else:
                self.data_list = pd.read_csv(tmp_hdf5_keys, header=None, names=["event_id"])["event_id"].values.tolist()
        elif data_list is not None:
            with open(data_list, "r") as f:
                self.data_list = f.read().splitlines()
        elif data_path is not None:
            if "hour" in format:
                days = glob(f"{data_path}/**/", recursive=False)
                hours = []
                for day in sorted(days):
                    hours += glob(f"{day}**/", recursive=False)
                self.data_list = [
                    f"{x}{prefix}*.{format.split('_')[0]}" for x in sorted(list(hours))
                ]
            else:
                self.data_list = [
                    x for x in sorted(list(glob(os.path.join(data_path, f"{prefix}*.{format}"))))
                ]
        else:
            self.data_list = None
        if self.data_list is not None:
            self.data_list = self.data_list[rank::world_size]
        
        if station_file is not None:
            self.station_info = json.load(open(station_file, "r"))
            
        self.data_path = data_path
        self.hdf5_file = hdf5_file
        self.phases = phases
        self.response_xml = response_xml
        self.sampling_rate = sampling_rate
        self.highpass_filter = highpass_filter
        self.format = format
        self.dataset = dataset
        self.sort = sort
        self.num_stations = num_stations

        ## training
        self.training = training

        ## prediction
        self.cut_patch = cut_patch
        self.nt_cut = nt_cut
        self.nx = nx
        self.min_nt = min_nt
        self.min_nx = min_nx
        
        if self.training:
            print(f"{self.data_path}: {len(self.data_list)} files")
        else:
            print(
                os.path.join(data_path, f".{format}"),
                f": {len(self.data_list)} files",
            )

        self._data_len = self._count()

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
            return iter(self.sample_test(data_list))

    def _count(self):
        if not self.cut_patch:
            return len(self.data_list)
        else:
            if self.format == "h5":
                with fsspec.open(self.data_list[0], "rb") as fs:
                    with h5py.File(fs, "r") as fp:
                        nx, nt = fp["data"].shape
            elif "mseed" in self.format:
                nx = len(self.station_info)
                for fname in self.data_list:
                    try:
                        stream = obspy.read(fname)
                        break
                    except:
                        pass
                begin_time = min([st.stats.starttime for st in stream])
                end_time = max([st.stats.endtime for st in stream])
                nt = int((end_time - begin_time) * self.sampling_rate)
            return len(self.data_list) * ((nt - 1) // self.nt_cut + 1) * ((nx - 1) // self.nx + 1)
            # return len(self.data_list) * ((nt - 1) // self.nt_cut + 1)
    
    def __len__(self):
        return self._data_len
    
    
    def _read_training_h5(self, event_id):
        num_stations = self.num_stations
        hdf5_fp = self.hdf5_fp
        event = hdf5_fp[event_id]
        station_ids = list(event.keys())
        if num_stations is not None and len(station_ids) < num_stations:
            return None
        else:
            station_ids = np.random.choice(station_ids, num_stations, replace=False)
        event_attrs = event.attrs

        # avoid stations with P arrival equals S arrival
        is_sick = False
        for sta_id in station_ids:
            attrs = event[sta_id].attrs
            p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
            s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
            if p_picks>=s_picks:
                is_sick = True
                break
            if ((p_picks) + (s_picks))[0] > self.nt * 2:
                is_sick = True
                break
        if is_sick:
            return None
        
        reference_latitude = 0
        reference_longitude = 0
        for sta_id in station_ids:
            reference_latitude += event[sta_id].attrs["latitude"]
            reference_longitude += event[sta_id].attrs["longitude"]
        reference_latitude/=len(station_ids)
        reference_longitude/=len(station_ids)
        
        b, a = butter(4, 0.1, btype="highpass", analog=False)
        waveforms = np.zeros([len(station_ids), 3, self.nt], dtype="float32")
        amplitude = np.zeros_like(waveforms)
        phase_pick = np.zeros_like(waveforms)
        event_center = np.zeros([len(station_ids), self.feature_nt])
        event_location = np.zeros([len(station_ids), 7, self.feature_nt])
        event_location_mask = np.zeros([len(station_ids), self.feature_nt])
        station_location = np.zeros([len(station_ids), 3])
        # reference_point = np.array([reference_longitude, reference_latitude])

        for i, sta_id in enumerate(station_ids):
            # trace_id = event_id + "/" + sta_id
            waveforms[i, :, :] = event[sta_id][:, :self.nt]
            amplitude[i, :, :] = event[sta_id][:, :self.nt]
            attrs = event[sta_id].attrs
            if attrs["unit"][-6:] == "m/s**2":
                # integrate acceleration to velocity
                amplitude[:, :, i] = np.cumsum(amplitude[:, :, i]*attrs["dt_s"], axis=1)
                for j in range(3): 
                    spline_i = LSQUnivariateSpline(np.arange(amplitude.shape[1]), amplitude[j, :, i], t=np.arange(amplitude.shape[1], step=amplitude.shape[1]/2048)[1:], k=3)
                    amplitude[j, :, i] -= spline_i(np.arange(amplitude.shape[1]))
                amplitude[:, :, i] = filtfilt(b, a, amplitude[:, :, i], axis=1)
            elif attrs["unit"][-3:] == "m/s": #TODO: temp
                amplitude[:, :, i] = amplitude[:, :, i] * 10e4
            
            p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
            s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
            phase_pick[i, :, :] = generate_label([p_picks, s_picks], nt=self.nt)

            ## TODO: how to deal with multiple phases
            # center = (attrs["phase_index"][::2] + attrs["phase_index"][1::2])/2.0
            ## assuming only one event with both P and S picks
            c0 = ((p_picks) + (s_picks)) / 2.0 # phase center
            c0_width = max(((s_picks - p_picks) * self.sampling_rate / 200.0).max(), 80)
            # c0_width = ((s_picks - p_picks) * self.sampling_rate / 200.0).max() # min=160
            assert c0_width>0
            dx = round(
                (event_attrs["longitude"] - attrs["longitude"])
                * np.cos(np.radians(reference_latitude))
                * self.degree2km,
                2,
            )
            dy = round(
                (event_attrs["latitude"] - attrs["latitude"])
                * self.degree2km,
                2,
            )
            dz = round(
                event_attrs["depth_km"] + attrs["elevation_m"] / 1e3,
                2,
            )

            assert c0[0]<self.nt
            c0 = c0/self.feature_scale
            assert c0[0]<self.feature_nt
            c0_width = c0_width/self.feature_scale
            #assert c0_width>=160/self.feature_scale
            c0_int = c0.astype(np.int32)
            assert c0_int[0]<self.feature_nt
            assert abs(c0-c0_int)[0]<1
            
            event_center[i, :] = generate_label(
                [
                    # [c0 / self.feature_scale],
                    c0_int,
                ],
                label_width=[
                    c0_width,
                ],
                # label_width=[
                #     10,
                # ],
                nt=self.feature_nt,
                # nt=self.nt,
            )[1, :]
            mask = event_center[i, :] >= 0.5
            event_location[i, 0, :] = (
                self.feature_scale * np.arange(self.feature_nt) - event_attrs["event_time_index"]
            ) / self.sampling_rate
            # event_location[0, :, i] = (np.arange(self.feature_nt) - 3000 / self.feature_scale) / self.sampling_rate
            # print(event_location[i, 1:, mask].shape, event_location.shape, event_location[i][1:, mask].shape)
            event_location[i][1:, mask] = np.array([dx, dy, dz, (c0-c0_int)[0], c0_width])[:, np.newaxis]
            event_location_mask[i, :] = mask

            ## station location
            station_location[i, 0] = round(
                (attrs["longitude"] - reference_longitude)
                * np.cos(np.radians(reference_latitude))
                * self.degree2km,
                2,
            )
            station_location[i, 1] = round((attrs["latitude"] - reference_latitude)
                                           * self.degree2km, 2)
            station_location[i, 2] =  round(-attrs["elevation_m"]/1e3, 2)

        # std = np.std(waveforms, axis=-1, keepdims=True)
        # std[std == 0] = 1.0
        # waveforms = (waveforms - np.mean(waveforms, axis=-1, keepdims=True)) / std
        # waveforms = waveforms.astype(np.float32)

        return {
            "data": torch.from_numpy(waveforms).float(),
            "amplitude": torch.from_numpy(amplitude).float(),
            "phase_pick": torch.from_numpy(phase_pick).float(),
            "event_center": torch.from_numpy(event_center).float(),
            "event_location": torch.from_numpy(event_location).float(),
            "event_location_mask": torch.from_numpy(event_location_mask).float(),
            "station_location": torch.from_numpy(station_location).float(),
            # "reference_point": torch.from_numpy(reference_point).float(),
        }


    def sample_train(self, data_list):
        while True:
            event_id = np.random.choice(data_list)
            # if True:
            try:
                meta = self._read_training_h5(event_id)
            except Exception as e:
                print(f"Error reading {event_id}:\n{e}")
                continue

            if meta is None:
                continue

            if self.sort:
                D = np.sqrt(((meta["station_location"][:, np.newaxis, :2] -  meta["station_location"][np.newaxis, :, :2])**2).sum(axis=-1))
                Tcsr = minimum_spanning_tree(D)
                index = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                
                for k in meta.keys():
                    meta[k] = meta[k][index]
            
            meta = reorder_keys(meta)
            meta = random_shift(meta, shift_range=(-160, 0), feature_scale=self.feature_scale)
            
            yield meta
            
    
    def read_mseed(self, fname, response_xml=None, highpass_filter=False, sampling_rate=100):
        try:
            stream = obspy.read(fname)
            stream = stream.merge(fill_value="latest")
            if response_xml is not None:
                response = obspy.read_inventory(response_xml)
                stream = stream.remove_sensitivity(response)
        except Exception as e:
            print(f"Error reading {fname}:\n{e}")
            assert 0, f"Error reading {fname}:\n{e}"
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

        station_keys = sorted(list(station_ids.keys()))
        station_location = np.zeros([len(station_keys), 3], dtype=np.float32)
        for i, sta in enumerate(station_keys):
            station_location[i, 0] = self.station_info[sta]["longitude"]
            station_location[i, 1] = self.station_info[sta]["latitude"]
            station_location[i, 2] = round(-self.station_info[sta]["elevation_m"]/1e3, 2)
        reference_latitude = np.mean(station_location[:, 1])
        station_location[:, 0] = np.round(
                (station_location[:, 0])
                * np.cos(np.radians(reference_latitude))
                * self.degree2km,
                2,
            )
        station_location[:, 1] = np.round(station_location[:, 1] * self.degree2km, 2)
        nx = len(station_ids)
        nt = len(stream[0].data)
        data = np.zeros([3, nt+((32- nt%32)%32), nx], dtype=np.float32) # the length of data should be multiple of 32
        amplitude = np.zeros_like(data)
        for i, sta in enumerate(station_keys):
            for c in station_ids[sta]:
                j = comp2idx[c]

                if len(stream.select(id=sta + c)) == 0:
                    print(f"Empty trace: {sta+c} {begin_time}")
                    continue

                trace = stream.select(id=sta + c)[0]

                tmp = trace.data.astype("float32")
                ## accerleration to velocity
                if sta[-1] == "N":
                    amp = trace.integrate().filter("highpass", freq=1.0).data.astype("float32")
                else:
                    amp = tmp
                data[j, : len(tmp), i] = tmp[:nt]
                amplitude[j, : len(tmp), i] = amp[:nt]
        
        if self.sort and len(station_keys) > 5:
            D = np.sqrt(((station_location[:, np.newaxis, :2] -  station_location[np.newaxis, :, :2])**2).sum(axis=-1))
            Tcsr = minimum_spanning_tree(D)
            index = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
            
            data = data[:, :, index]
            amplitude = amplitude[:, :, index]
            station_keys = [station_keys[i] for i in index]
            station_location = station_location[index]

        return {
            "data": torch.from_numpy(data),
            "amplitude": torch.from_numpy(amplitude),
            "station_id": station_keys,
            "station_location": torch.from_numpy(station_location).float(),
            "begin_time": begin_time.datetime.isoformat(timespec="milliseconds"),
            "dt_s": 1 / sampling_rate,
        }
            
    def read_hdf5(self, event_id):
        if self.hdf5_fp is None:
            raise ("HDF5 file is not opened")
        else:
            hdf5_fp = self.hdf5_fp
            event = hdf5_fp[event_id]
            station_ids = list(event.keys())
            event_attrs = event.attrs
            begin_time = event_attrs["begin_time"]
            dt_s = event[station_ids[0]].attrs["dt_s"]
            # data shape
            nt = event[station_ids[0]].shape[1]
            b, a = butter(4, 0.1, btype="highpass", analog=False)
            waveforms = np.zeros([3, nt+((32-(nt%32))%32), len(station_ids)], dtype="float32")
            amplitude = np.zeros_like(waveforms)
            station_location = np.zeros([len(station_ids), 3])
            reference_latitude = 0
            for sta_id in station_ids:
                reference_latitude += event[sta_id].attrs["latitude"]
            reference_latitude/=len(station_ids)
            for i, sta_id in enumerate(station_ids):
                # trace_id = event_id + "/" + sta_id
                waveforms[:, :nt, i] = event[sta_id][:, :]
                amplitude[:, :nt, i] = event[sta_id][:, :]
                attrs = event[sta_id].attrs
                if attrs["unit"][-6:] == "m/s**2":
                    # integrate acceleration to velocity
                    amplitude[:, :, i] = np.cumsum(amplitude[:, :, i]*attrs["dt_s"], axis=1)
                    for j in range(3): 
                        spline_i = LSQUnivariateSpline(np.arange(amplitude.shape[1]), amplitude[j, :, i], t=np.arange(amplitude.shape[1], step=amplitude.shape[1]/2048)[1:], k=3)
                        amplitude[j, :, i] -= spline_i(np.arange(amplitude.shape[1]))
                    amplitude[:, :, i] = filtfilt(b, a, amplitude[:, :, i], axis=1)
                elif attrs["unit"][-3:] == "m/s": #TODO: temp
                    amplitude[:, :, i] = amplitude[:, :, i] * 10e4

                ## station location
                station_location[i, 0] = round(
                    (attrs["longitude"])
                    * np.cos(np.radians(reference_latitude))
                    * self.degree2km,
                    2,
                )
                station_location[i, 1] = round((attrs["latitude"])
                                               * self.degree2km, 2)
                station_location[i, 2] =  round(-attrs["elevation_m"]/1e3, 2)

            # std = np.std(waveforms, axis=1, keepdims=True)
            # std[std == 0] = 1.0
            # waveforms = (waveforms - np.mean(waveforms, axis=1, keepdims=True)) / std
            # waveforms = waveforms.astype(np.float32)
            
            if self.sort and len(station_ids)>5:
                D = np.sqrt(((station_location[:, np.newaxis, :2] -  station_location[np.newaxis, :, :2])**2).sum(axis=-1))
                Tcsr = minimum_spanning_tree(D)
                index = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                
                waveforms = waveforms[:, :, index]
                amplitude = amplitude[:, :, index]
                station_ids = [station_ids[i] for i in index]
                station_location = station_location[index]

        return {
            "data": torch.from_numpy(waveforms).float(),
            "amplitude": torch.from_numpy(amplitude).float(),
            "station_id": station_ids,
            "station_location": torch.from_numpy(station_location).float(),
            "begin_time": begin_time,
            "dt_s": dt_s,
        }
        
    def read_das_hdf5(self, fname):
        meta = {}
        with fsspec.open(fname, "rb") as fs:
            try:
                with h5py.File(fs, "r") as fp:
                    # raw_data = fp["data"][()]  # [nt, nx]
                    raw_data = fp["data"][:, :].T  # (nx, nt) -> (nt, nx)
                    raw_data = raw_data - np.mean(raw_data, axis=0, keepdims=True)
                    raw_data = raw_data - np.median(raw_data, axis=1, keepdims=True)
                    std = np.std(raw_data, axis=0, keepdims=True)
                    std[std == 0] = 1.0
                    raw_data = raw_data / std
                    attrs = fp["data"].attrs
                    nt, nx = raw_data.shape
                    data = np.zeros([3, nt, nx], dtype=np.float32)
                    data[-1, :, :] = raw_data[:, :]
                    meta["waveform"] = torch.from_numpy(data)
                    if "station_id" in attrs:
                        station_id = attrs["station_name"]
                    else:
                        station_id = [f"{i}" for i in range(nx)]
                    meta["station_location"] = torch.from_numpy(
                        np.zeros([nx, 3], dtype=np.float32)
                    )
                    meta["station_location"][:, 0] = torch.arange(nx)/1000 # TODO: get real location
                    meta["station_id"] = station_id
                    meta["begin_time"] = attrs["begin_time"]
                    meta["dt_s"] = attrs["dt_s"]
            except Exception as e:
                print(f"Error reading {fname}:\n{e}")
                return None
        return meta
    
    def sample_test(self, data_list):
        for fname in data_list:
            if "mseed" in self.format:
                meta = self.read_mseed(
                    fname,
                    response_xml=self.response_xml,
                    highpass_filter=self.highpass_filter,
                    sampling_rate=self.sampling_rate,
                )
            elif (self.format == "h5") and (self.dataset == "seismic_network"):
                meta = self.read_hdf5(fname)
            elif (self.format == "h5") and (self.dataset == "das"):
                meta = self.read_das_hdf5(fname)
            else:
                raise NotImplementedError
            if meta is None:
                continue
            if self.format == "h5":
                meta["file_name"] = fname
            elif "mseed" in self.format:
                if "hour" in self.format:
                    meta["file_name"] = fname.split("/")[-3]+"/"+fname.split("/")[-2]
                else:
                    meta["file_name"] = os.path.basename(fname).split(".")[0]

            if not self.cut_patch:
                yield meta
            else:
                _, nt, nx = meta["data"].shape
                for i in list(range(0, nt, self.nt_cut)):
                    for j in list(range(0, nx, self.nx)):
                        assert meta["data"][:, i : i + self.nt_cut, j : j + self.nx].shape[1] > 256, f"Error: {fname} {i} {j}, too short {meta['data'][:, i : i + self.nt_cut, j : j + self.nx].shape[1]}, please check the data or adjust cut_patch parameters"
                        yield {
                            "data": meta["data"][:, i : i + self.nt_cut, j : j + self.nx],
                            "amplitude": meta["amplitude"][:, i : i + self.nt_cut, j : j + self.nx],
                            "station_location": meta["station_location"][j : j + self.nx, :], # [nx, 3]
                            "station_id": meta["station_id"][j : j + self.nx],
                            "begin_time": (
                                datetime.fromisoformat(meta["begin_time"]) + timedelta(seconds=i * meta["dt_s"])
                            ).isoformat(timespec="milliseconds"),
                            "begin_time_index": i,
                            "dt_s": meta["dt_s"],
                            "file_name": meta["file_name"] + f"_{i:04d}_{j:04d}",
                        }



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SeismicNetworkIterableDataset("/Users/weiqiang/Research/EQNet/datasets/NCEDC/ncedc_event.h5")
    for x in dataset:
        # print(x)
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))
        for i in range(x["data"].shape[-1]):
            axes[0].plot((x["data"][-1, :, i]) / torch.std(x["data"][-1, :, i]) / 10 + i)
            axes[1].plot(x["amplitude"][-1, :, i] / torch.std(x["amplitude"][-1, :, i]) / 10 + i)

            axes[2].plot(x["phase_pick"][1, :, i] + i)
            axes[2].plot(x["phase_pick"][2, :, i] + i)

            axes[3].plot(x["event_center"][:, i] + i - 0.5)
            # axes[2].scatter(x["event_location"][0, :, i], x["event_location"][1, :, i])

            axes[4].plot(x["event_location"][0, :, i] / 10 + i)

            t = np.arange(x["event_location"].shape[1])[x["event_location_mask"][:, i] == 1]
            axes[5].plot(t, x["event_location"][1, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")
            axes[5].plot(t, x["event_location"][2, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")
            axes[5].plot(t, x["event_location"][3, x["event_location_mask"][:, i] == 1, i] / 10 + i, color=f"C{i}")

        plt.savefig("test.png")
        plt.show()

        raise
