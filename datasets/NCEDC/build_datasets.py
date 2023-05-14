# %%
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import multiprocessing as mp

plt.rcParams["figure.facecolor"] = "white"


def calc_snr(waveform, picks, noise_window=300, signal_window=300, gap_window=50):

    noises = []
    signals = []
    snr = []
    for i in range(waveform.shape[0]):
        for j in picks:
            if j + gap_window < waveform.shape[1]:
                noise = np.std(waveform[i, j - noise_window : j - gap_window])
                signal = np.std(waveform[i, j + gap_window : j + signal_window])
                # noise = np.max(np.abs(waveform[i, j - noise_window : j - gap_window]))
                # signal = np.max(np.abs(waveform[i, j + gap_window : j + signal_window]))
                if (noise > 0) and (signal > 0):
                    signals.append(signal)
                    noises.append(noise)
                    snr.append(signal / noise)
                else:
                    signals.append(0)
                    noises.append(0)
                    snr.append(0)

    return snr


# %%
h5_in = "ncedc.h5"
h5_out = "ncedc_event_dataset_polarity_2.h5"
event_csv = pd.read_hdf(h5_in, "events")
event_csv["time_id"] = event_csv["time"].apply(
    lambda x: x[0:4] + x[5:7] + x[8:10] + x[11:13] + x[14:16] + x[17:19] + x[20:22]
)
phase_csv = pd.read_hdf(h5_in, "catalog")
phase_csv.set_index("event_index", inplace=True)
with open("event_id.json", "r") as f:
    time_to_event_id = json.load(f)

# %%
polarity = pd.read_csv("picks_polarity.csv")
polarity.set_index("event_id", inplace=True)

# %%
pre_window = 3000
post_window = 9000
sampling_rate = 100
# plt.figure()

with h5py.File(h5_in, "r") as fp_in:
    # with h5py.File(output_path.joinpath(f"{event['index']:06}.h5"), "w") as fp_out:
    with h5py.File(h5_out, "w") as fp_out:

        for i, (_, event) in tqdm(enumerate(event_csv.iterrows()), total=len(event_csv)):

            first_p_arrival = datetime.fromisoformat(phase_csv.loc[[event["index"]]]["p_time"].min())
            anchor_time = first_p_arrival  ## put anchor at 30s of the window
            # anchor_time = event_time

            # event_id = event["index"]
            event_id = "nc" + time_to_event_id[event["time_id"]]
            event_time = datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S.%f")
            event_time_index = int((event_time - anchor_time).total_seconds() * sampling_rate) + pre_window
            event_latitude = event["latitude"]
            event_longitude = event["longitude"]
            event_depth_km = event["depth_km"]

            if f"{event_id}" in fp_out:
                print(f"{event_id} already exists!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

            group = fp_out.create_group(f"{event_id}")
            group.attrs["event_id"] = event_id
            group.attrs["event_time"] = event_time.isoformat(timespec="milliseconds")
            group.attrs["event_time_index"] = event_time_index
            group.attrs["begin_time"] = (anchor_time - timedelta(seconds=pre_window / sampling_rate, )).isoformat(
                timespec="milliseconds"
            )
            group.attrs["end_time"] = (anchor_time + timedelta(seconds=post_window / sampling_rate)).isoformat(
                timespec="milliseconds"
            )
            # group.attrs["time_reference"] = anchor_time.isoformat(timespec="milliseconds")
            # group.attrs["time_before"] = pre_window/sampling_rate
            # group.attrs["time_after"] = post_window/sampling_rate
            group.attrs["latitude"] = event_latitude
            group.attrs["longitude"] = event_longitude
            group.attrs["depth_km"] = event_depth_km
            group.attrs["magnitude"] = event["magnitude"]
            group.attrs["magnitude_type"] = event["magnitude_type"]
            group.attrs["num_stations"] = len(phase_csv.loc[[event["index"]]])

            ##
            has_polarity = False
            if event_id in polarity.index:
                has_polarity = True
                group.attrs["strike"] = list(set(polarity.loc[event_id]["strike"].values))[0]
                group.attrs["dip"] = list(set(polarity.loc[event_id]["dip"].values))[0]
                group.attrs["rake"] = list(set(polarity.loc[event_id]["rake"].values))[0]

            for j, (_, phase) in enumerate(phase_csv.loc[[event["index"]]].iterrows()):

                trace = fp_in[f"data/{phase['fname']}"]

                SNR = calc_snr(trace[:].T, [6000])  ## P arrivals are at 6000
                SNR = np.array(SNR)
                # if not ((len(SNR) >= 3) and (np.all(SNR) > 0) and (np.max(SNR) > 2.0)):
                # if not (np.all(SNR) > 0):
                #     continue

                p_time = datetime.fromisoformat(trace.attrs["p_time"])
                s_time = datetime.fromisoformat(trace.attrs["s_time"])
                p_arrival_index = int((p_time - anchor_time).total_seconds() * sampling_rate) + pre_window
                s_arrival_index = int((s_time - anchor_time).total_seconds() * sampling_rate) + pre_window

                begin_index = trace.attrs["p_idx"] - p_arrival_index
                end_index = begin_index + pre_window + post_window
                if begin_index < 0:
                    print(trace.attrs["p_idx"], p_arrival_index, begin_index, end_index)
                # if trace[begin_index:end_index,:].shape != (9000, 3):
                #     print(trace.shape, p_arrival_index, begin_index, end_index)

                waveform = np.zeros([pre_window + post_window, 3], dtype=np.float32)
                waveform[: trace[max(0, begin_index) : max(0, end_index), :].shape[0], :] = (
                    trace[max(0, begin_index) : max(0, end_index), :] * 1e6
                )
                network = trace.attrs["network"]
                station = trace.attrs["station"]
                location = trace.attrs["location_code"] if trace.attrs["location_code"] != "--" else ""
                channels = trace.attrs["channels"].split(",")

                begin_time = anchor_time - timedelta(seconds=pre_window / sampling_rate)
                end_time = anchor_time + timedelta(seconds=post_window / sampling_rate)
                distance_km = trace.attrs["distance_km"]
                azimuth = trace.attrs["azimuth"]
                p_polarity = trace.attrs["first_motion"]
                if p_polarity not in ["U", "D"]:
                    p_polarity = "N"
                s_polarity = "N"
                if has_polarity:
                    station_id = f"{network}.{station}.{location}.{channels[0][:-1]}"
                    station_ids = polarity.loc[event_id]["station_id"].values
                    if station_id in station_ids:
                        radiation = polarity.loc[event_id].loc[station_ids == station_id]
                        p_radiation = radiation[radiation["phase_type"] == "P"]["radiation"].values[0]
                        s_radiation = radiation[radiation["phase_type"] == "S"]["radiation"].values[0]
                        p_radiation = round(p_radiation, 3)
                        s_radiation = round(s_radiation, 3)
                        if ((np.sign(p_radiation) > 0) and (p_polarity == "D")) or ((np.sign(p_radiation) < 0) and (p_polarity == "U")):
                            print("P polarity is wrong", p_radiation, p_polarity, event_id, station_id)
                            p_polarity = "N"
                            s_polarity = "N"
                        else:
                            if np.sign(p_radiation) > 0:
                                p_polarity = "U"
                            elif np.sign(p_radiation) < 0:
                                p_polarity = "D"
                            else:
                                print("P radiation is 0", p_radiation, p_polarity, event_id, station_id)
                            if np.sign(s_radiation) > 0:
                                s_polarity = "U"
                            elif np.sign(s_radiation) < 0:
                                s_polarity = "D"
                            else:
                                print("S radiation is 0", s_radiation, s_polarity, event_id, station_id)
                    
                emergence_angle = trace.attrs["emergence_angle"]
                station_latitude = trace.attrs["station_latitude"]
                station_longitude = trace.attrs["station_longitude"]
                station_elevation_m = trace.attrs["station_elevation_m"]
                dt_s = trace.attrs["dt"]
                unit = trace.attrs["unit"]
                snr = trace.attrs["snr"]
                phase_type = ["P", "S"]
                phase_index = [p_arrival_index, s_arrival_index]
                phase_time = [p_time.isoformat(timespec="milliseconds"), s_time.isoformat(timespec="milliseconds")]
                phase_score = [phase["p_weight"], phase["s_weight"]]
                phase_remark = [phase["p_remark"], phase["s_remark"]]
                phase_polarity = [p_polarity, s_polarity]
                event_ids = [event_id, event_id]
                assert dt_s == 1.0 / sampling_rate

                station_id = f"{network}.{station}.{location}.{channels[0][:-1]}"
                fp_out[f"{event_id}/{station_id}"] = waveform * 1e6
                attrs = fp_out[f"{event_id}/{station_id}"].attrs
                attrs["network"] = network
                attrs["station"] = station
                attrs["location"] = location
                attrs["component"] = [x[-1] for x in channels]
                attrs["distance_km"] = distance_km
                attrs["azimuth"] = azimuth
                attrs["emergence_angle"] = emergence_angle
                attrs["latitude"] = station_latitude
                attrs["longitude"] = station_longitude
                attrs["elevation_m"] = station_elevation_m
                attrs["dt_s"] = dt_s
                attrs["unit"] = "1e-6" + unit
                attrs["snr"] = SNR
                attrs["phase_type"] = phase_type
                attrs["phase_index"] = phase_index
                attrs["phase_time"] = phase_time
                attrs["phase_score"] = phase_score
                attrs["phase_remark"] = phase_remark
                attrs["phase_polarity"] = phase_polarity
                attrs["event_id"] = event_ids

                # print(begin_time, end_time)
                # plt.plot(trace[begin_index:end_index,-1]/np.std(trace[begin_index:end_index,-1])/10 + j)
                # plt.plot([(p_time - begin_time).total_seconds()*sampling_rate, (p_time - begin_time).total_seconds()*sampling_rate], [j-2, j+2], '--r')
                # plt.plot([(s_time - begin_time).total_seconds()*sampling_rate, (s_time - begin_time).total_seconds()*sampling_rate], [j-2, j+2], '--b')

            # if i > 2000:
            #     break

# plt.show()

# %%
