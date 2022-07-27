# %%
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
plt.rcParams['figure.facecolor'] = 'white'

# %%
h5_file = "ncedc.h5"
event_csv = pd.read_hdf(h5_file, "events")
phase_csv = pd.read_hdf(h5_file, "catalog")
phase_csv.set_index("event_index", inplace=True)

# %%
pre_window = 3000
post_window = 6000
sampling_rate = 100
# plt.figure()

with h5py.File(h5_file, "r") as fp_in:
    # with h5py.File(output_path.joinpath(f"{event['index']:06}.h5"), "w") as fp_out:
    with h5py.File("ncedc_seismic_dataset.h5", "w") as fp_out:

        for i, (_, event) in tqdm(enumerate(event_csv.iterrows()), total=len(event_csv)):

            event_id = event["index"]
            event_time = datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S.%f")
            event_latitude = event["latitude"]
            event_longitude = event["longitude"]
            event_depth_km = event["depth_km"]
            
            group = fp_out.create_group(f"{event_id:06d}")
            group.attrs["event_id"] = event_id
            group.attrs["event_time"] = event_time.isoformat(timespec='milliseconds')
            group.attrs["event_time_index"] = pre_window
            group.attrs["event_latitude"] = event_latitude
            group.attrs["event_longitude"] = event_longitude
            group.attrs["event_depth_km"] = event_depth_km

            for j, (_, phase) in enumerate(phase_csv.loc[[event["index"]]].iterrows()):

                trace = fp_in[f"data/{phase['fname']}"]
                p_time = datetime.fromisoformat(trace.attrs["p_time"])
                s_time = datetime.fromisoformat(trace.attrs["s_time"])
                p_arrival = (p_time - event_time).total_seconds()
                s_arrival = (s_time - event_time).total_seconds()
                p_arrival_index = int(p_arrival * sampling_rate) + pre_window
                s_arrival_index = int(s_arrival * sampling_rate) + pre_window

                begin_index = trace.attrs["p_idx"] - p_arrival_index
                end_index = begin_index + pre_window + post_window

                waveform = np.zeros([pre_window + post_window, 3])
                waveform[:trace[begin_index:end_index,:].shape[0], :] = trace[begin_index:end_index,:]
                network = trace.attrs["network"]
                station = trace.attrs["station"]
                location = trace.attrs["location_code"] if trace.attrs["location_code"] != "--" else ""
                channels = trace.attrs["channels"].split(",")
                
                begin_time = event_time - timedelta(seconds=pre_window/sampling_rate)
                end_time = event_time + timedelta(seconds=post_window/sampling_rate)
                distance_km = trace.attrs["distance_km"]
                azimuth = trace.attrs["azimuth"]
                first_motion = trace.attrs["first_motion"]
                emergence_angle = trace.attrs["emergence_angle"]
                station_latitude = trace.attrs["station_latitude"]
                station_longitude = trace.attrs["station_longitude"]
                station_elevation_m = trace.attrs["station_elevation_m"]
                dt_s = trace.attrs["dt"]
                unit = trace.attrs["unit"]
                snr = trace.attrs["snr"]
                phase_type = ["P", "S"]
                phase_index = [p_arrival_index, s_arrival_index]
                phase_time = [p_time.isoformat(timespec='milliseconds'), s_time.isoformat(timespec='milliseconds')]
                phase_score = [phase["p_weight"], phase["s_weight"]]
                phase_remark = [phase["p_remark"], phase["s_remark"]]
                assert(dt_s == 1.0/sampling_rate)
                
                station_id = f"{network}.{station}.{location}.{channels[0][:2]}"
                fp_out[f"{event_id:06d}/{station_id}"] = waveform * 1e6
                attrs = fp_out[f"{event_id:06d}/{station_id}"].attrs
                attrs["network"] = network
                attrs["station"] = station
                attrs["location"] = location
                attrs["channels"] = channels
                attrs["begin_time"] = begin_time.isoformat(timespec='milliseconds')
                attrs["end_time"] = end_time.isoformat(timespec='milliseconds')
                attrs["distance_km"] = distance_km
                attrs["azimuth"] = azimuth
                attrs["first_motion"] = first_motion
                attrs["emergence_angle"] = emergence_angle
                attrs["station_latitude"] = station_latitude
                attrs["station_longitude"] = station_longitude
                attrs["station_elevation_m"] = station_elevation_m
                attrs["dt_s"] = dt_s
                attrs["unit"] = "u"+unit
                attrs["snr"] = snr
                attrs["phase_type"] = phase_type
                attrs["phase_index"] = phase_index
                attrs["phase_time"] = phase_time
                attrs["phase_score"] = phase_score
                attrs["phase_remark"] = phase_remark

                # print(begin_time, end_time)
                # plt.plot(trace[begin_index:end_index,-1]/np.std(trace[begin_index:end_index,-1])/10 + j)
                # plt.plot([(p_time - begin_time).total_seconds()*sampling_rate, (p_time - begin_time).total_seconds()*sampling_rate], [j-2, j+2], '--r')
                # plt.plot([(s_time - begin_time).total_seconds()*sampling_rate, (s_time - begin_time).total_seconds()*sampling_rate], [j-2, j+2], '--b')

            # if i > 100:
            #     break

# plt.show()

# %%



