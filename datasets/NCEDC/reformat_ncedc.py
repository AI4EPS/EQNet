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

# %%
h5_in = "ncedc.h5"
h5_out = "ncedc_renamed.h5"
event_csv = pd.read_hdf(h5_in, "events")
event_csv["time_id"] = event_csv["time"].apply(
    lambda x: x[0:4] + x[5:7] + x[8:10] + x[11:13] + x[14:16] + x[17:19] + x[20:22]
)
phase_csv = pd.read_hdf(h5_in, "catalog")
phase_csv.set_index("event_index", inplace=True)
with open("event_id.json", "r") as f:
    time_to_event_id = json.load(f)

#%%
with h5py.File(h5_in, "r") as fp_in:
    # with h5py.File(output_path.joinpath(f"{event['index']:06}.h5"), "w") as fp_out:
    with h5py.File(h5_out, "a") as fp_out:

        for i, (_, event) in tqdm(enumerate(event_csv.iterrows()), total=len(event_csv)):

            event_id = "nc" + time_to_event_id[event["time_id"]]

            for j, (_, phase) in enumerate(phase_csv.loc[[event["index"]]].iterrows()):

                # print(f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}")
                if f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}" in fp_out:
                    continue

                trace_in = fp_in[f"data/{phase['fname']}"]
                trace_out = fp_out.create_dataset(
                    f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}", data=trace_in[()]
                )

                for k in trace_in.attrs:
                    if k == "event_index":
                        trace_out.attrs["event_id"] = event_id
                    elif k == "event_time":
                        trace_out.attrs["event_time"] = datetime.strptime(
                            trace_in.attrs["event_time"], "%Y-%m-%dT%H:%M:%S.%f"
                        ).isoformat(timespec="milliseconds")
                    elif k == "location_code":
                        if trace_in.attrs["location_code"] == "--":
                            trace_out.attrs["location_code"] = ""
                        else:
                            trace_out.attrs["location_code"] = trace_in.attrs["location_code"]
                    elif k == "snr":
                        trace_out.attrs["snr"] = "[" + ", ".join([f"{x:.1f}" for x in trace_in.attrs["snr"]]) + "]"
                    elif k == "first_motion":
                        first_motion = trace_in.attrs["first_motion"]
                        if first_motion not in ["U", "D"]:
                            first_motion = "N"
                        trace_out.attrs["p_polarity"] = first_motion
                    else:
                        trace_out.attrs[k] = trace_in.attrs[k]

                # print(dict(trace_out.attrs))
                # fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].attrs = trace.attrs

                # print(fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].shape)
                # print(dict(fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].attrs))

                # print(event_id, phase['fname'])
                # print(dict(trace.attrs))

                # plt.figure()
                # plt.plot(fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"][:,2])
                # plt.show()

            # raise
# %%
