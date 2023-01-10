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
h5_file = "ncedc.h5"
event_csv = pd.read_hdf(h5_file, "events")
event_csv["time_id"] = event_csv["time"].apply(
    lambda x: x[0:4] + x[5:7] + x[8:10] + x[11:13] + x[14:16] + x[17:19] + x[20:22]
)
phase_csv = pd.read_hdf(h5_file, "catalog")
phase_csv.set_index("event_index", inplace=True)
with open("event_id.json", "r") as f:
    time_to_event_id = json.load(f)

#%%
with h5py.File(h5_file, "r") as fp_in:
    # with h5py.File(output_path.joinpath(f"{event['index']:06}.h5"), "w") as fp_out:
    with h5py.File("ncedc_renamed.h5", "w") as fp_out:

        for i, (_, event) in tqdm(enumerate(event_csv.iterrows()), total=len(event_csv)):

            event_id = "nc" + time_to_event_id[event["time_id"]]

            for j, (_, phase) in enumerate(phase_csv.loc[[event["index"]]].iterrows()):

                trace = fp_in[f"data/{phase['fname']}"]

                # print(f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}")
                fp_out.create_dataset(f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}", data=trace[()])

                for k in trace.attrs:
                    if k == "event_index":
                        fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].attrs["event_id"] = event_id
                    elif k == "event_time":
                        fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].attrs[
                            "event_time"
                        ] = datetime.strptime(trace.attrs["event_time"], "%Y-%m-%dT%H:%M:%S.%f").isoformat(
                            timespec="milliseconds"
                        )
                    elif k == "location_code":
                        if trace.attrs["location_code"] == "--":
                            fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].attrs["location_code"] = ""
                        else:
                            fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].attrs[
                                "location_code"
                            ] = trace.attrs["location_code"]
                    elif k == "snr":
                        fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].attrs["snr"] = (
                            "[" + ", ".join([f"{x:.1f}" for x in trace.attrs["snr"]]) + "]"
                        )
                    else:
                        fp_out[f"{event_id}_{'.'.join(phase['fname'].split('.')[:4])}"].attrs[k] = trace.attrs[k]
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
