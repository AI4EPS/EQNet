# %%
import requests
import h5py
import numpy as np

h5_file = "./38554943.h5"

with h5py.File(h5_file, "r") as f:
    data = f["data"][:]
    data = data[np.newaxis, :, :]
    timestamp = f["data"].attrs["begin_time"]
    data_id = f'{f["data"].attrs["event_id"]}'


# %%

PHASENET_DAS_API_URL = "http://127.0.0.1:8000"
# PHASENET_DAS_API_URL = "http://test.quakeflow.com:8001" ## local machine

req = {"id": [data_id], "timestamp": [timestamp], "vec": [data.tolist()], "dt_s": [0.01]}

resp = requests.post(f"{PHASENET_DAS_API_URL}/predict", json=req)
print(resp)
print("Picks", resp.json())



# %%
