# %%
import os
from pathlib import Path
import fsspec
import h5py

# %%
protocol = "gs"
bucket = "gs://quakeflow_das"

# %%
fs = fsspec.filesystem(protocol)
h5_list = fs.glob(f"{bucket}/mammoth_north/data/*h5")

# %%
result_path = Path("results")
if not result_path.exists():
    result_path.mkdir(parents=True)

# %%

with open(result_path / "h5_list.txt", "w") as f:
    for h5 in h5_list:
        f.write(f"{protocol}://" + h5 + "\n")

# %%
# print(h5_list[0])
# with fsspec.open("gs://"+h5_list[0], "rb") as fs:
#     with h5py.File(fs, "r", lib_version='latest', swmr=True) as f:
#         print(f.keys())
        # print(f["data"].shape)
        # print(f["label"].shape)
        # print(f["event"].shape)
        # print(f["polarity"].shape)
        # print(f["event"][...])
        # print(f["polarity"][...])
        # print(f["label"][...])

# %%
cmd = f"python ../predict.py --model phasenet --add_polarity --add_event --format h5 --data_list {result_path/'h5_list.txt'} --batch_size 1 --result_path {result_path}/phasenet"
print(cmd)
# os.system(cmd)
# %%
