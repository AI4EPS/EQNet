# %%
import os
from pathlib import Path
import fsspec
import torch

# %%
protocol = "gs"
bucket = "gs://quakeflow_das"

# %%
fs = fsspec.filesystem(protocol)
folder = "mammoth_north"
h5_list = fs.glob(f"{bucket}/{folder}/data/*h5")

# %%
result_path = Path(f"results/phasenet/{folder}")
if not result_path.exists():
    result_path.mkdir(parents=True)

# %%
with open(result_path / "h5_list.txt", "w") as f:
    for i, h5 in enumerate(h5_list):
        f.write(f"{protocol}://" + h5 + "\n")

# %%
num_gpu = torch.cuda.device_count()
if num_gpu == 0:
    cmd = f"python ../predict.py --model phasenet --add_polarity --add_event --format h5 --data_list {result_path/'h5_list.txt'} --batch_size 1 --result_path {result_path}/phasenet/{folder} --dataset=das  --cut_patch --nx=1024"
else:
    cmd = f"torchrun --standalone --nproc_per_node 4   ../predict.py --model phasenet --add_polarity --add_event --format h5 --data_list {result_path/'h5_list.txt'} --batch_size 1 --result_path {result_path}/phasenet --dataset=das  --cut_patch --nx=1024"
print(cmd)
# os.system(cmd)

# %%
cmd = f"gsutil -m cp -r {result_path}/{folder}/phasenet/picks_phasenet {bucket}/{folder}/phasenet/picks"
print(cmd)
# os.system(cmd)

