# %%
import os
from pathlib import Path

import fsspec
import torch

os.environ["MKL_NUM_THREADS"] = "8"  # numpy

# %%
protocol = "gs://"
bucket = "das_mammoth"
# model = "phasenet_das"
model = "phasenet_das_v1"

# %%
fs = fsspec.filesystem(protocol.replace("://", ""))
folders = ["north", "south"]

for folder in folders:
    h5_list = fs.glob(f"{bucket}/{folder}/data/*h5")

    # %%
    result_path = Path(f"results/{model}/{folder}")
    if not result_path.exists():
        result_path.mkdir(parents=True)

    # %%
    with open(result_path / f"h5_list.txt", "w") as f:
        for i, h5 in enumerate(h5_list):
            f.write(f"{protocol}" + h5 + "\n")

    # %%
    num_gpu = torch.cuda.device_count()
    base_cmd = f"../predict.py --model phasenet_das --format h5 --data_list {result_path/'h5_list.txt'}  --result_path {result_path} --dataset=das --min_prob=0.5 --batch_size=1 --workers=0 --system=optasense --cut_patch --resample_time"  # --skip_existing"
    if num_gpu == 0:
        cmd = f"python {base_cmd} --device=cpu"
    elif num_gpu == 1:
        cmd = f"python {base_cmd}"
    else:
        cmd = f"torchrun --standalone --nproc_per_node {num_gpu} {base_cmd}"
    print(cmd)
    os.system(cmd)

    # %%
    cmd = f"gsutil -m cp -r {result_path}/picks_phasenet_das {protocol}{bucket}/{folder}/{model}/picks"
    print(cmd)
    os.system(cmd)
