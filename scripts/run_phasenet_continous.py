# %%
import os
from pathlib import Path

import fsspec
import torch

# %%
protocol = "gs://"
bucket = "das_mammoth"

# %%
fs = fsspec.filesystem(protocol.replace("://", ""))
folders = ["north", "south"]

for folder in folders:
    h5_list = fs.glob(f"{bucket}/{folder}/data/*h5")

    # %%
    result_path = Path(f"results/phasenet/{folder}")
    if not result_path.exists():
        result_path.mkdir(parents=True)

    # %%
    with open(result_path / f"h5_list.txt", "w") as f:
        for i, h5 in enumerate(h5_list):
            f.write(f"{protocol}" + h5 + "\n")

    # %%
    num_gpu = torch.cuda.device_count()
    base_cmd = f"../predict.py --model phasenet --add_polarity --add_event --format h5 --data_list {result_path/'h5_list.txt'} --result_path {result_path} --dataset=das --system=optasense --cut_patch --nx=1024 --resample_time --batch_size=1 --workers=0"
    if num_gpu == 0:
        cmd = f"python {base_cmd} --device=cpu"
    elif num_gpu == 1:
        cmd = f"python {base_cmd}"
    else:
        cmd = f"torchrun --standalone --nproc_per_node {num_gpu}  {base_cmd}"
    print(cmd)
    os.system(cmd)

    # %%
    cmd = f"gsutil -m cp -r {result_path}/picks_phasenet {protocol}{bucket}/{folder}/phasenet/picks"
    print(cmd)
    os.system(cmd)
