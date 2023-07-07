# %%
import os
import torch

# %%
protocol = "gs://"
bucket = "quakeflow_das"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["OMP_NUM_THREADS"] = "4"

# %%
num_gpu = torch.cuda.device_count()
# args = f"../train.py --model phasenet_das --compile --sync-bn --amp --batch-size=4 --epochs=10 --wd=1e-1  --stack-event --stack-noise --resample-space --resample-time --masking --random-crop --output=model_phasenet_das --wandb --wandb-project=phasenet-das"
args = f"../train.py --model phasenet_das --sync-bn --amp --batch-size=4 --epochs=10 --wd=1e-1  --stack-event --stack-noise --resample-space --resample-time --masking --random-crop --output=model_phasenet_das --wandb --wandb-project=phasenet-das"
args += f" --data-path {protocol}{bucket} --data-list results/training/data.txt --label-list results/training/labels_train.txt --noise-list results/training/noise_train.txt --test-data-path {protocol}{bucket} --test-data-list results/training/data.txt --test-label-list results/training/labels_test.txt --test-noise-list results/training/noise_test.txt"
if num_gpu == 0:
    cmd = f"python {args} --device cpu"
elif num_gpu == 1:
    cmd = f"python {args}"
else:
    cmd = f"torchrun --standalone --nproc_per_node={num_gpu} {args}"
print(cmd)
os.system(cmd)
# %%
