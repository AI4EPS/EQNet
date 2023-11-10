# %%
import os
import torch


# %% example command
## python train_phasenet_das.py --resume --data_path=results/training_v1/


# %%
def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run GaMMA")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--data_path", type=str, default="results/training/")

    return parser


args = get_args_parser().parse_args()

# %%
protocol = "gs://"
bucket = "quakeflow_das"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["OMP_NUM_THREADS"] = "4"

# %%
num_gpu = torch.cuda.device_count()
# base_cmd = f"../train.py --model phasenet_das --compile --sync-bn --amp --batch-size=4 --epochs=10 --wd=1e-1  --stack-event --stack-noise --resample-space --resample-time --masking --random-crop --output=model_phasenet_das --wandb --wandb-project=phasenet-das"
base_cmd = f"../train.py --model phasenet_das --sync-bn --batch-size=4 --epochs=10 --wd=1e-1  --stack-event --stack-noise --resample-space --resample-time --masking --random-crop --output=model_phasenet_das --wandb --wandb-project=phasenet-das"
if args.resume:
    base_cmd += " --resume=True"
# base_cmd += f" --data-path {protocol}{bucket} --data-list {args.data_path}/data.txt --label-list {args.data_path}/labels_train.txt --noise-list {args.data_path}/noise_train.txt --test-data-path {protocol}{bucket} --test-data-list {args.data_path}/data.txt --test-label-list {args.data_path}/labels_test.txt --test-noise-list {args.data_path}/noise_test.txt"
base_cmd += f" --data-path {protocol}{bucket} --data-list {args.data_path}/data.txt --label-list {args.data_path}/labels_train.txt --noise-list {args.data_path}/noise_train.txt --test-data-path {protocol}{bucket} --test-data-list {args.data_path}/data.txt --test-label-list results/training/labels_test.txt --test-noise-list results/training/noise_test.txt"
if num_gpu == 0:
    cmd = f"python {base_cmd} --device cpu"
elif num_gpu == 1:
    cmd = f"python {base_cmd}"
else:
    cmd = f"torchrun --standalone --nproc_per_node={num_gpu} {base_cmd}"
print(cmd)
os.system(cmd)
# %%
