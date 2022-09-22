## Install

```
pip install -r requirements.txt
```

## Training
- Using Single GPU/CPU
```bash
python train.py --output-dir output
```

- Using Multi-GPUs
```bash
torchrun --standalone --nproc_per_node=4 train.py --output-dir output
```

### Training PhaseNet
```
python train.py --model=phasenet --backbone=resnet50 --dataset=/atomic-data/poggiali/test1.h5 --output-dir result_phasenet
```

### Training EQNet
```
python train.py --model=eqnet --backbone=resnet50 --output-dir result_eqnet
```

### Training DeepDenoiser
### Training PhaseNet-DAS
### Training AutoEncoder


## Prediction

- Using the pretrained model on default
  - add the *--plot_figure* argument to plot results. 
```bash
python predict.py --data_path /path_to_data --result_path ./result
```

- Using the local pretrained model
```bash
python predict.py --data_path /path_to_data --result_path ./result --resume  pretrained_model.pth
```


