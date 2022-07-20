## Install

```
pip install -r requirements.txt
```

## Training
```bash
torchrun --standalone --nproc_per_node=4 train.py --output-dir output
```

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


