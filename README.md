## Training
```bash
torchrun --standalone --nproc_per_node=4 train.py --config-file configs/das/per_pixel_baseline_unet.yaml --output-dir model
```

## Prediction

- Using the pretrained model on default
```bash
python predict.py --config-file configs/das/per_pixel_baseline_unet.yaml --data_path /path_to_data --result_path ./result
```

- Using the local pretrained model
```bash
python predict.py --config-file configs/das/per_pixel_baseline_unet.yaml --data_path /path_to_data --result_path ./result --resume  pretrained_model.pth
```


