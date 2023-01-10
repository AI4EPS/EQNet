## Install
```
pip install -r requirements.txt
```

## Prediction
### PhaseNet
```
python predict.py --model phasenet --data_path /path_to_data --result_path ./results --batch_size=1
```

### PhaseNet-Polarity
```
python predict.py --model phasenet --add_polarity --data_path /path_to_data --result_path ./results --batch_size=1
```

e.g.,
```
python predict.py --model phasenet --data_path /kuafu/jxli/Data/SeismicEventData/mammoth_south/data/ --data_list mammoth_south.txt  --format mseed --batch_size 1 --result_path mammoth_south --add_polarity
```
```
torchrun --standalone --nproc_per_node 8 predict.py --model phasenet --data_path /atomic-data/zhuwq/Hawaii_Loa --format mseed  --batch_size=1 --result_path Hawaii_Loa --response_xml /atomic-data/zhuwq/response_hawaii_loa.xml --min_prob 0.3 --highpass_filter --add_polarity 
```

### PhaseNet-DAS
```
python predict.py --model phasenet_das --data_path /path_to_data --result_path ./results --cut_patch
```

e.g.,
```
python predict.py --model phasenet_das --data_path Forge_Utah/data --result_path ./results --area forge --phases P S PS
```

Arguments:
- add the *--plot_figure* argument to plot results. 


## Training


### Training PhaseNet-DAS
```
torchrun --standalone --nproc_per_node=4 train.py --model phasenet_das --batch-size=4 --stack-event --stack-noise --resample-space --resample-time --masking --amp --output=model_phasenet_das --epochs=30 --wd=1e-1
```

e.g.,
```
python train.py --nt 5000 --nx 1280 -b 3 --phases P S SP --output Utah
```

### Training PhaseNet
```
torchrun --standalone --nproc_per_node 4 train.py --model phasenet --batch-size=256 --hdf5-file /scratch/zhuwq/EQNet_update2/datasets/NCEDC/ncedc_event_dataset_3c.h5 --lr 0.01 --workers=32 --stack-event --flip-polarity --drop-channel --output model_phasenet
```
```
torchrun --standalone --nproc_per_node 4 train.py --model phasenet --batch-size=256 --data-path datasets/NCEDC/ncedc_h5 --lr 0.01 --workers=32 --stack-event --flip-polarity --drop-channel --output model_phasenet
```

### Training EQNet
```
torchrun --standalone --nproc_per_node 4 train.py --model=eqnet --backbone=resnet50 --output-dir result_eqnet
```

### Training DeepDenoiser

### Training AutoEncoder


Options:
- Using Single GPU/CPU
```bash
python train.py --output-dir output
```

- Using Multi-GPUs
```bash
torchrun --standalone --nproc_per_node=4 train.py --output-dir output
```