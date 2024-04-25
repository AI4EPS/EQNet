## EQNet: EarthQuake Neural Networks

![](https://img.shields.io/badge/python-3.8%2B-blue)

### [PhaseNet](https://github.com/AI4EPS/PhaseNet)
Original PhaseNet for picking arrival time.

### PhaseNet+
PhaseNet for picking arrival time and polarity.
```
python predict.py --model phasenet_plus --data_list mseed.txt --result_path ./results --batch_size=1 --format mseed
```

### [PhaseNet-TF](https://ziyixi.science/researchprojects/phasenet-tf/)
PhaseNet in time-frequency domain for deep earthquakes


### [PhaseNet-DAS](https://arxiv.org/abs/2302.08747)
PhaseNet for Distributed Acoustic Sensing (DAS) data.

- Example: [notebook](https://ai4eps.github.io/EQNet/phasenet_das)

<!-- e.g.,
```
python predict.py --model phasenet --add_polarity --add_event --format mseed --data_path /kuafu/jxli/Data/SeismicEventData/mammoth_south/data/ --data_list mammoth_south.txt --batch_size 1 --result_path mammoth_south 
```
```
torchrun --standalone --nproc_per_node 8 predict.py --model phasenet --data_path /atomic-data/zhuwq/Hawaii_Loa --format mseed  --batch_size=1 --result_path Hawaii_Loa --response_xml /atomic-data/zhuwq/response_hawaii_loa.xml --min_prob 0.3 --highpass_filter --add_polarity  --add_event
```
```
torchrun --standalone --nproc_per_node 4 predict.py --model phasenet --batch_size=32 --hdf5-file datasets/NCEDC/ncedc_event_dataset.h5 --result_path results_ncedc_event_dataset_0.1 --add_polarity --add_event  --dataset=seismic_trace --min_prob 0.1
``` -->

<!-- ## Install
```
pip install -r requirements.txt
```

## Prediction -->

<!-- 
*Test data*
```
wget https://huggingface.co/datasets/AI4EPS/quakeflow_das/resolve/main/data/ridgecrest_north/ci37280444.h5
```

```
python predict.py --model phasenet_das --amp --data_path /path_to_data --result_path ./results --cut_patch
```

e.g.,
```
python predict.py --model phasenet_das --amp  --data_path /net/kuafu/mnt/tank/data/EventData/Arcata_Spring2022/data --result_path Arcata_Spring2022 --plot_figure
```
```
python predict.py --model phasenet_das --amp --data_path Forge_Utah/data --result_path ./results --location forge --phases P S PS
```
Continuous data:
```
torchrun --standalone --nproc_per_node=4 predict.py --data_path /kuafu/DASdata/Hawaii_desampled/ --result_path Hawaii_5Hz --system optasense --cut_patch --nt=20480 --nx=3000 --workers=4 --batch_size=1 --amp --highpass_filter=5.0
```

Arguments:
- add the *--plot_figure* argument to plot results. 
- add the *--highpass_filter 1.0* to add highpass filter.

## Training


### Training PhaseNet-DAS
```
torchrun --standalone --nproc_per_node=4 train.py --model phasenet_das --batch-size=4 --stack-event --stack-noise --resample-space --resample-time --masking --amp --random-crop --output=model_phasenet_das --epochs=30 --wd=1e-1
```

```
python ../train.py --model phasenet_das --batch-size=4 --stack-event --stack-noise --resample-space --resample-time --masking --amp --random-crop --output=model_phasenet_das --epochs=30 --wd=1e-1 --data-list results/training/mammoth_north/data.txt results/training/mammoth_south/data.txt results/training/ridgecrest_north/data.txt results/training/ridgecrest_south/data.txt --label-path results/training/mammoth_north/labels/ results/training/mammoth_south/labels/ results/training/ridgecrest_north/labels/ results/training/ridgecrest_south/labels/ --noise-list results/training/mammoth_north/noise.txt results/training/mammoth_south/noise.txt  results/training/ridgecrest_north/noise.txt results/training/ridgecrest_south/noise.txt
```

e.g.,
```
python train.py --nt 5000 --nx 1280 -b 3 --phases P S SP --output Utah -->
<!-- ``` -->

<!-- ### Training PhaseNet
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
``` -->
