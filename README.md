# EQNet2

EQNet2 provide different backbone sizes (`swin2` for Base, `swin_nano` for Small, `swin_pico` for Tiny). The checkpoint of the model can be found in [AI4EPS/models](https://github.com/AI4EPS/models/releases/tag/EQNet2-v0).

## training

Training command for EQNet2 with Swin2 backbone and UperNet2 head with hdf5 dataset. Add `--no-prompt` to disable prompt-based association. Use `--resume` to resume training from output_dir or `--resume /path/to/checkpoint.pth` to resume from a specific checkpoint.

```bash
torchrun --standalone --nproc_per_node=4 train.py --model=eqnet --backbone=swin2 --head=upernet2 --no-prompt \
--output-dir /path/to/output --hdf5-file /path/to/waveform_train.h5 --test-hdf5-file /path/to/waveform_test.h5 \
--num-stations-list 16 --use-station-location \
--stack-event --cat-event --sample-by-distance \
--batch-size=40 --grad-accum 1 --workers=14 \
--mtl dwa_inverse --temp 1.5 \
--weight-decay 0.05 --norm-weight-decay 0.0 --bias-weight-decay 0.0 --custom-weight-decay 0.0 --timeout=30 \
--wandb --wandb-name wandb-run-name --wandb-project wandb-project-name --wandb-watch --wandb-group wandb-group-name --wandb-dir /tmp/ 
```

## inference

Inference command for EQNet2 with Swin2 backbone and UperNet2 head with hdf5 dataset. Add `--no-prompt` to disable prompt-based association. Use `--resume /path/to/checkpoint.pth` to load a specific checkpoint.

```bash
torchrun --standalone --nproc_per_node 4 predict.py --model eqnet --backbone=swin2 --head=upernet2 --no-prompt --dataset seismic_network --use-station-location \
--hdf5-file /path/to/waveform_test.h5 --result_path /path/to/result --resume /path/to/checkpoint.pth \
--is_pad --batch_size=1 --num_stations 16 --min_prob 0.1 \
> test_predict_32_2.txt 2>&1
```

Evaluation command

```bash
python metrics_old.py --pick_dir /path/to/result/picks_eqnet --event_dir /path/to/result/events_eqnet --suffix test --h5_file /path/to/waveform_test.h5 > test_predict_metric.txt 2>&1
python metrics.py --save_path /path/to/result/metrics --pick_dir /path/to/result/picks_eqnet --event_dir /path/to/result/events_eqnet --suffix test --h5_file /path/to/waveform_test.h5 > test_predict_metric.txt 2>&1
```

Inference command for mseed continuous data. Add `--no-prompt` to disable prompt-based association module. Add `--association` to enable association output. Use `--resume /path/to/checkpoint.pth` to load a specific checkpoint. Continuous data follow the [QuakeFlow](https://github.com/AI4EPS/QuakeFlow) format.

```bash
torchrun --standalone --nproc_per_node 2 predict.py --model eqnet --backbone=swin2 --head=upernet2 --dataset seismic_network \
--batch_size=1 --workers=4 \
--data_path /continuous_test/Ridgecrest/waveforms/2019 --use-station-location --format mseed_hour --resume /path/to/model.pth --response_xml /continuous_test/Ridgecrest/obspy/inventory.xml --station_file /continuous_test/Ridgecrest/obspy/valid_stations.csv --result_path /path/to/result --cut_patch --nx 96 --nt 360000 --sort --association > test_predict_mseed_test.txt 2>&1
```

Run below script to do time-space clustering into preliminary catalog and compare with the ground truth catalog. Then use the output catalog for location with ADLoc from [QuakeFlow](https://github.com/AI4EPS/QuakeFlow)

```bash
python metrics_continuous.py --event_dir /path/to/result/events_eqnet --catalog_dir /continuous_test/Ridgecrest/obspy/catalog.csv --station_info_dir /continuous_test/Ridgecrest/obspy/stations.json --catalog_pred_dir /path/to/result/catalog_pred_timespace.csv --suffix timespace --cluster_method time_space --sliding_window=86400 --gpu > test_Ridgecrest_metric_timespace.txt 2>&1
```