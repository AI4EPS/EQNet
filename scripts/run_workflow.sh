#! /bin/bash

#### Training PhasetNet-DAS v0 #####
python run_phasenet.py > log_run_phasenet.txt 2>&1

python run_gamma.py --folder=mammoth_north --picker=phasenet && python run_gamma.py --folder=mammoth_south --picker=phasenet && python run_gamma.py --folder=ridgecrest_north --picker=phasenet && python run_gamma.py --folder=ridgecrest_south --picker=phasenet

# edit hardcoding in build_training.py
python build_training.py

python train_phasenet_das.py --resume --data_path=results/training_v0/ > log_train_phasenet_das.txt 2>&1 

#### Training PhasetNet-DAS v1 #####
# edit hardcoded model in build_training.py
python run_phasenet_das.py > log_run_phasenet_das.txt 2>&1

python run_gamma.py --folder=mammoth_north --picker=phasenet_das && python run_gamma.py --folder=mammoth_south --picker=phasenet_das && python run_gamma.py --folder=ridgecrest_north --picker=phasenet_das && python run_gamma.py --folder=ridgecrest_south --picker=phasenet_das

# edit hardcoded model in build_training.py
python build_training.py

python train_phasenet_das.py --resume --data_path=results/training_v0/ > log_train_phasenet_das.txt 2>&1 

#### Training PhasetNet-DAS v2 #####
# edit hardcoded model in build_training.py
python run_phasenet_das.py > log_run_phasenet_das.txt 2>&1

python run_gamma.py --folder=mammoth_north --picker=phasenet_das_v1 && python run_gamma.py --folder=mammoth_south --picker=phasenet_das_v1 && python run_gamma.py --folder=ridgecrest_north --picker=phasenet_das_v1 && python run_gamma.py --folder=ridgecrest_south --picker=phasenet_das_v1

# edit hardcoded model in build_training.py
python build_training.py

# training not done yet

sudo poweroff