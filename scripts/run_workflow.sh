#! /bin/bash

python run_phasenet.py > log_phasenet.txt 2>&1 && python run_gamma.py --folder=ridgecrest_north > log_gamma.txt 2>&1  && sudo shutdown -h now