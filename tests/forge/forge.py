import os

cmd = f"python ../../predict.py --model phasenet_das --data_list data.lst --result_path results --format=forge_segy --batch_size 1 --workers 0"

os.system(cmd)
