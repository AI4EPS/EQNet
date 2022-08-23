# %%
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# %%
path_list = ["../../../Mammoth_north_debug", "../../../Mammoth_south_debug"]

# %%
for path_ in path_list:

    picks_path = Path(f"{path_}/picks_phasenet_das/")
    output_path = Path(f"{path_}/picks_phasenet_das_merged/")

    if not output_path.exists():
        output_path.mkdir()

    files = picks_path.glob("*.csv")

    file_group = defaultdict(list)
    for file in files:
        file_group[file.stem.split("_")[0]].append(file)

    num_picks = 0
    for k in tqdm(file_group, desc=f"{output_path}"):
        picks = []
        for i, file in enumerate(sorted(file_group[k])):
            with open(file, "r") as f:
                tmp = f.readlines()
                if i == 0:
                    picks.extend(tmp)
                else:
                    picks.extend(tmp[1:])  ## wihout header
        with open(output_path.joinpath(f"{k}.csv"), "w") as f:
            f.writelines(picks)
        num_picks += len(picks)
    print(f"Number of picks: {num_picks}")

# %%
