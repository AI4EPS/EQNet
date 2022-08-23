# %%
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# %%
shift_channel = 5000
root_path = "/net/kuafu/mnt/tank/data/EventData/"
picks_path = "/net/arius/scratch/zhuwq/EQNet/"

# %%
das_north = pd.read_csv(root_path + "Mammoth_north/das_info.csv", index_col="index")
das_south = pd.read_csv(root_path + "Mammoth_south/das_info.csv", index_col="index")
assert das_south.index.max() < shift_channel
assert das_north.index.max() < shift_channel
das_north.index = shift_channel - das_north.index
das_south.index = shift_channel + das_south.index
tmp = pd.concat([das_north, das_south])
tmp.to_csv("das_info.csv")

# %%
picks_north = sorted(list(Path(picks_path + "Mammoth_north_debug/picks_phasenet_das_merged/").rglob("*.csv")))
picks_south = sorted(list(Path(picks_path + "Mammoth_south_debug/picks_phasenet_das_merged/").rglob("*.csv")))
output_path = Path("picks_phasenet_das")
if not output_path.exists():
    output_path.mkdir()

# %%
start_date = datetime.fromisoformat(
    min(picks_north[0].name.lstrip("Mammoth-North-"), picks_south[0].name.lstrip("Mammoth-South-")).split("T")[0]
)
end_date = datetime.fromisoformat(
    min(picks_north[-1].name.lstrip("Mammoth-North-"), picks_south[-1].name.lstrip("Mammoth-South-")).split("T")[0]
)

# %%
picks = []
for f in tqdm(picks_north):
    tmp = pd.read_csv(f)
    tmp["channel_index"] = shift_channel - tmp["channel_index"]
    tmp.drop(columns=["phase_index"], inplace=True)
    picks.append(tmp)

for f in tqdm(picks_south):
    tmp = pd.read_csv(f)
    tmp["channel_index"] = shift_channel + tmp["channel_index"]
    tmp.drop(columns=["phase_index"], inplace=True)
    picks.append(tmp)

combined_picks = pd.concat(picks)
combined_picks.to_csv(f"mammoth_picks.csv", index=False)

# %%
