# %%
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# %%
shift_channel = 5000

# %%
das_north = pd.read_csv("/net/kuafu/mnt/tank/data/EventData/Mammoth_north/das_info.csv", index_col="index")
das_south = pd.read_csv("/net/kuafu/mnt/tank/data/EventData/Mammoth_south/das_info.csv", index_col="index")
assert(das_south.index.max() < shift_channel)
assert(das_north.index.max() < shift_channel)
das_north.index = shift_channel - das_north.index
das_south.index = shift_channel + das_south.index
tmp = pd.concat([das_north, das_south])
tmp.to_csv("das_info.csv")

# %%
picks_north = list(Path("../../Mammoth_north_16/picks_phasenet_das_merged").rglob('*.csv'))
picks_south = list(Path("../../Mammoth_south_16/picks_phasenet_das_merged").rglob('*.csv'))
output_path = Path("picks_phasenet_das")
if not output_path.exists():
    output_path.mkdir()

# %%
events = set()
for pick in picks_north:
    events.add(pick.name)
for pick in picks_south:
    events.add(pick.name)
events = list(events)

# %%
for event in tqdm(events):
    if Path(f"../../Mammoth_north_16/picks_phasenet_das_merged/{event}").exists():
        tmp_north = pd.read_csv(f"../../Mammoth_north_16/picks_phasenet_das_merged/{event}")
        tmp_north["channel_index"] = shift_channel - tmp_north["channel_index"]
    if Path(f"../../Mammoth_south_16/picks_phasenet_das_merged/{event}").exists():
        tmp_south = pd.read_csv(f"../../Mammoth_south_16/picks_phasenet_das_merged/{event}")
        tmp_south["channel_index"] = shift_channel + tmp_south["channel_index"]
    tmp_combined = pd.concat([tmp_north, tmp_south])
    tmp_combined.to_csv(f"{output_path}/{event}", index=False)

# %%



