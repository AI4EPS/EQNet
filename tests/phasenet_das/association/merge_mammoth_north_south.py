# %%
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# %%
shift_channel = 5000
# data_path = Path("/net/kuafu/mnt/tank/data/EventData/")
data_path = Path("/kuafu/DASEventData")
north_data_dir = "mammoth_north"
south_data_dir = "mammoth_south"
pick_path = Path("../../../")
north_pick_dir = "Mammoth_north_new"
south_pick_dir = "Mammoth_south_new" 
output_path = Path("picks_phasenet_das")
if not output_path.exists():
    output_path.mkdir()

# %%
das_north = pd.read_csv(data_path / f"{north_data_dir}/das_info.csv", index_col="index")
das_south = pd.read_csv(data_path / f"{south_data_dir}/das_info.csv", index_col="index")
assert(das_south.index.max() < shift_channel)
assert(das_north.index.max() < shift_channel)
das_north.index = shift_channel - das_north.index
das_south.index = shift_channel + das_south.index
tmp = pd.concat([das_north, das_south])
tmp.to_csv("das_info.csv")

# %%
picks_north = list((pick_path / f"{north_pick_dir}/picks_phasenet_das_raw/").glob('*.csv'))
picks_south = list((pick_path / f"{south_pick_dir}/picks_phasenet_das_raw/").glob('*.csv'))


# %%
events = set()
for pick in picks_north:
    events.add(pick.name)
for pick in picks_south:
    events.add(pick.name)
events = list(events)

# %%
for event in tqdm(events):

    tmp_north = None
    if (pick_path / f"{north_pick_dir}/picks_phasenet_das_raw/{event}").exists():
        
        try:
            tmp_north = pd.read_csv(pick_path / f"{north_pick_dir}/picks_phasenet_das_raw/{event}")
            tmp_north["channel_index"] = shift_channel - tmp_north["channel_index"]
        except:
            pass

    tmp_south = None
    if (pick_path / f"{south_pick_dir}/picks_phasenet_das_raw/{event}").exists():
        try:
            tmp_south = pd.read_csv(pick_path / f"{south_pick_dir}/picks_phasenet_das_raw/{event}")
            tmp_south["channel_index"] = shift_channel + tmp_south["channel_index"]
        except:
            pass
        
    if (tmp_north is None) and (tmp_south is None):
        continue
    elif (tmp_north is None) and (tmp_south is not None):
        tmp_combined = tmp_south
    elif (tmp_north is not None) and (tmp_south is None):
        tmp_combined = tmp_north
    else:
        tmp_combined = pd.concat([tmp_north, tmp_south])

    tmp_combined.to_csv(f"{output_path}/{event}", index=False)

# %%



