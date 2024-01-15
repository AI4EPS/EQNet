# %%
import fsspec
import pandas as pd
from datetime import datetime, timezone
import os
from pathlib import Path
from tqdm import tqdm

# %%
protocol = "gs://"
bucket = "quakeflow_das"
fs = fsspec.filesystem(protocol.replace("://", ""))

# %%
folder = "ridgecrest_north"
result_path = Path(f"data/{folder}")
if not result_path.exists():
    result_path.mkdir(parents=True)

# %%
catalog = pd.read_csv(f"{protocol}{bucket}/{folder}/catalog_data.csv", parse_dates=["event_time"])

# %%
begin_time = datetime.fromisoformat("2020-06-20T00:00:00").replace(tzinfo=timezone.utc)
end_time = datetime.fromisoformat("2020-07-30T00:00:00").replace(tzinfo=timezone.utc)

# %%
catalog = catalog[(catalog.event_time >= begin_time) & (catalog.event_time <= end_time)]

# %%
event_ids = catalog.event_id.unique()

# %%
# download all event_ids from gsutil
for event_id in tqdm(event_ids):
    # cmd = f"gsutil -m cp -r {protocol}{bucket}/{folder}/data/{event_id}.h5 {result_path}/"
    cmd = f"rclone copy -P {protocol}{bucket}/{folder}/data/{event_id}.h5 {result_path}/"
    print(cmd)
    os.system(cmd)
    raise

# %%
