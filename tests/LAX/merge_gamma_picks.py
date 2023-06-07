# %%
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import io
import matplotlib.pyplot as plt
import obspy
from obspy.clients.fdsn.client import Client
from obspy import UTCDateTime
import re
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import seaborn as sns
import json
from collections import defaultdict

# %%
gamma_picks = []
for pick in tqdm(sorted(list(Path("gamma").glob("*.csv")))):
    tmp = pd.read_csv(pick)
    tmp["file_id"] = pick.stem
    gamma_picks.append(tmp)
gamma_picks = pd.concat(gamma_picks)

# %%
# event_id = file_id + event_index
gamma_picks["event_index"] = gamma_picks["file_id"] + "_" + gamma_picks["event_index"].astype(str)

# map event_id to a new index of int
event_id_map = {e: i for i, e in enumerate(sorted(gamma_picks["event_index"].unique()))}
gamma_picks["event_index"] = gamma_picks["event_index"].map(event_id_map)

# rename event_id to event_index
gamma_picks.drop(columns=["file_id"], inplace=True)
gamma_picks.to_csv("gamma_picks.csv", index=False)

# %%
gamma_events = pd.read_csv("catalog_gamma.csv")
gamma_events["event_index"] = gamma_events["event_id"] + "_" + gamma_events["event_index"].astype(str)
gamma_events["event_index"] = gamma_events["event_index"].map(event_id_map)
gamma_events.drop(columns=["event_id"], inplace=True)
gamma_events.drop(columns=["x(km)", "y(km)", "z(km)"], inplace=True)
gamma_events.to_csv("gamma_events.csv", index=False)
# %%
