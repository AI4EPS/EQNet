# %%
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# %%
picks_csv = sorted(list(Path("/kuafu/DASdata/PhaseNet-DAS/Hawaii/picks_phasenet_das/").rglob("*.csv")))


# %%
picks = []
for f in tqdm(picks_csv):
    tmp = pd.read_csv(f)
    # tmp.drop(columns=["phase_index"], inplace=True)
    picks.append(tmp)

combined_picks = pd.concat(picks)
combined_picks.to_csv(f"hawaii_picks.csv", index=False)

# %%
