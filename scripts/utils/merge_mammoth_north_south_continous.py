# %%
import multiprocessing as mp
from pathlib import Path

import fsspec
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm


def filter_duplicates(df):
    # Sort the DataFrame
    # df.sort_values(["station_id", "phase_index"], ascending=[True, True], inplace=True)
    df.sort_values(["channel_index", "phase_index"], ascending=[True, True], inplace=True)

    # # Define a custom function to merge rows
    # def merge_rows(group):
    #     merged_rows = []
    #     previous_index = None

    #     for i, row in group.iterrows():
    #         if (previous_index is not None) and (abs(row["phase_index"] - previous_index) <= 10):
    #             # merged_rows[-1] = row
    #             pass
    #         else:
    #             merged_rows.append(row)

    #         previous_index = row["phase_index"]

    #     return pd.DataFrame(merged_rows)

    # # Group by station_id and apply the custom function
    # # df = df.groupby("station_id", group_keys=False).apply(merge_rows).reset_index(drop=True)
    # df = df.groupby("channel_index", group_keys=False).apply(merge_rows).reset_index(drop=True)

    merged_rows = []
    previous_channel_index = None
    previous_phase_index = None
    for _, row in df.iterrows():
        if (
            (previous_channel_index is not None)
            and (previous_phase_index is not None)
            and (row["channel_index"] == previous_channel_index)
            and (abs(row["phase_index"] - previous_phase_index) <= 10)
        ):
            # merged_rows[-1] = row ## keep the last one
            pass  ## keep the first one
        else:
            merged_rows.append(row)
        previous_channel_index = row["channel_index"]
        previous_phase_index = row["phase_index"]
    df = pd.DataFrame(merged_rows).reset_index(drop=True)

    return df


# %%
def run(hour, fs, protocol, bucket, picker, shift_channel, pick_path, sampling_rate=100):
    picks_north = []
    for h in [hour, hour - pd.Timedelta("1h")]:
        for fname in fs.glob(f"{bucket}/north/{picker}/picks/Mammoth-North-{h.strftime('%Y-%m-%dT%H')}????Z.csv"):
            # picks_north.append(pd.read_csv(f"{protocol}{fname}", parse_dates=["phase_time"]))
            try:
                tmp = pd.read_csv(f"{protocol}{fname}", parse_dates=["phase_time"])
                picks_north.append(tmp)
            except:
                print(f"Failed to read {fname}")
    if len(picks_north) > 0:
        picks_north = pd.concat(picks_north)
        if "station_id" in picks_north.columns:
            picks_north.rename(columns={"station_id": "channel_index"}, inplace=True)
        picks_north["channel_index"] = shift_channel - picks_north["channel_index"]
    else:
        picks_north = pd.DataFrame()

    # try:
    picks_south = []
    for h in [hour, hour - pd.Timedelta("1h")]:
        for fname in fs.glob(f"{bucket}/south/{picker}/picks/Mammoth-South-{h.strftime('%Y-%m-%dT%H')}????Z.csv"):
            # picks_south.append(pd.read_csv(f"{protocol}{fname}", parse_dates=["phase_time"]))
            try:
                tmp = pd.read_csv(f"{protocol}{fname}", parse_dates=["phase_time"])
                picks_south.append(tmp)
            except:
                print(f"Failed to read {fname}")
    if len(picks_south) > 0:
        picks_south = pd.concat(picks_south)
        if "station_id" in picks_south.columns:
            picks_south.rename(columns={"station_id": "channel_index"}, inplace=True)
        picks_south["channel_index"] = shift_channel + picks_south["channel_index"]
    else:
        picks_south = pd.DataFrame()

    picks = pd.concat([picks_north, picks_south])
    # picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    picks = picks[(picks["phase_time"] >= hour) & (picks["phase_time"] < hour + pd.Timedelta("1h"))]
    picks["phase_index"] = picks["phase_time"].apply(lambda x: x - hour).dt.total_seconds() * 100
    picks["phase_index"] = picks["phase_index"].astype(int)

    if not (pick_path / picker / "picks").exists():
        (pick_path / picker / "picks").mkdir(parents=True)

    if len(picks) > 0:
        picks = filter_duplicates(picks)
        picks["phase_time"] = picks["phase_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
        picks.sort_values(["phase_time", "channel_index"], inplace=True)
        picks.to_csv(pick_path / picker / "picks" / f"{hour.isoformat()}.csv", index=False)

    return 0


# %%
if __name__ == "__main__":
    # %%
    protocol = "gs://"
    bucket = "das_mammoth"
    fs = fsspec.filesystem(protocol.replace("://", ""))

    # %%
    pickers = ["phasenet", "phasenet_das", "phasenet_das_v1"]
    # pickers = ["phasenet"]
    pickers = pickers[::-1]
    picker_name = {
        "phasenet": "PhaseNet",
        "phasenet_das": "PhaseNet-DAS v1",
        "phasenet_das_v1": "PhaseNet-DAS v2",
    }

    folders = ["north", "south"]

    figure_path = Path("paper_figures")
    if not figure_path.exists():
        figure_path.mkdir(parents=True)

    pick_path = Path("merged_picks")
    if not pick_path.exists():
        pick_path.mkdir(parents=True)

    shift_channel = 5000

    # %%
    das_north = pd.read_csv(f"{protocol}{bucket}/north/das_info.csv", index_col="index")
    das_south = pd.read_csv(f"{protocol}{bucket}/south/das_info.csv", index_col="index")
    assert das_south.index.max() < shift_channel
    assert das_north.index.max() < shift_channel
    das_north.index = shift_channel - das_north.index
    das_south.index = shift_channel + das_south.index
    tmp = pd.concat([das_north, das_south])
    tmp.to_csv("das_info.csv")

    # %%
    for picker in pickers:
        begin_time = pd.to_datetime("2020-11-17T00:00:00.000+00:00")
        end_time = pd.to_datetime("2020-11-25T00:00:00.000+00:00")
        hours = pd.date_range(begin_time, end_time, freq="H", inclusive="left")

        ncpu = mp.cpu_count()
        pbar = tqdm(total=len(hours))

        # for event in tqdm(events):
        #     run(event, protocol, bucket, picker, shift_channel, pick_path)

        with mp.get_context("spawn").Pool(ncpu) as p:
            results = []
            for hour in hours:
                r = p.apply_async(
                    run,
                    args=(hour, fs, protocol, bucket, picker, shift_channel, pick_path),
                    callback=lambda _: pbar.update(),
                )
                results.append(r)
            for r in results:
                r.get()
            p.close()
            p.join()

# %%
