# %%
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import fsspec
import multiprocessing as mp
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
def run(event, protocol, bucket, picker, shift_channel, pick_path):
    # for event in tqdm(events):
    # fs check if csv is empty
    try:
        fname = f"{protocol}{bucket}/mammoth_north/{picker}/picks/{event}.csv"
        picks_north = pd.read_csv(fname)
        if "station_id" in picks_north.columns:
            picks_north.rename(columns={"station_id": "channel_index"}, inplace=True)
        picks_north["channel_index"] = shift_channel - picks_north["channel_index"]
    except:
        picks_north = pd.DataFrame()

    try:
        fname = f"{protocol}{bucket}/mammoth_south/{picker}/picks/{event}.csv"
        picks_south = pd.read_csv(fname)
        if "station_id" in picks_south.columns:
            picks_south.rename(columns={"station_id": "channel_index"}, inplace=True)
        picks_south["channel_index"] = shift_channel + picks_south["channel_index"]
    except:
        picks_south = pd.DataFrame()

    picks = pd.concat([picks_north, picks_south])

    if not (pick_path / picker).exists():
        (pick_path / picker).mkdir(parents=True)

    if len(picks) > 0:
        picks = filter_duplicates(picks)
        picks.to_csv(pick_path / picker / f"{event}.csv", index=False)


# %%
if __name__ == "__main__":
    # %%
    protocol = "gs://"
    bucket = "quakeflow_das"
    fs = fsspec.filesystem(protocol.replace("://", ""))

    # %%
    pickers = ["phasenet", "phasenet_das", "phasenet_das_v1"]
    pickers = ["phasenet"]
    pickers = pickers[::-1]
    picker_name = {
        "phasenet": "PhaseNet",
        "phasenet_das": "PhaseNet-DAS v1",
        "phasenet_das_v1": "PhaseNet-DAS v2",
    }

    folders = ["mammoth_north", "mammoth_south"]

    figure_path = Path("paper_figures")
    if not figure_path.exists():
        figure_path.mkdir(parents=True)

    pick_path = Path("picks")
    if not pick_path.exists():
        pick_path.mkdir(parents=True)

    shift_channel = 5000

    # %%
    das_north = pd.read_csv(f"{protocol}{bucket}/mammoth_north/das_info.csv", index_col="index")
    das_south = pd.read_csv(f"{protocol}{bucket}/mammoth_south/das_info.csv", index_col="index")
    assert das_south.index.max() < shift_channel
    assert das_north.index.max() < shift_channel
    das_north.index = shift_channel - das_north.index
    das_south.index = shift_channel + das_south.index
    tmp = pd.concat([das_north, das_south])
    tmp.to_csv("das_info.csv")

    # %%
    for picker in pickers:
        events_north = fs.glob(f"{protocol}{bucket}/mammoth_north/data/*.h5")
        events_south = fs.glob(f"{protocol}{bucket}/mammoth_south/data/*.h5")
        events_north = [x.split("/")[-1].split(".")[0] for x in events_north]
        events_south = [x.split("/")[-1].split(".")[0] for x in events_south]

        events = set(events_north) & (set(events_south))
        events = list(events)

        ncpu = mp.cpu_count()
        pbar = tqdm(total=len(events))

        # for event in tqdm(events):
        #     run(event, protocol, bucket, picker, shift_channel, pick_path)

        with mp.get_context("spawn").Pool(ncpu) as p:
            for event in events:
                p.apply_async(
                    run,
                    args=(event, protocol, bucket, picker, shift_channel, pick_path),
                    callback=lambda _: pbar.update(),
                )
            p.close()
            p.join()

# %%
