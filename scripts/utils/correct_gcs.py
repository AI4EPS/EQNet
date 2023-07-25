# %%
import fsspec
from tqdm.auto import tqdm
import multiprocessing as mp


# %%
def rename(source, target, fs):
    fs.mv(source, target)


# %%
if __name__ == "__main__":
    # %%
    bucket = "gs://quakeflow_das"
    # folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south"]
    folders = [
        "mammoth_north",
    ]
    picker = "phasenet"

    # %%
    for folder in folders:
        fs = fsspec.filesystem(
            "gs", token="/home/weiqiang/codespaces/.config/gcloud/application_default_credentials.json"
        )

        # source_path = f"{bucket}/{folder}/gamma/{picker}/picks_bak"
        # target_path = f"{bucket}/{folder}/gamma/{picker}/picks"
        source_path = f"{bucket}/{folder}/{picker}/picks"
        target_path = f"{bucket}/{folder}/{picker}/picks"

        csv = fs.glob(f"{source_path}/*csv")

        # for c in tqdm(csv):
        #     event_id = c.split("/")[-1].split("_")[-1].split(".")[0]
        #     # fs.cp(c, f"{target_path}/{event_id}.csv")
        #     fs.mv(c, f"{target_path}/{event_id}.csv")
        #     # print(c, f"{target_path}/{event_id}.csv")

        ncpu = mp.cpu_count()
        pbar = tqdm(total=len(csv))
        with mp.get_context("spawn").Pool(ncpu) as pool:
            for c in csv:
                event_id = c.split("/")[-1].split("_")[-1].split(".")[0]
                pool.apply_async(
                    rename, args=(c, f"{target_path}/{event_id}.csv", fs), callback=lambda _: pbar.update()
                )
            pool.close()
            pool.join()
