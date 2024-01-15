# %%
import os
from pathlib import Path
import fsspec
import h5py
import numpy as np
import multiprocessing as mp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# %%
protocol = "gs://"
bucket = "quakeflow_das"

# %%
fs = fsspec.filesystem(protocol.replace("://", ""))
folders = ["mammoth_north", "mammoth_south", "ridgecrest_north", "ridgecrest_south"]
# folders = ["mammoth_north", "mammoth_south", "ridgecrest_south"]
# folders = ["ridgecrest_north"]

pickers = ["phasenet", "phasenet_das", "phasenet_das_v1"]
pickers = pickers[::-1]

# for folder in folders:
#     h5_list = fs.glob(f"{bucket}/{folder}/data/*h5")

#     result_path = Path(f"results/stats/{folder}")
#     if not result_path.exists():
#         result_path.mkdir(parents=True)

#     with open(result_path / f"h5_list.txt", "w") as f:
#         for i, h5 in enumerate(h5_list):
#             f.write(f"{protocol}" + h5 + "\n")


# %%
sampling_rate = 100
n0 = 20 * sampling_rate
n1 = 25 * sampling_rate
s0 = 35 * sampling_rate
s1 = 40 * sampling_rate


def get_snr(h5, snr_list, lock):
    with fsspec.open(h5, "rb") as fp:
        with h5py.File(fp, "r") as f:
            assert f["data"].shape[1] == 12000

            N = np.std(f["data"][:, n0:n1])
            S = np.std(f["data"][:, s0:s1])

            SNR = 10 * np.log10((S + 1e-10) / (N + 1e-10))
            # print(f"{h5}: {SNR:.3f}")
            with lock:
                snr_list.append([h5, SNR])


if __name__ == "__main__":
    # %%
    for folder in folders:
        result_path = Path(f"results/stats/{folder}")
        if not result_path.exists():
            result_path.mkdir(parents=True)

        h5_list = fs.glob(f"{bucket}/{folder}/data/*h5")
        with open(result_path / f"h5_list.txt", "w") as f:
            for i, h5 in enumerate(h5_list):
                f.write(f"{protocol}{h5}\n")
        print("Number of h5 files:", len(h5_list))

        manager = mp.Manager()
        snr_list = manager.list()
        lock = manager.Lock()
        ncpu = mp.cpu_count() * 2
        pbar = tqdm(total=len(h5_list))
        with mp.get_context("spawn").Pool(ncpu) as p:
            # p.starmap(get_snr, [(f"{protocol}{h5}", snr_list, lock) for h5 in h5_list])
            for h5 in h5_list:
                p.apply_async(get_snr, args=(f"{protocol}{h5}", snr_list, lock), callback=lambda x: pbar.update())
            p.close()
            p.join()
        snr_list = list(snr_list)
        with open(result_path / f"snr_list.txt", "w") as f:
            for snr in snr_list:
                f.write(f"{snr[0]},{snr[1]:.3f}\n")

        # # %%
        # snr_df = pd.read_csv(result_path / f"snr_list.txt", header=None, names=["h5", "snr"])
        # snr_df["event_id"] = snr_df["h5"].apply(lambda x: x.split("/")[-1].split(".")[0])
        # plt.figure()
        # plt.hist(snr_df["snr"], bins=100)
        # plt.yscale("log")
        # plt.xlabel("SNR (dB)")
        # plt.ylabel("Frequency")
        # plt.show()

        # # %%
        # for picker in pickers:
        #     # detected_event = fs.glob(f"{bucket}/{folder}/{picker}/picks/*csv")
        #     detected_event = fs.glob(f"{bucket}/{folder}/gamma/{picker}/picks/*csv")
        #     detected_event_id = [e.split("/")[-1].split("_")[-1].split(".")[0] for e in detected_event]
        #     snr_df[f"{picker}"] = snr_df["event_id"].apply(lambda x: x in detected_event_id)
        #     # raise

        # # %%
        # plt.figure()
        # bins = np.linspace(-4, 16, 20 + 1)
        # for picker in pickers:
        #     plt.hist(snr_df["snr"][snr_df[picker]], bins=bins, label=picker, alpha=0.5, edgecolor="white")
        # # plt.xlim()
        # plt.xticks(bins[::2])
        # plt.yscale("log")
        # plt.xlabel("SNR (dB)")
        # plt.ylabel("Frequency")


# %%
# %%
