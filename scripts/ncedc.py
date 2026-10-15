# %%
import json
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from glob import glob

import fsspec
import obspy
from tqdm import tqdm

# %%
protocol = "s3"
bucket = "ncedc-pds"
folder = "continuous_waveforms"
fs = fsspec.filesystem(protocol=protocol, anon=True)

# %%
valid_channels = ["3", "2", "1", "E", "N", "Z"]
valid_instruments = ["BH", "HH", "EH", "HN", "DP"]

# %%
mseed_dir = "mseed_list"
if not os.path.exists(f"{mseed_dir}"):
    os.makedirs(f"{mseed_dir}")

for year in range(2023, 2024):
    networks = fs.glob(f"{bucket}/{folder}/*")
    for i, network in enumerate(tqdm(networks)):
        mseed_list = []
        # years = fs.glob(f"{network}/????")
        # for year in years:
        jdays = fs.glob(f"{network}/{year}/????.???")
        for jday in jdays:
            mseeds = fs.glob(f"{jday}/*.{jday.split('/')[-1]}")
            mseed_list.extend(mseeds)

        mseed_list = sorted([f"{protocol}://{mseed}" for mseed in mseed_list])
        if len(mseed_list) > 0:
            with open(f"{mseed_dir}/{year}_{network.split('/')[-1]}.txt", "w") as fp:
                fp.write("\n".join(mseed_list))

        groups = defaultdict(list)
        for mseed in mseed_list:
            tmp = mseed.split(".")
            if (tmp[3][-1] in valid_channels) and (tmp[3][:2] in valid_instruments):
                key = ".".join(tmp[:3]) + "." + tmp[3][:-1] + "." + ".".join(tmp[4:])
                groups[key].append(mseed)
            # else:
            #     print(f"Invalid channel: {mseed}")

        if len(groups) > 0:
            with open(f"{mseed_dir}/{year}_{network.split('/')[-1]}_3c.txt", "w") as fp:
                keys = sorted(groups.keys())
                for key in keys:
                    fp.write(",".join(sorted(groups[key])) + "\n")

# %% TESTING

with open("mseed_list/2023_NC.txt", "r") as fp:
    lines = fp.read().splitlines()


def read_mseed(line):
    # for i, line in enumerate(tqdm(lines)):
    with fsspec.open(line, "rb") as fp:
        st = obspy.read(fp)
    return st


time0 = time.time()
MAX_THREADS = 8
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = set()
    for i, line in enumerate(tqdm(lines)):
        thread = executor.submit(read_mseed, line)
        futures.add(thread)
        if len(futures) >= MAX_THREADS:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
