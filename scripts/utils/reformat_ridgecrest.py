# %%
import os
from pathlib import Path
import fsspec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm.auto import tqdm
import multiprocessing as mp
from datetime import datetime, timedelta

# %%

def run(catalog, id):

    # %%
    protocol = "gs"
    bucket = "gs://quakeflow_das"

    # %%
    fs = fsspec.filesystem(protocol)
    folder = "ridgecrest_old"
    new_folder = "ridgecrest_north"

    tmp_folder = f"tmp/{id:03d}/"
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    i = 1
    missing = 0
    for _, row in tqdm(catalog.iterrows(), total=len(catalog), position=id):
        # print(i, row)

        if not fs.exists(f"{bucket}/{folder}/data/{row['event_id'][2:]}.h5"):
            # print(f"{bucket}/{folder}/data/{row['event_id'][2:]}.h5 not found")
            missing += 1
            continue
    
        if fs.exists(f"{bucket}/{new_folder}/data/{row['event_id']}.h5"):
            continue
        else:
            print(f"{bucket}/{new_folder}/data/{row['event_id']}.h5 not exist")

        try:
            f1 = h5py.File(fs.open(f"{bucket}/{folder}/data/{row['event_id'][2:]}.h5", "rb"), "r")
        # with h5py.File(fs.open(f"{bucket}/{folder}/data/{row['event_id'][2:]}.h5", "rb"), "r") as f1:
            # print(f1["data"].shape)
            # for k, v in f1["data"].attrs.items():
                # print(k, v)

            with h5py.File(f"{tmp_folder}/{row['event_id']}.h5", "w") as f2:
                f2["data"] = f1["data"][:].T
                for k, v in f1["data"].attrs.items():
                    if k == "event_id":
                        v = row["event_id"]
                    if k == "source":
                        v = "ci"
                    if k == "das_array":
                        continue
                    f2["data"].attrs[k] = v

                begin_time = datetime.fromisoformat(f1["data"].attrs["begin_time"])
                begin_time = begin_time.replace(tzinfo=None)
                event_time = datetime.fromisoformat(f1["data"].attrs["event_time"])
                event_time = event_time.replace(tzinfo=None)
                event_time_index = int((event_time - begin_time).total_seconds() * 100)
                unit = "microstrain/s"
                f2["data"].attrs["event_time_index"] = event_time_index
                f2["data"].attrs["unit"] = unit
        except Exception as e:
            print(e)
            continue


        if i % 4 == 0:
            # print(f"Uploading {i}")
            os.system(f"gsutil mv -r {tmp_folder}/*.h5 {bucket}/{new_folder}/data/ >/dev/null 2>&1")
            # os.system(f"gsutil cp -r {tmp_folder}/*.h5 {bucket}/{new_folder}/data/")

        i += 1
    os.system(f"gsutil mv -r {tmp_folder}/*.h5 {bucket}/{new_folder}/data/ >/dev/null 2>&1")
    # os.system(f"gsutil cp -r {tmp_folder}/*.h5 {bucket}/{new_folder}/data/")

    print(f"Missing {missing} events")
# %%

if __name__ == "__main__":

    # %%
    protocol = "gs"
    bucket = "gs://quakeflow_das"

    # %%
    fs = fsspec.filesystem(protocol)
    folder = "ridgecrest_old"
    new_folder = "ridgecrest_north"

    # %%
    catalog = pd.read_csv(f"{bucket}/{folder}/catalog.csv")

    # %%
    das_info = pd.read_csv(f"{bucket}/{folder}/das_info.csv")

    # %%
    lat0 = das_info["latitude"].mean()
    lon0 = das_info["longitude"].mean()
    depth0 = -das_info["elevation_m"].mean()

    # %%
    deg2km = 111.1949
    catalog["distance_km"] = catalog.apply(lambda x: (((x["latitude"] - lat0) * deg2km) ** 2 + ((x["longitude"] - lon0) * deg2km) ** 2 + (x["depth_km"] - depth0)**2) ** 0.5, axis=1)
    # %%
    plt.figure()
    plt.scatter(catalog["magnitude"], catalog["distance_km"], s=1)
    plt.figure()
    plt.scatter(catalog["longitude"], catalog["latitude"], s=1)

    # %%
    def calc_detectable_distance(magnitudes):
        scaling = {
            "detectable_amplitude": -1.9,
            # "detectable_amplitude": -1,
            # "detectable_amplitude": 0.0,
            "mean_site_term_S": 0.4,
            "mean_site_term_P": 0.4,
        }
        M_coef = (0.437, 0.69)
        D_coef = (-1.2693, -1.5875)

        detectable_amplitude = 10 ** scaling["detectable_amplitude"]
        mean_site_term_P = 10 ** scaling["mean_site_term_P"]
        mean_site_term_S = 10 ** scaling["mean_site_term_S"]

        D_sense_P = 10 ** (
            (-magnitudes * M_coef[0] + (np.log10(detectable_amplitude) - np.log10(mean_site_term_P))) / D_coef[0]
        )
        D_sense_S = 10 ** (
            (-magnitudes * M_coef[1] + (np.log10(detectable_amplitude) - np.log10(mean_site_term_S))) / D_coef[1]
        )

        return D_sense_S

    detectable_distance_km = calc_detectable_distance(catalog["magnitude"].values)
    catalog["detectable_distance_km"] = detectable_distance_km

    # %%
    idx = catalog["distance_km"] < catalog["detectable_distance_km"]
    plt.figure()
    plt.scatter(catalog[idx]["magnitude"], catalog[idx]["distance_km"],  s=1)
    plt.figure()
    plt.scatter(catalog[idx]["longitude"], catalog[idx]["latitude"], s=1)
    # %%
    print(f"Number of events: {len(catalog)}")
    print(f"Number of detectable events: {len(catalog[idx])}")

    # %%
    catalog_data = catalog[idx]
    catalog_data["event_id"] = catalog_data["event_id"].apply(lambda x: f"ci{x}")


    # %%
    # catalog_data.to_csv(f"catalog_data.csv", index=False)
    # os.system(f"gsutil cp catalog_data.csv {bucket}/{new_folder}/catalog_data.csv")

    # # %%
    # os.system(f"gsutil cp {bucket}/{folder}/das_info.csv {bucket}/{new_folder}/das_info.csv")

    # # %%
    # os.system(f"gsutil cp {bucket}/{folder}/meta.txt {bucket}/{new_folder}/meta.txt")

    # %%

    num_cpu = mp.cpu_count() -1
    # num_cpu = 1
    print(f"Number of CPUs: {num_cpu}")
    with mp.get_context("spawn").Pool(num_cpu) as pool:
        pool.starmap(run, [(catalog_data.iloc[i::num_cpu], i) for i in range(num_cpu)])
    