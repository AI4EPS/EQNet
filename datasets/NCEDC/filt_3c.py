# %%
import h5py
import numpy as np
from tqdm import tqdm

# %%
def calc_snr(waveform, picks, noise_window=300, signal_window=300, gap_window=50):

    noises = []
    signals = []
    snr = []
    for i in range(waveform.shape[0]):
        for j in picks:
            if j + gap_window < waveform.shape[1]:
                # noise = np.std(waveform[i, j - noise_window : j - gap_window])
                # signal = np.std(waveform[i, j + gap_window : j + signal_window])
                noise = np.max(np.abs(waveform[i, j - noise_window : j - gap_window]))
                signal = np.max(np.abs(waveform[i, j + gap_window : j + signal_window]))
                if (noise > 0) and (signal > 0):
                    signals.append(signal)
                    noises.append(noise)
                    snr.append(signal / noise)
                else:
                    signals.append(0)
                    noises.append(0)
                    snr.append(0)

    # if len(snr) == 0:
    #     return 0.0, 0.0, 0.0
    # else:
    #     return snr[-1], signals[-1], noises[-1]
    return snr


# %%
h5_in = "ncedc_event_dataset.h5"
h5_out = "ncedc_event_dataset_3c.h5"
num = 0
with h5py.File(h5_out, "w") as fp_out:
    with h5py.File(h5_in, "r") as fp_in:
        for event_id, event in tqdm(fp_in.items()):
            # print(event_id, event.attrs["num_stations"])
            # print(dict(event.attrs))
            for station_id, station in event.items():

                # print(station_id, station.shape)
                # print(dict(station.attrs))
                waveform = station[()].T
                # print(waveform.shape)

                P_index = station.attrs["phase_index"][station.attrs["phase_type"] == "P"]
                # S_index = station.attrs["phase_index"][station.attrs["phase_type"] == "S"]
                SNR = calc_snr(waveform, P_index)
                # S_SNR = calc_snr(waveform, S_index)

                if len(SNR) >= 3:
                    if (np.all(np.array(SNR) > 0)) and (np.max(np.array(SNR)) > 2.0):
                        if event_id not in fp_out:
                            fp_out.create_group(event_id)
                            for key, value in event.attrs.items():
                                fp_out[event_id].attrs[key] = value
                        if station_id not in fp_out[event_id]:
                            fp_out[event_id].create_dataset(station_id, data=waveform)
                            for key, value in station.attrs.items():
                                fp_out[event_id][station_id].attrs[key] = value
                            fp_out[event_id][station_id].attrs["snr"] = SNR
                            num += 1
                        else:
                            print("Duplicate station", event_id, station_id)

                #     print("3C", SNR)
                #     # continue
                #     raise
                # else:
                #     print("xC", SNR)
                #     continue
print(f"Total number of samples: {num}")

# %%
