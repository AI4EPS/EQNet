# data augmentation functions for huggingface datasets
import os
import random
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import datasets
import torch
from collections import defaultdict
from datetime import timedelta, datetime
from scipy import signal


def _get_augmentations():
    """Returns a list of data augmentation functions."""
    pass


def random_shift(example, shift_range=(-160, 16), feature_scale=16):
    feature_shift = np.random.randint(*shift_range)
    shift = feature_shift * feature_scale
    # shift at time axis
    example["data"] = torch.roll(example["data"], shift, dims=-2).contiguous()
    example["phase_pick"] = torch.roll(example["phase_pick"], shift, dims=-2).contiguous()
    example["event_center"] = torch.roll(example["event_center"], feature_shift, dims=-2).contiguous()
    example["event_location"] = torch.roll(example["event_location"], feature_shift, dims=-2).contiguous()
    example["event_location_mask"] = torch.roll(example["event_location_mask"], feature_shift, dims=-2).contiguous()
    
    return example
    

def stack_event(
    meta1,
    meta2,
    max_shift=1024 * 4,
):
    pass


def cut_data(meta, nt=1024 * 4, min_point=200):
    pass


def drop_channel(meta):
    pass


def calc_snr(self, waveform, picks, noise_window=300, signal_window=300, gap_window=50):
    noises = []
    signals = []
    snr = []

    for i in range(waveform.shape[0]):
        for j in picks:
            if (j - gap_window > 0) and (j + gap_window < waveform.shape[1]):
                # noise = np.std(waveform[i, j - noise_window : j - gap_window])
                # signal = np.std(waveform[i, j + gap_window : j + signal_window])
                noise = np.max(np.abs(waveform[i, max(0, j - noise_window) : j - gap_window]))
                signal = np.max(np.abs(waveform[i, j + gap_window : j + signal_window]))
                if (noise > 0) and (signal > 0):
                    signals.append(signal)
                    noises.append(noise)
                    snr.append(signal / noise)
                else:
                    signals.append(0)
                    noises.append(0)
                    snr.append(0)

    if len(snr) == 0:
        return 0.0, 0.0, 0.0
    else:
        # return snr[-1], signals[-1], noises[-1]
        return np.max(snr), np.max(signals), np.max(noises)
    # else:
    # idx = np.argmax(snr).item()
    # return snr[idx], signals[idx], noises[idx]

    # def resample_time(self, waveform, picks, factor=1.0):
    #     nch, nt = waveform.shape
    #     scale_factor = random.uniform(min(1, factor), max(1, factor))
    #     with torch.no_grad():
    #         data_ = F.interpolate(data.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
    #         if noise is not None:
    #             noise_ = F.interpolate(noise.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
    #         else:
    #             noise_ = None
    #     picks_ = []
    #     for phase in picks:
    #         tmp = []
    #         for p in phase:
    #             tmp.append([p[0], p[1] * scale_factor])
    #         picks_.append(tmp)
    #     return data_, picks_, noise_

def taper(stream):
    for tr in stream:
        tr.taper(max_percentage=0.05, type="cosine")
    return stream
    
