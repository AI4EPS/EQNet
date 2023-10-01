# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Address all TODOs and remove all explanatory comments
# Lint as: python3
"""QuakeFlow_NC: A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format."""


import h5py
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import fsspec

import datasets
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import butter, filtfilt


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {NCEDC dataset for QuakeFlow},
author={Zhu et al.},
year={2023}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_REPO = "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/data"
_FILES = [
    "NC1970-1989.h5",
    "NC1990-1994.h5",
    "NC1995-1999.h5",
    "NC2000-2004.h5",
    "NC2005-2009.h5",
    "NC2010.h5",
    "NC2011.h5",
    "NC2012.h5",
    "NC2013.h5",
    "NC2014.h5",
    "NC2015.h5",
    "NC2016.h5",
    "NC2017.h5",
    "NC2018.h5",
    "NC2019.h5",
    "NC2020.h5",
]

_PATH = "/home/zhuwq/quakeflow_share/quakeflow_nc/data"
_NAMES = ["2016.h5", "2017.h5", "2018.h5", "2019.h5", "2020.h5"]#, "2021.h5", "2022.h5"]
_URLS = {
    "station": [f"{_REPO}/{x}" for x in _FILES],
    "event": [f"{_REPO}/{x}" for x in _FILES],
    #"event": [f"{_PATH}/{x}" for x in _NAMES],
    "station_train": [f"{_REPO}/{x}" for x in _FILES[:-1]],
    "event_train": [f"{_REPO}/{x}" for x in _FILES[:-1]],
    "station_test": [f"{_REPO}/{x}" for x in _FILES[-1:]],
    "event_test": [f"{_REPO}/{x}" for x in _FILES[-1:]],
    "event_large": ["/home/wanghy/tests/EQNet/datasets/large_network_test_15.h5"]
}


class BatchBuilderConfig(datasets.BuilderConfig):
    """
    yield a batch of event-based sample, so the number of sample stations can vary among batches
    Batch Config for QuakeFlow_NC
    :param batch_size: number of samples in a batch
    :param num_stations_list: possible number of stations in a batch
    """

    def __init__(self, batch_size: int, num_stations_list: List, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_stations_list = num_stations_list


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class QuakeFlow_NC(datasets.GeneratorBasedBuilder):
    """QuakeFlow_NC: A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format."""

    VERSION = datasets.Version("1.1.0")

    degree2km = 111.32
    nt = 8192
    feature_nt = 512
    feature_scale = int(nt / feature_nt)
    sampling_rate = 100.0

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    # default config, you can change batch_size and num_stations_list when use `datasets.load_dataset`
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="station", version=VERSION, description="yield station-based samples one by one of whole dataset"
        ),
        datasets.BuilderConfig(
            name="event", version=VERSION, description="yield event-based samples one by one of whole dataset"
        ),
        datasets.BuilderConfig(
            name="station_train",
            version=VERSION,
            description="yield station-based samples one by one of training dataset",
        ),
        datasets.BuilderConfig(
            name="event_train", version=VERSION, description="yield event-based samples one by one of training dataset"
        ),
        datasets.BuilderConfig(
            name="station_test", version=VERSION, description="yield station-based samples one by one of test dataset"
        ),
        datasets.BuilderConfig(
            name="event_test", version=VERSION, description="yield event-based samples one by one of test dataset"
        ),
        datasets.BuilderConfig(
            name="event_large", version=VERSION, description="yield event-based samples with 15+ stations one by one of test dataset"
        ),
        datasets.BuilderConfig(
            name="event_custom", version=VERSION, description="yield event-based samples one by one of custom dataset"
        ),
        datasets.BuilderConfig(
            name="station_custom", version=VERSION, description="yield station-based samples one by one of custom dataset"
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        "station_test"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if (
            (self.config.name == "station")
            or (self.config.name == "station_train")
            or (self.config.name == "station_test")
        ):
            features=datasets.Features(
                {
                    "data": datasets.Array2D(shape=(3, self.nt), dtype='float32'),
                    "phase_pick": datasets.Array2D(shape=(3, self.nt), dtype='float32'),
                    "event_location": datasets.Sequence(datasets.Value("float32")),
                    "station_location": datasets.Sequence(datasets.Value("float32")),
                })
            
        elif (
            (self.config.name == "event") 
            or (self.config.name == "event_train") 
            or (self.config.name == "event_test")
            or (self.config.name == "event_large")
        ):
            features=datasets.Features(
                {
                    "data": datasets.Array3D(shape=(None, 3, self.nt), dtype='float32'),
                    "phase_pick": datasets.Array3D(shape=(None, 3, self.nt), dtype='float32'),
                    "event_center" : datasets.Array2D(shape=(None, self.feature_nt), dtype='float32'),
                    "event_location": datasets.Array3D(shape=(None, 7, self.feature_nt), dtype='float32'),
                    "event_location_mask": datasets.Array2D(shape=(None, self.feature_nt), dtype='float32'),
                    "station_location": datasets.Array2D(shape=(None, 3), dtype="float32"),
                    "amplitude": datasets.Array3D(shape=(None, 3, self.nt), dtype="float32"),
                    # "reference_point": datasets.Sequence(datasets.Value("float32")),
                }
            )
            
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        if "custom" in self.config.name:
            urls = self.config.data_files
        else:
            urls = _URLS[self.config.name]
        # files = dl_manager.download(urls)
        files = dl_manager.download_and_extract(urls)
        print(files)

        if self.config.name == "station" or self.config.name == "event":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": files[:-1],
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": files[-1:], "split": "test"},
                ),
            ]
        elif self.config.name == "station_train" or self.config.name == "event_train":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": files,
                        "split": "train",
                    },
                ),
            ]
        elif self.config.name == "station_test" or self.config.name == "event_test":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": files, "split": "test"},
                ),
            ]
        elif self.config.name == "event_large":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": files, "split": "test"},
                ),
            ]
        elif self.config.name == "event_custom" or self.config.name == "station_custom":
            generator_list = []
            try:
                generator_list.append(
                    datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": files["train"], "split": "train"},
                    ))
            except:
                pass
            try:
                generator_list.append(
                    datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": files["test"], "split": "test"},
                    ))
            except:
                pass
            
            return generator_list
        else:
            raise ValueError("config.name is not in BUILDER_CONFIGS")

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        for file in filepath:
            with fsspec.open(file, "rb") as fs:
                with h5py.File(fs, "r") as fp:
                    # for event_id in sorted(list(fp.keys())):
                    event_ids = list(fp.keys())
                    for event_id in event_ids:
                        event = fp[event_id]
                        station_ids = list(event.keys())
                        if (
                            (self.config.name == "station")
                            or (self.config.name == "station_train")
                            or (self.config.name == "station_test")
                            or (self.config.name == "station_custom")
                        ):
                            waveforms = np.zeros([3, self.nt], dtype="float32")
                            phase_pick = np.zeros_like(waveforms)
                            attrs = event.attrs
                            event_location = [
                                attrs["longitude"],
                                attrs["latitude"],
                                attrs["depth_km"],
                                attrs["event_time_index"],
                            ]
                            
                            for i, sta_id in enumerate(station_ids):
                                waveforms[:, : self.nt] = event[sta_id][:, :self.nt]
                                # waveforms[:, : self.nt] = event[sta_id][: self.nt, :].T
                                attrs = event[sta_id].attrs
                                p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
                                s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
                                # phase_pick[:, :self.nt] = generate_label([p_picks, s_picks], nt=self.nt)
                                station_location = [attrs["longitude"], attrs["latitude"], -attrs["elevation_m"] / 1e3]

                                yield f"{event_id}/{sta_id}", {
                                    "data": torch.from_numpy(waveforms).float(),
                                    "phase_pick": torch.from_numpy(phase_pick).float(),
                                    "event_location": torch.from_numpy(np.array(event_location)).float(),
                                    "station_location": torch.from_numpy(np.array(station_location)).float(),
                                }


                        elif (
                            (self.config.name == "event")
                            or (self.config.name == "event_train")
                            or (self.config.name == "event_test")
                            or (self.config.name == "event_large")
                            or (self.config.name == "event_custom")
                        ):
                            event_attrs = event.attrs

                            # avoid stations with P arrival equals S arrival
                            is_sick = False
                            for sta_id in station_ids:
                                attrs = event[sta_id].attrs
                                # if len(attrs.keys()) <22:
                                #     is_sick = True
                                #     break
                                # try:
                                #     print(event_id)
                                #     p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
                                # except:
                                #     print(event_id, event_attrs)
                                #     print(sta_id, attrs)
                                #     raise
                                p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
                                s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
                                p_events = attrs["event_id"][attrs["phase_type"] == "P"]
                                s_events = attrs["event_id"][attrs["phase_type"] == "S"]
                                for p_pick, s_pick, p_event, s_event in zip(p_picks, s_picks, p_events, s_events):
                                    if p_pick>=s_pick and p_event==s_event:
                                        is_sick = True
                                        break
                                p_pick = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "P", attrs["event_id"] == event_attrs["event_id"])]
                                s_pick = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "S", attrs["event_id"] == event_attrs["event_id"])]
                                if ((p_pick) + (s_pick))[0] > self.nt * 2:
                                    is_sick = True
                                    break
                            if is_sick:
                                continue
                            
                            reference_latitude = 0
                            reference_longitude = 0
                            for sta_id in station_ids:
                                reference_latitude += event[sta_id].attrs["latitude"]
                                reference_longitude += event[sta_id].attrs["longitude"]
                            reference_latitude/=len(station_ids)
                            reference_longitude/=len(station_ids)
                            
                            b, a = butter(4, 0.1, btype="highpass", analog=False)
                            
                            waveforms = np.zeros([len(station_ids), 3, self.nt], dtype="float32")
                            amplitude = np.zeros_like(waveforms)
                            phase_pick = np.zeros_like(waveforms)
                            event_center = np.zeros([len(station_ids), self.feature_nt])
                            event_location = np.zeros([len(station_ids), 7, self.feature_nt])
                            event_location_mask = np.zeros([len(station_ids), self.feature_nt])
                            station_location = np.zeros([len(station_ids), 3])
                            # reference_point = np.array([reference_longitude, reference_latitude])

                            for i, sta_id in enumerate(station_ids):
                                # trace_id = event_id + "/" + sta_id
                                waveforms[i, :, :] = event[sta_id][:, :self.nt]
                                amplitude[i, :, :] = event[sta_id][:, :self.nt]
                                attrs = event[sta_id].attrs
                                p_picks = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "P", attrs["event_id"] == event_attrs["event_id"])]
                                s_picks = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "S", attrs["event_id"] == event_attrs["event_id"])]
                                phase_pick[i, :, :] = generate_label([p_picks, s_picks], nt=self.nt)
                                if attrs["unit"][-6:] == "m/s**2":
                                    # integrate acceleration to velocity
                                    amplitude[i] = np.cumsum(amplitude[i]*attrs["dt_s"], axis=-1)
                                    for j in range(3): 
                                        spline_i = LSQUnivariateSpline(np.arange(self.nt), amplitude[i, j, :], t=np.arange(self.nt, step=self.nt/2048)[1:], k=3)
                                        amplitude[i, j, :] -= spline_i(np.arange(self.nt))
                                    amplitude[i] = filtfilt(b, a, amplitude[i], axis=-1)
                                elif attrs["unit"][-3:] == "m/s":
                                    amplitude[i] = amplitude[i] * 10e4 #TODO: temp
                                    

                                ## TODO: how to deal with multiple phases
                                # center = (attrs["phase_index"][::2] + attrs["phase_index"][1::2])/2.0
                                ## assuming only one event with both P and S picks
                                assert len(p_picks)==len(s_picks), f'{event_id} {sta_id}: p_picks:{p_picks}, s_picks:{s_picks}'
                                c0 = ((p_picks) + (s_picks)) / 2.0 # phase center
                                c0_width = max(((s_picks - p_picks) * self.sampling_rate / 200.0).max(), 0)
                                # c0_width = ((s_picks - p_picks) * self.sampling_rate / 200.0).max() # min=160
                                assert c0_width>0
                                dx = round(
                                    (event_attrs["longitude"] - attrs["longitude"])
                                    * np.cos(np.radians(reference_latitude))
                                    * self.degree2km,
                                    2,
                                )
                                dy = round(
                                    (event_attrs["latitude"] - attrs["latitude"])
                                    * self.degree2km,
                                    2,
                                )
                                dz = round(
                                    event_attrs["depth_km"] + attrs["elevation_m"] / 1e3,
                                    2,
                                )
                                magnitude = round(event_attrs["magnitude"], 2)

                                assert c0[0]<self.nt
                                c0 = c0/self.feature_scale
                                assert c0[0]<self.feature_nt
                                c0_width = c0_width/self.feature_scale
                                #assert c0_width>=160/self.feature_scale
                                c0_int = c0.astype(np.int32)
                                assert c0_int[0]<self.feature_nt
                                assert abs(c0-c0_int)[0]<1
                                
                                event_center[i, :] = generate_label(
                                    [
                                        # [c0 / self.feature_scale],
                                        c0_int,
                                    ],
                                    label_width=[
                                        10,
                                    ],
                                    # label_width=[
                                    #     10,
                                    # ],
                                    nt=self.feature_nt,
                                    # nt=self.nt,
                                )[1, :]
                                mask = event_center[i, :] >= 0.5
                                event_location[i, 0, :] = (
                                    self.feature_scale * np.arange(self.feature_nt) - event_attrs["event_time_index"]
                                ) / self.sampling_rate
                                # event_location[0, :, i] = (np.arange(self.feature_nt) - 3000 / self.feature_scale) / self.sampling_rate
                                # print(event_location[i, 1:, mask].shape, event_location.shape, event_location[i][1:, mask].shape)
                                event_location[i][1:, mask] = np.array([dx, dy, dz, magnitude, (c0-c0_int)[0], c0_width])[:, np.newaxis]
                                event_location_mask[i, :] = mask

                                ## station location
                                station_location[i, 0] = round(
                                    (attrs["longitude"] - reference_longitude)
                                    * np.cos(np.radians(reference_latitude))
                                    * self.degree2km,
                                    2,
                                )
                                station_location[i, 1] = round((attrs["latitude"] - reference_latitude)
                                                               * self.degree2km, 2)
                                station_location[i, 2] =  round(-attrs["elevation_m"]/1e3, 2)

                            # std = np.std(waveforms, axis=-1, keepdims=True)
                            # std[std == 0] = 1.0
                            # waveforms = (waveforms - np.mean(waveforms, axis=-1, keepdims=True)) / std
                            # waveforms = waveforms.astype(np.float32)
                            
                            if (self.config.name == "event_large" or self.config.name == "event") and len(station_ids) > 5:
                                D = np.sqrt(((station_location[:, np.newaxis, :2] -  station_location[np.newaxis, :, :2])**2).sum(axis=-1))
                                Tcsr = minimum_spanning_tree(D)
                                index = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                                waveforms = waveforms[index]
                                amplitude = amplitude[index]
                                phase_pick = phase_pick[index]
                                event_center = event_center[index]
                                event_location = event_location[index]
                                event_location_mask = event_location_mask[index]
                                station_location = station_location[index]

                            yield event_id, {
                                "data": torch.from_numpy(waveforms).float(),
                                "phase_pick": torch.from_numpy(phase_pick).float(),
                                "event_center": torch.from_numpy(event_center).float(),
                                "event_location": torch.from_numpy(event_location).float(),
                                "event_location_mask": torch.from_numpy(event_location_mask).float(),
                                "station_location": torch.from_numpy(station_location).float(),
                                "amplitude": torch.from_numpy(amplitude).float(),
                                # "reference_point": torch.from_numpy(reference_point).float(),
                            }


def generate_label(phase_list, label_width=[150, 150], nt=8192):
    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)

    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.1] = 0.0
            target[i + 1, :] += gaussian

    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    return target