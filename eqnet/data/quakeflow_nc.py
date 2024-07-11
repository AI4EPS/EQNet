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
from datetime import datetime, timedelta

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
_REPO = "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/waveform_h5"
_FILES = [
    "1987.h5",
    "1988.h5",
    "1989.h5",
    "1990.h5",
    "1991.h5",
    "1992.h5",
    "1993.h5",
    "1994.h5",
    "1995.h5",
    "1996.h5",
    "1997.h5",
    "1998.h5",
    "1999.h5",
    "2000.h5",
    "2001.h5",
    "2002.h5",
    "2003.h5",
    "2004.h5",
    "2005.h5",
    "2006.h5",
    "2007.h5",
    "2008.h5",
    "2009.h5",
    "2010.h5",
    "2011.h5",
    "2012.h5",
    "2013.h5",
    "2014.h5",
    "2015.h5",
    "2016.h5",
    "2017.h5",
    "2018.h5",
    "2019.h5",
    "2020.h5",
    "2021.h5",
    "2022.h5",
    "2023.h5",
]

_PATH = "/nfs/quakeflow_dataset/NC/waveform_h5"
_NAMES = ["2014.h5", "2015.h5", "2016.h5", "2017.h5", "2018.h5", "2019.h5", "2020.h5"]#, "2021.h5", "2022.h5"]
_URLS = {
    "station": [f"{_REPO}/{x}" for x in _FILES],
    #"event": [f"{_REPO}/{x}" for x in _FILES],
    "event": [f"{_PATH}/{x}" for x in _NAMES],
    "station_train": [f"{_REPO}/{x}" for x in _FILES[:-1]],
    "event_train": [f"{_REPO}/{x}" for x in _FILES[:-1]],
    "station_test": [f"{_REPO}/{x}" for x in _FILES[-1:]],
    "event_test": [f"{_REPO}/{x}" for x in _FILES[-1:]],
    #"event_large": ["/home/wanghy/tests/EQNet/datasets/large_network_test_15.h5"]
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
    raw_nt = 12000 # temporary hard code
    feature_scale = int(nt / feature_nt)
    raw_feature_nt = int(raw_nt / feature_scale)
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
        "event"  # It's not mandatory to have a default configuration. Just use one if it make sense.
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
                    "num_stations": datasets.Value("int32"),
                    "num_manual_stations": datasets.Value("int32"),
                    # "reference_point": datasets.Sequence(datasets.Value("float32")),
                }
            )
        elif (
            self.config.name == "event_test"
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
                    "num_stations": datasets.Value("int32"),
                    "num_manual_stations": datasets.Value("int32"),
                    "file_name": datasets.Value("string"),
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
                    gen_kwargs={
                        "filepath": files[-1:], 
                        "split": "test"
                    },
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
                        
                        # exclude ridgecrest earthquake sequence
                        event_time0 = datetime.fromisoformat(event.attrs["event_time"])
                        if event_time0 >= datetime(2019,7,4,0,0,0) and event_time0 <= datetime(2019,7,10,0,0,0):
                            if event.attrs["latitude"] >= 35.2 and event.attrs["latitude"] <= 36.2 and event.attrs["longitude"] >= -118.0 and event.attrs["longitude"] <= -117.0:
                                continue
                        
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
                            auto_count = 0
                            for sta_id in station_ids:
                                if len(np.unique(event[sta_id].attrs["phase_type"])) < 2:
                                    auto_count += 1
                                    continue # only one phase type indicates automatic picks
                                valid_manual_events = 0
                                attrs = event[sta_id].attrs
                                for e_id in [event_id]:#sorted(np.unique(event[sta_id].attrs["event_id"])):
                                    if len(np.unique(attrs["phase_type"][attrs["event_id"]==e_id])) < 2:
                                        continue
                                    
                                    p_picks = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "P", attrs["event_id"] == e_id)]
                                    s_picks = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "S", attrs["event_id"] == e_id)]
                                    if p_picks.min()>self.nt or s_picks.min()>self.nt:
                                        continue # phases too late
                                    # try:
                                    #     if len(p_picks)!=len(s_picks):
                                    #         is_sick = True
                                    #         break
                                    # except:
                                    #     pass
                                    assert len(p_picks)==len(s_picks), f"{event_id}/{sta_id} manual P&S picks num don't match"
                                    if np.any(p_picks+5>=s_picks):
                                        print(f"\n{event_id} when {sta_id} during {e_id} p pick arrives after s pick")
                                        is_sick = True
                                        break
                                    valid_manual_events += 1
                                if is_sick:
                                    break    
                                if valid_manual_events == 0:
                                    auto_count += 1
                            if auto_count >= len(station_ids):
                                is_sick = True
                                print(f"\nevent {event_id} all auto picks")
                            if is_sick:
                                continue
                            
                            
                            b, a = butter(4, 0.1, btype="highpass", analog=False)
                            
                            waveforms = np.zeros([len(station_ids), 3, self.raw_nt], dtype="float32")
                            amplitude = np.zeros_like(waveforms)
                            phase_pick = np.zeros_like(waveforms)
                            event_center = np.zeros([len(station_ids), self.raw_feature_nt])
                            event_location = np.zeros([len(station_ids), 7, self.raw_feature_nt])
                            event_location_mask = np.zeros([len(station_ids), self.raw_feature_nt])
                            station_location = np.zeros([len(station_ids), 3])
                            # reference_point = np.array([reference_longitude, reference_latitude])
                            manual_stas = [] # index of stations with at least one manual pick pair 
                            auto_stas = [] # index of stations with only automatic picks

                            for i, sta_id in enumerate(station_ids):
                                waveforms[i, :, :] = event[sta_id][:, :]
                                amplitude[i, :, :] = event[sta_id][:, :]
                                attrs = event[sta_id].attrs
                                if attrs["unit"][-6:] == "m/s**2":
                                    # integrate acceleration to velocity
                                    amplitude[i] = np.cumsum(amplitude[i]*attrs["dt_s"], axis=-1)
                                # time cost too much, so we only use highpass filter
                                # for j in range(3): 
                                #     spline_i = LSQUnivariateSpline(np.arange(self.nt), amplitude[i, j, :], t=np.arange(self.nt, step=self.nt/2048)[1:], k=3)
                                #     amplitude[i, j, :] -= spline_i(np.arange(self.nt))
                                amplitude[i] = filtfilt(b, a, amplitude[i], axis=-1)
                                
                                ## station location
                                # cartesian coordinate
                                # station_location[i, 0] = round(
                                #     (attrs["longitude"] - reference_longitude)
                                #     * np.cos(np.radians(reference_latitude))
                                #     * self.degree2km,
                                #     4,
                                # )
                                # station_location[i, 1] = round((attrs["latitude"] - reference_latitude)
                                #                                * self.degree2km, 4)
                                # polar coordinate
                                station_location[i, 0] = round(attrs["longitude"], 4)
                                station_location[i, 1] = round(attrs["latitude"], 4)

                                station_location[i, 2] =  round(-attrs["elevation_m"]/1e3, 2)
                                
                                if len(np.unique(attrs["phase_type"])) < 2:
                                    auto_stas.append(i)
                                    phase_pick[i, 1:, :] = 0
                                    phase_pick[i, 0, :] = 1
                                    event_center[i, :] = 0
                                    event_location[i, :, :] = 0
                                    event_location_mask[i, :] = 0
                                    continue
                                
                                manual_event_ids = []
                                p_pick_stas = []
                                s_pick_stas = []
                                
                                event_time0 = datetime.fromisoformat(event_attrs["event_time"])
                                for e_id in sorted(np.unique(attrs["event_id"])):
                                    e_attrs = fp[e_id].attrs
                                    if len(np.unique(attrs["phase_type"][attrs["event_id"]==e_id])) < 2:
                                        continue
                                    
                                    p_picks = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "P", attrs["event_id"] == e_id)]
                                    s_picks = attrs["phase_index"][np.logical_and(attrs["phase_type"] == "S", attrs["event_id"] == e_id)]
                                    if p_picks.min()>self.nt or s_picks.min()>self.nt:
                                        continue # phases too late
                                    if len(p_pick_stas)!=0 and np.abs(p_picks-np.array(p_pick_stas)).min() < 30:
                                        continue
                                    if len(s_pick_stas)!=0 and np.abs(s_picks-np.array(s_pick_stas)).min() < 30:
                                        continue

                                    ## TODO: how to deal with multiple phases
                                    # center = (attrs["phase_index"][::2] + attrs["phase_index"][1::2])/2.0
                                    ## assuming only one event with both P and S picks
                                    assert len(p_picks)==len(s_picks), f'{event_id} {sta_id}: p_picks:{p_picks}, s_picks:{s_picks}'
                                    c0 = ((p_picks) + (s_picks)) / 2.0 # phase center
                                    c0_width = max(((s_picks - p_picks) * self.sampling_rate / 200.0).max(), 10)
                                    # c0_width = ((s_picks - p_picks) * self.sampling_rate / 200.0).max() # min=160
                                    assert c0_width>0
                                    # cartesian
                                    # dx = round(
                                    #     (e_attrs["longitude"] - attrs["longitude"])
                                    #     * np.cos(np.radians(reference_latitude))
                                    #     * self.degree2km,
                                    #     4,
                                    # )
                                    # dy = round(
                                    #     (e_attrs["latitude"] - attrs["latitude"])
                                    #     * self.degree2km,
                                    #     4,
                                    # )
                                    dz = round(
                                        e_attrs["depth_km"] + attrs["elevation_m"] / 1e3,
                                        2,
                                    )
                                    # dist = np.sqrt(dx**2 + dy**2)
                                    # azimuth = np.arctan2(dy, dx)
                                    dlon = e_attrs["longitude"] - attrs["longitude"]
                                    dlat = e_attrs["latitude"] - attrs["latitude"]
                                    a = np.sin(np.radians(dlat) / 2) ** 2 \
                                        + np.cos(np.radians(attrs["latitude"])) * np.cos(np.radians(e_attrs["latitude"])) * np.sin(np.radians(dlon) / 2) ** 2
                                    dist = round(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 6371, 4)
                                    y = np.sin(np.radians(dlon)) * np.cos(np.radians(e_attrs["latitude"]))
                                    x = np.cos(np.radians(attrs["latitude"])) * np.sin(np.radians(e_attrs["latitude"])) \
                                        - np.sin(np.radians(attrs["latitude"])) * np.cos(np.radians(e_attrs["latitude"])) * np.cos(np.radians(dlon))
                                    azimuth = round(np.arctan2(y, x), 4)
                                    
                                    magnitude = round(e_attrs["magnitude"], 2)
                                    
                                    begin_time_index_shift = (int(round(
                                        (datetime.fromisoformat(e_attrs["event_time"]) - event_time0).total_seconds()
                                        / attrs["dt_s"]
                                    )) if e_id != event_id else 0
                                    )
                                    t0 = e_attrs["event_time_index"] + begin_time_index_shift

                                    assert len(c0)==1, f"{event_id}/{sta_id} multiple phase centers at {e_id}: {c0}"
                                    assert c0[0]<self.nt
                                    c0 = c0/self.feature_scale
                                    assert c0[0]<self.feature_nt
                                    c0_width = c0_width/self.feature_scale
                                    #assert c0_width>=160/self.feature_scale
                                    c0_int = c0.astype(np.int32)
                                    assert c0_int[0]<self.feature_nt
                                    assert abs(c0-c0_int)[0]<1

                                    event_center_eid = generate_label(
                                        [
                                            # [c0 / self.feature_scale],
                                            c0_int,
                                        ],
                                        # label_width=[
                                        #     c0_width,
                                        # ],
                                        label_width=[
                                            10,
                                        ],
                                        nt=self.raw_feature_nt,
                                        # nt=self.nt,
                                    )[1, :]
                                    #mask = event_center_eid >= 0.5
                                    mask = np.zeros_like(event_center_eid, dtype=bool)
                                    mask[max(0, c0_int[0]-4):min(self.feature_nt, c0_int[0]+4)] = True
                                    #print(f"{event_id}/{sta_id}: {e_id} c0:{c0}, c0_int:{c0_int}, mask:{np.sum(mask)}")
                                    assert np.max(event_center[i, :])<=1, f"{event_id}/{sta_id} event_center overlap when {e_id}:{event_center[i, :]}"
                                    if not np.all(event_location[i, :, mask]==0):
                                        print(f"\n{event_id}/{sta_id} event_location overlap when {e_id}: previous {np.nonzero(event_location_mask[i, :])}, current {np.nonzero(mask)}")
                                        continue
                                    
                                    event_center[i, :] += event_center_eid
                                    event_location[i, 0, max(0, c0_int[0]-4):min(self.feature_nt, c0_int[0]+4)] = ((
                                        self.feature_scale * np.arange(self.raw_feature_nt) - t0
                                    )/self.sampling_rate * mask.astype(np.float32))[max(0, c0_int[0]-4):min(self.feature_nt, c0_int[0]+4)]
                                    # print(event_location[i, 1:, mask].shape, event_location.shape, event_location[i][1:, mask].shape)
                                    event_location[i][1:, mask] += np.array([dist, azimuth, dz, magnitude, (c0-c0_int)[0], c0_width])[:, np.newaxis]
                                    event_location_mask[i, :] = np.logical_or(event_location_mask[i, :], mask)
                                    
                                    manual_event_ids.append(e_id)
                                    p_pick_stas.append(p_picks)
                                    s_pick_stas.append(s_picks)
                                    
                                if len(manual_event_ids) == 0:
                                    auto_stas.append(i)
                                    phase_pick[i, 1:, :] = 0
                                    phase_pick[i, 0, :] = 1
                                    event_center[i, :] = 0
                                    event_location[i, :, :] = 0
                                    event_location_mask[i, :] = 0
                                else:
                                    manual_stas.append(i)
                                    phase_pick[i, :, :] = generate_label([np.concatenate(p_pick_stas), np.concatenate(s_pick_stas)], nt=self.raw_nt)

                            # std = np.std(waveforms, axis=-1, keepdims=True)
                            # std[std == 0] = 1.0
                            # waveforms = (waveforms - np.mean(waveforms, axis=-1, keepdims=True)) / std
                            # waveforms = waveforms.astype(np.float32)
                            len_manual = len(manual_stas)
                            assert len_manual>0, f"event {event_id} no manual picks"
                            assert len_manual==np.any(event_location_mask>0, axis=1).sum(), f"{event_id} manual picks num don't match, len_manual:{len_manual}, manual_stas:{manual_stas}, event_location_mask:{np.any(event_location_mask>0, axis=1)}"
                            manual_cluster_index = np.array(manual_stas+auto_stas)
                            meta, it = cut_data({
                                "waveforms": waveforms,
                                "amplitude": amplitude,
                                "phase_pick": phase_pick,
                                "event_center": event_center,
                                "event_location": event_location,
                                "event_location_mask": event_location_mask,
                                "num_manual_stations": len_manual,
                            }, nt=self.nt, feature_scale=self.feature_scale, if_reverse_roll=False)
                            if abs(it) <= 160:
                                print(f"\n{event_id} data cut off shift smaller than {160/self.sampling_rate} s: {it/self.sampling_rate} s")
                            waveforms = meta["waveforms"][manual_cluster_index]
                            amplitude = meta["amplitude"][manual_cluster_index]
                            phase_pick = meta["phase_pick"][manual_cluster_index]
                            event_center = meta["event_center"][manual_cluster_index]
                            event_location = meta["event_location"][manual_cluster_index]
                            event_location_mask = meta["event_location_mask"][manual_cluster_index]
                            station_location = station_location[manual_cluster_index]
                            
                            if (self.config.name == "event_large" or self.config.name == "event") and len(station_ids) > 10:
                                D = np.sqrt(((station_location[:len_manual, np.newaxis, :2] -  station_location[np.newaxis, :len_manual, :2])**2).sum(axis=-1))
                                Tcsr = minimum_spanning_tree(D)
                                index_manual = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                                waveforms[:len_manual] = waveforms[index_manual]
                                amplitude[:len_manual] = amplitude[index_manual]
                                phase_pick[:len_manual] = phase_pick[index_manual]
                                event_center[:len_manual] = event_center[index_manual]
                                event_location[:len_manual] = event_location[index_manual]
                                event_location_mask[:len_manual] = event_location_mask[index_manual]
                                station_location[:len_manual] = station_location[index_manual]
                                
                                if len_manual <= 15 and len(station_ids) > 20:
                                    cut_off = 20
                                    D = np.sqrt(((station_location[len_manual:cut_off, np.newaxis, :2] -  station_location[np.newaxis, len_manual:cut_off, :2])**2).sum(axis=-1))
                                    Tcsr = minimum_spanning_tree(D)
                                    index_auto = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                                    waveforms[len_manual:cut_off] = waveforms[len_manual+index_auto]
                                    amplitude[len_manual:cut_off] = amplitude[len_manual+index_auto]
                                    phase_pick[len_manual:cut_off] = phase_pick[len_manual+index_auto]
                                    event_center[len_manual:cut_off] = event_center[len_manual+index_auto]
                                    event_location[len_manual:cut_off] = event_location[len_manual+index_auto]
                                    event_location_mask[len_manual:cut_off] = event_location_mask[len_manual+index_auto]
                                    station_location[len_manual:cut_off] = station_location[len_manual+index_auto]

                            assert np.any(phase_pick[:, 1:, :]>0), f"{event_id} no manual phase picks"
                            assert np.any(event_center>0), f"{event_id} no manual event picks"
                            meta = {
                                "data": torch.from_numpy(waveforms).float(),
                                "phase_pick": torch.from_numpy(phase_pick).float(),
                                "event_center": torch.from_numpy(event_center).float(),
                                "event_location": torch.from_numpy(event_location).float(),
                                "event_location_mask": torch.from_numpy(event_location_mask).float(),
                                "station_location": torch.from_numpy(station_location).float(),
                                "amplitude": torch.from_numpy(amplitude).float(),
                                "num_stations": waveforms.shape[0],
                                "num_manual_stations": len_manual,
                                # "reference_point": torch.from_numpy(reference_point).float(),
                            }
                            
                            if (self.config.name == "event_test"):
                                meta["file_name"] = event_id
                            
                            yield event_id, meta


def generate_label(phase_list, label_width=[150, 150], nt=8192):
    target = np.zeros([len(phase_list) + 1, nt], dtype=np.float32)

    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for phase_time in picks:
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.05] = 0.0
            target[i + 1, :] += gaussian

    target[0:1, :] = np.maximum(0, 1 - np.sum(target[1:, :], axis=0, keepdims=True))

    return target

def generate_event_label(
    reference_center,
    event_param,
    label_width=[100],
    nt=8192,
    mask_width=None,
):
    target_center = np.zeros(nt, dtype=np.float32)
    target_param = np.zeros([len(event_param[0]), nt], dtype=np.float32)
    mask = np.zeros(nt, dtype=np.float32)

    if len(label_width) == 1:
        label_width = label_width * len(reference_center)
    if mask_width is None:
        mask_width = [int(x * 1.5) for x in label_width]
    else:
        mask_width = [min(int(x * 1.5), label_width) for x in label_width]

    for i, (c0, t0, w, m) in enumerate(zip(reference_center, event_param, label_width, mask_width)):
        t = np.arange(nt) - c0
        gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
        gaussian[gaussian < 0.05] = 0.0
        target_center += gaussian
        mask[max(0, int(c0) - m) : min(nt, int(c0) + m)] = 1.0
        t = np.arange(nt) - t0
        target_param[max(0, int(c0) - m) : min(nt, int(c0) + m)] = t[max(0, int(c0) - m) : min(nt, int(c0) + m)]

    return target_center, target_param, mask

def cut_data(meta, nt=8192, nt0=2560, feature_scale=16, if_reverse_roll=False):

    tries = 0
    max_tries = 100
    phase_mask = meta["phase_pick"].copy().sum(axis=0)
    event_mask = meta["event_location_mask"].copy().sum(axis=0)
    p_start = np.nonzero(phase_mask[1, :])[0].min() // feature_scale
    event_start = np.nonzero(event_mask[:])[0].min()
    if not if_reverse_roll:
        rand_lower = 0
    else:
        s_end = np.nonzero(phase_mask[2, :])[0].max() // feature_scale
        event_end = np.nonzero(event_mask[:])[0].max()
        rand_lower = -max(100, nt//feature_scale-10-max(s_end, event_end))
    rand_upper = max(1, min(nt0//feature_scale, min(p_start, event_start)-5))
    raw_rand_upper = max(1, nt0//feature_scale)
    it = np.random.randint(rand_lower, raw_rand_upper)
    
    p_points = np.nonzero(phase_mask[1, :nt])[0].shape[0]
    s_points = np.nonzero(phase_mask[2, :nt])[0].shape[0]
    event_points = np.nonzero(event_mask[:nt//feature_scale])[0].shape[0]
    tmp_phase = np.roll(phase_mask, -it*feature_scale, axis=-1)
    tmp_event = np.roll(event_mask, -it, axis=-1)
    tmp_p_points = np.nonzero(tmp_phase[1, :nt])[0].shape[0]
    tmp_s_points = np.nonzero(tmp_phase[2, :nt])[0].shape[0]
    tmp_event_points = np.nonzero(tmp_event[:nt//feature_scale])[0].shape[0]
    
    if not all([tmp_p_points>=5, tmp_s_points>=5, tmp_event_points>=5]):
        print(f"using smaller rand_upper={rand_upper}")
    
        while any([tmp_p_points < p_points, tmp_s_points < s_points, tmp_event_points < event_points]):
            it = np.random.randint(rand_lower, rand_upper)
            tmp_phase = np.roll(phase_mask, -it*feature_scale, axis=-1)
            tmp_event = np.roll(event_mask, -it, axis=-1)
            tmp_p_points = np.nonzero(tmp_phase[1, :nt])[0].shape[0]
            tmp_s_points = np.nonzero(tmp_phase[2, :nt])[0].shape[0]
            tmp_event_points = np.nonzero(tmp_event[:nt//feature_scale])[0].shape[0]
            tries += 1
            if tries > max_tries:
                print(f"cut data failed, tries={tries}, it={it}")
                it = 0
                break

    waveforms = np.roll(meta["waveforms"], -it*feature_scale, axis=-1)[:, :, :nt]
    amplitude = np.roll(meta["amplitude"], -it*feature_scale, axis=-1)[:, :, :nt]
    phase_pick = np.roll(meta["phase_pick"], -it*feature_scale, axis=-1)[:, :, :nt]
    event_center = np.roll(meta["event_center"], -it, axis=-1)[:, :nt//feature_scale]
    event_location = np.roll(meta["event_location"], -it, axis=-1)[:, :, :nt//feature_scale]
    event_location_mask = np.roll(meta["event_location_mask"], -it, axis=-1)[:, :nt//feature_scale]

    return {
        "waveforms": waveforms,
        "amplitude": amplitude,
        "phase_pick": phase_pick,
        "event_center": event_center,
        "event_location": event_location,
        "event_location_mask": event_location_mask,
    }, it*feature_scale