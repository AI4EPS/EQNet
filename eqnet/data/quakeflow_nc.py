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

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
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
_URLS = {
    "NCEDC": [f"{_REPO}/ncedc_event_dataset_{i:03d}.h5" for i in range(1)],
    "NCEDC_full_size": [f"{_REPO}/ncedc_event_dataset_{i:03d}.h5" for i in range(1)],
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
    sampling_rate=100.0
    num_stations = 10
    
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
        datasets.BuilderConfig(name="NCEDC", version=VERSION, description="yield event-based samples one by one, the number of sample stations is fixed(default: 10)"),
        datasets.BuilderConfig(name="NCEDC_full_size", version=VERSION, description="yield event-based samples one by one, the number of sample stations is the same as the number of stations in the event"),
    ]

    DEFAULT_CONFIG_NAME = "NCEDC_full_size"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name=="NCEDC":
            features=datasets.Features(
                {
                    "data": datasets.Array3D(shape=(3, self.nt, self.num_stations), dtype='float32'),
                    "phase_pick": datasets.Array3D(shape=(3, self.nt, self.num_stations), dtype='float32'),
                    "event_center" : datasets.Array2D(shape=(self.feature_nt, self.num_stations), dtype='float32'),
                    "event_location": datasets.Array3D(shape=(4, self.feature_nt, self.num_stations), dtype='float32'),
                    "event_location_mask": datasets.Array2D(shape=(self.feature_nt, self.num_stations), dtype='float32'),
                    "station_location": datasets.Array2D(shape=(self.num_stations, 3), dtype="float32"),
                })
            
        elif self.config.name=="NCEDC_full_size":
            features=datasets.Features(
                {
                    "data": datasets.Array3D(shape=(None, 3, self.nt), dtype='float32'),
                    "phase_pick": datasets.Array3D(shape=(None, 3, self.nt), dtype='float32'),
                    "event_center" : datasets.Array2D(shape=(None, self.feature_nt), dtype='float32'),
                    "event_location": datasets.Array3D(shape=(None, 4, self.feature_nt), dtype='float32'),
                    "event_location_mask": datasets.Array2D(shape=(None, self.feature_nt), dtype='float32'),
                    "station_location": datasets.Array2D(shape=(None, 3), dtype="float32"),
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
        urls = _URLS[self.config.name]
        # files = dl_manager.download(urls)
        files = dl_manager.download_and_extract(urls)
        # files = ["./data/ncedc_event_dataset_000.h5"]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": files,
                    "split": "train",
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "dev.jsonl"),
            #         "split": "dev",
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "test.jsonl"),
            #         "split": "test"
            #     },
            # ),
        ]
        
   

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        num_stations = self.num_stations
        for file in filepath:
            with h5py.File(file, "r") as fp:
                # for event_id in sorted(list(fp.keys())):
                for event_id in fp.keys():
                    event = fp[event_id]
                    station_ids = list(event.keys())
                    event_attrs = event.attrs
                    
                    if self.config.name=="NCEDC":
                        if len(station_ids) < num_stations:
                            continue
                        else:
                            station_ids = np.random.choice(station_ids, num_stations, replace=False)

                    waveforms = np.zeros([3, self.nt, len(station_ids)])
                    phase_pick = np.zeros([3, self.nt, len(station_ids)])
                    event_center = np.zeros([self.nt, len(station_ids)])
                    event_location = np.zeros([4, self.nt, len(station_ids)])
                    event_location_mask = np.zeros([self.nt, len(station_ids)])
                    station_location = np.zeros([len(station_ids), 3])
                    
                    for i, sta_id in enumerate(station_ids):
                        # trace_id = event_id + "/" + sta_id
                        waveforms[:, :, i] = event[sta_id][:,:self.nt]
                        attrs = event[sta_id].attrs
                        p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
                        s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
                        phase_pick[:, :, i] = generate_label([p_picks, s_picks], nt=self.nt)
                        
                        ## TODO: how to deal with multiple phases
                        # center = (attrs["phase_index"][::2] + attrs["phase_index"][1::2])/2.0
                        ## assuming only one event with both P and S picks
                        c0 = ((p_picks) + (s_picks)) / 2.0 # phase center
                        c0_width = ((s_picks - p_picks) * self.sampling_rate / 200.0).max()
                        dx = round(
                            (event_attrs["longitude"] - attrs["longitude"])
                            * np.cos(np.radians(event_attrs["latitude"]))
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
                        
                        event_center[:, i] = generate_label(
                            [
                                # [c0 / self.feature_scale],
                                c0,
                            ],
                            label_width=[
                                c0_width,
                            ],
                            # label_width=[
                            #     10,
                            # ],
                            # nt=self.feature_nt,
                            nt=self.nt,
                        )[1, :]
                        mask = event_center[:, i] >= 0.5
                        event_location[0, :, i] = (
                            np.arange(self.nt) - event_attrs["event_time_index"]
                        ) / self.sampling_rate
                        # event_location[0, :, i] = (np.arange(self.feature_nt) - 3000 / self.feature_scale) / self.sampling_rate
                        event_location[1:, mask, i] = np.array([dx, dy, dz])[:, np.newaxis]
                        event_location_mask[:, i] = mask
                        
                        ## station location
                        station_location[i, 0] = round(
                            attrs["longitude"]
                            * np.cos(np.radians(attrs["latitude"]))
                            * self.degree2km,
                            2,
                        )
                        station_location[i, 1] = round(attrs["latitude"] * self.degree2km, 2)
                        station_location[i, 2] =  round(-attrs["elevation_m"]/1e3, 2)
                        
                    std = np.std(waveforms, axis=1, keepdims=True)
                    std[std == 0] = 1.0
                    waveforms = (waveforms - np.mean(waveforms, axis=1, keepdims=True)) / std
                    waveforms = waveforms.astype(np.float32)
                    
                    if self.config.name=="NCEDC":
                        yield {
                            "data": torch.from_numpy(waveforms).float(),
                            "phase_pick": torch.from_numpy(phase_pick).float(),
                            "event_center": torch.from_numpy(event_center[::self.feature_scale]).float(),
                            "event_location": torch.from_numpy(event_location[:, ::self.feature_scale]).float(),
                            "event_location_mask": torch.from_numpy(event_location_mask[::self.feature_scale]).float(),
                            "station_location": torch.from_numpy(station_location).float(),
                        }
                    elif self.config.name=="NCEDC_full_size":
                        
                        yield event_id, {
                            "data": torch.from_numpy(waveforms).float().permute(2,0,1),
                            "phase_pick": torch.from_numpy(phase_pick).float().permute(2,0,1),
                            "event_center": torch.from_numpy(event_center[:: self.feature_scale]).float().permute(1,0),
                            "event_location": torch.from_numpy(event_location[:, ::self.feature_scale]).float().permute(2,0,1),
                            "event_location_mask": torch.from_numpy(event_location_mask[::self.feature_scale]).float().permute(1,0),
                            "station_location": torch.from_numpy(station_location).float(),
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