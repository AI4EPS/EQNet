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
from scipy import signal


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
    "event": [f"{_PATH}/{x}" for x in _FILES],
    "station_train": [f"{_REPO}/{x}" for x in _FILES[:-1]],
    "event_train": [f"{_REPO}/{x}" for x in _FILES[:-1]],
    "station_test": [f"{_REPO}/{x}" for x in _FILES[-1:]],
    "event_test": [f"{_REPO}/{x}" for x in _FILES[-1:]],
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


@np.vectorize
def mapping_phase_type(phase):
    if phase in ["P", "Pn", "Pg", "Pb"]:
        return "P"
    elif phase in ["S", "Sn", "Sg", "Sb"]:
        return "S"
    else:
        return phase
    
def calc_freq_index(waveforms, picks, upper_band=(10, 15), lower_band=(1, 5), before=1, after=6, dt_s=0.01):
    r'''FI = log_10 (\bar{A}_upper / \bar{A}_lower)
    \bar{A}_upper and \bar{A}_lower are the mean spectral amplitudes in a predefined high-frequency
    band and a low-frequency band, respectively. Time windows starting 1s prior to and ending 6s after P arrivals 
    are extracted to calculate FIs. 
    input is a 3-component waveform, output is a numpy array of shape (3,), each element is a frequency index
    if one component is all zeros, the corresponding frequency index is np.nan
    if picks is empty or last pick is larger than waveforms length, all frequency index are np.nan
    
    Args:
        waveforms: numpy.ndarray, (3, nt)
        picks: numpy.ndarray, (2,)
        upper_band: tuple, (low, high)
        lower_band: tuple, (low, high)
    '''
    
    if len(picks) == 0 or picks.max() >= waveforms.shape[1]:
        return np.array([np.nan, np.nan, np.nan])
    
    fi = np.zeros(3)
    for i in range(3):
        #if np.all(waveforms[i]==0):
        if np.abs(waveforms[i]).max() == 0:
            fi[i] = np.nan
            continue
        window = waveforms[i, int(picks[0]-before/dt_s):int(picks[0]+after/dt_s)]
        freqs, psd = signal.welch(window, fs=1/dt_s, nperseg=512)
        upper = np.where(np.logical_and(freqs >= upper_band[0], freqs <= upper_band[1]))[0]
        lower = np.where(np.logical_and(freqs >= lower_band[0], freqs <= lower_band[1]))[0]
        fi[i] = np.log10(psd[upper].mean() / psd[lower].mean())
    return fi

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class Ncedc_Catalog(datasets.GeneratorBasedBuilder):
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
                    "event_id": datasets.Value("string"),
                    "station_id": datasets.Value("string"),
                    "phase_pick": datasets.Sequence(datasets.Value("float32")),
                    "event_location": datasets.Sequence(datasets.Value("float32")),
                    "station_location": datasets.Sequence(datasets.Value("float32")),
                })
            
        elif (
            (self.config.name == "event") 
            or (self.config.name == "event_train") 
            or (self.config.name == "event_test")
            or (self.config.name == "event_custom")
        ):
            features=datasets.Features(
                {
                    "event_id": datasets.Value("string"),
                    "num_stations": datasets.Value("int32"),
                    "origin_time": datasets.Value("string"),
                    "longitude": datasets.Value("float32"),
                    "latitude": datasets.Value("float32"),
                    "depth_km": datasets.Value("float32"),
                    "magnitude": datasets.Value("float32"),
                    "magnitude_type": datasets.Value("string"),
                    "event_frequency_index": datasets.Value("float32"),
                    "station_ids": datasets.Sequence(datasets.Value("string")),
                    "station_location": datasets.Array2D(shape=(None, 3), dtype='float32'),
                    "station_dist_azi": datasets.Array2D(shape=(None, 2), dtype='float32'),
                    "station_frequency_index": datasets.Array2D(shape=(None, 3), dtype='float32'),
                    "station_snr": datasets.Array2D(shape=(None, 3), dtype='float32'),
                    "picks": datasets.Array2D(shape=(None, 2), dtype='float32'),
                    "dt_s": datasets.Sequence(datasets.Value("float32")),
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
                    print(f"total {len(event_ids)} events in {file}")
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
                                # waveforms[:, : self.nt] = event[sta_id][: self.nt, :].T
                                attrs = event[sta_id].attrs
                                p_picks = attrs["phase_index"][attrs["phase_type"] == "P"]
                                s_picks = attrs["phase_index"][attrs["phase_type"] == "S"]
                                phase_pick = np.array([p_picks[0], s_picks[0]])
                                # phase_pick[:, :self.nt] = generate_label([p_picks, s_picks], nt=self.nt)
                                station_location = [attrs["longitude"], attrs["latitude"], -attrs["elevation_m"] / 1e3]

                                yield f"{event_id}/{sta_id}", {
                                    "event_id": event_id,
                                    "station_id": sta_id,
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
                                if len(np.unique(mapping_phase_type(event[sta_id].attrs["phase_type"]))) < 2:
                                    auto_count += 1
                                    continue # only one phase type indicates automatic picks
                                valid_manual_events = 0
                                attrs = event[sta_id].attrs
                                sta_phase_type = mapping_phase_type(attrs["phase_type"])
                                for e_id in [event_id]:#sorted(np.unique(event[sta_id].attrs["event_id"])):
                                    if len(np.unique(sta_phase_type[attrs["event_id"]==e_id])) < 2:
                                        continue
                                    
                                    p_picks = attrs["phase_index"][np.logical_and(sta_phase_type == "P", attrs["event_id"] == e_id)]
                                    s_picks = attrs["phase_index"][np.logical_and(sta_phase_type == "S", attrs["event_id"] == e_id)]
                                    if p_picks.min()>self.nt or s_picks.min()>self.nt:
                                        continue # phases too late
                                    # try:
                                    #     if len(p_picks)!=len(s_picks):
                                    #         is_sick = True
                                    #         break
                                    # except:
                                    #     pass
                                    #assert len(p_picks)==len(s_picks), f"{event_id}/{sta_id} manual P&S picks num don't match"
                                    if len(p_picks)!=len(s_picks):
                                        print(f"{event_id}/{sta_id} manual P&S picks num don't match")
                                        is_sick = True
                                        break
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

                            num_stations = len(station_ids)
                            origin_time = event_attrs["event_time"]
                            location = np.array([
                                event_attrs["longitude"],
                                event_attrs["latitude"],
                                event_attrs["depth_km"]
                            ])
                            magnitude = event_attrs["magnitude"]
                            magnitude_type = event_attrs["magnitude_type"]
                            station_location = np.zeros((num_stations, 3))
                            station_dist_azi = np.zeros((num_stations, 2))
                            station_picks = np.zeros((num_stations, 2))
                            station_frequency_index = np.zeros((num_stations, 3))
                            station_snr = np.zeros((num_stations, 3))
                            dt_s = np.zeros(num_stations)

                            for i, sta_id in enumerate(station_ids):
                                attrs = event[sta_id].attrs
                                sta_phase_type = mapping_phase_type(attrs["phase_type"])
                                p_picks = attrs["phase_index"][np.logical_and(sta_phase_type == "P", attrs["event_id"] == event_attrs["event_id"])]
                                s_picks = attrs["phase_index"][np.logical_and(sta_phase_type == "S", attrs["event_id"] == event_attrs["event_id"])]
                                if len(p_picks)==0:
                                    p_picks = np.array([1])
                                if len(s_picks)==0:
                                    s_picks = np.array([1])
                                station_picks[i, 0] = p_picks[0]
                                station_picks[i, 1] = s_picks[0]

                                station_location[i, 0] = round(attrs["longitude"], 4)
                                station_location[i, 1] = round(attrs["latitude"], 4) 
                                station_location[i, 2] = round(-attrs["elevation_m"]/1e3, 2)
                                station_dist_azi[i, 0] = attrs["distance_km"]
                                station_dist_azi[i, 1] = attrs["azimuth"]
                                station_snr[i] = attrs['snr']
                                station_frequency_index[i] = calc_freq_index(event[sta_id][:,:], p_picks)
                                if np.all(np.isnan(station_frequency_index[i])):
                                    if len(p_picks) == 0:
                                        print(f"{event_id}/{sta_id} no p pick")
                                    elif np.all(event[sta_id][:,:]==0):
                                        print(f"{event_id}/{sta_id} waveform all zeros")
                                    elif p_picks[0] >= event[sta_id][:,:].shape[-1]:
                                        print(f"{event_id}/{sta_id} p pick too late")
                                dt_s[i] = attrs["dt_s"]
                                
                            # mean of station_frequency_index exclude NaN
                            event_freq_index = np.nanmean(station_frequency_index) if not np.all(np.isnan(station_frequency_index)) else np.nan
                            
                            station_ids = np.array(station_ids)
                            
                            if (self.config.name == "event_large" or self.config.name == "event") and len(station_ids) > 5:
                                D = np.sqrt(((station_location[:, np.newaxis, :2] -  station_location[np.newaxis, :, :2])**2).sum(axis=-1))
                                Tcsr = minimum_spanning_tree(D)
                                index = breadth_first_order(Tcsr, i_start=0, directed=False, return_predecessors=False)
                                station_ids = station_ids[index]
                                station_location = station_location[index]
                                station_dist_azi = station_dist_azi[index]
                                station_picks = station_picks[index]
                                station_frequency_index = station_frequency_index[index]
                                station_snr = station_snr[index]
                                dt_s = dt_s[index]

                            yield event_id, {
                                "event_id": event_id,
                                "num_stations": num_stations,
                                "origin_time": origin_time,
                                "longitude": location[0],
                                "latitude": location[1],
                                "depth_km": location[2],
                                "magnitude": magnitude,
                                "magnitude_type": magnitude_type,
                                "event_frequency_index": event_freq_index,
                                "station_ids": station_ids,
                                "station_location": torch.from_numpy(station_location).float(),
                                "station_dist_azi": torch.from_numpy(station_dist_azi).float(),
                                "station_frequency_index": torch.from_numpy(station_frequency_index).float(),
                                "station_snr": torch.from_numpy(station_snr).float(),
                                "picks": torch.from_numpy(station_picks).float(),
                                "dt_s": torch.from_numpy(dt_s).float(),
                            }