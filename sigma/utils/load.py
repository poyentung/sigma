import os
import numpy as np
import hyperspy.api as hs

from typing import Union, Tuple, List
from pathlib import Path
from PIL import Image, ImageOps
from os.path import isfile, join
from skimage.transform import resize
from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy._signals.signal2d import Signal2D
from .base import BaseDataset

class SEMDataset(BaseDataset):
    def __init__(self, file_path: Union[str, Path], nag_file_path: Union[str, Path]=None):
        super().__init__(file_path)

        # for .bcf files:
        if file_path.endswith('.bcf'):
            for dataset in self.base_dataset:
                if (self.nav_img is None) and (type(dataset) is Signal2D):
                    self.original_nav_img = dataset
                    self.nav_img = dataset  # load BSE data
                elif (self.nav_img is not None) and (type(dataset) is Signal2D):
                    old_w, old_h = self.nav_img.data.shape
                    new_w, new_h = dataset.data.shape
                    if (new_w + new_h) < (old_w + old_h):
                        self.original_nav_img = dataset
                        self.nav_img = dataset
                elif type(dataset) is EDSSEMSpectrum:
                    self.original_spectra = dataset
                    self.spectra = dataset  # load spectra data from bcf file

        # for .hspy files:
        elif file_path.endswith('.hspy'):
            if nag_file_path is not None:
                assert nag_file_path.endswith('.hspy')
                nav_img = hs.load(nag_file_path)
            else:
                nav_img = Signal2D(self.base_dataset.sum(axis=2).data).T
            
            self.original_nav_img = nav_img
            self.nav_img = nav_img
            self.original_spectra = self.base_dataset
            self.spectra = self.base_dataset
        
        self.spectra.change_dtype("float32")  # change spectra data from unit8 into float32
        
        # reserve a copy of the raw data for quantification
        self.spectra_raw = self.spectra.deepcopy()

        self.feature_list = self.spectra.metadata.Sample.xray_lines
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}

class IMAGEDataset(object):
    def __init__(self, 
                 chemical_maps_dir: Union[str, Path], 
                 intensity_map_path: Union[str, Path]
                 ):

        chemical_maps_paths = [join(chemical_maps_dir, f) for f in os.listdir(chemical_maps_dir)]
        chemical_maps = [Image.open(p) for p in chemical_maps_paths]
        chemical_maps = [ImageOps.grayscale(p) for p in chemical_maps]
        chemical_maps = [np.asarray(img) for img in chemical_maps]

        self.chemical_maps = np.stack(chemical_maps,axis=2).astype(np.float32)
        self.intensity_map = np.asarray(Image.open(intensity_map_path)).astype(np.int32)

        self.chemical_maps_bin = None
        self.intensity_map_bin = None
        
        self.feature_list = [f.split('.')[0] for f in os.listdir(chemical_maps_dir)]
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        
    def set_feature_list(self, feature_list):
        self.feature_list = feature_list
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        print(f"Set feature_list to {self.feature_list}")
    
    def rebin_signal(self, size:Tuple=(2,2)):
        for (i, maps) in enumerate([self.chemical_maps, self.intensity_map]):
            w, h = maps.shape[:2]
            new_w, new_h = int(w/size[0]), int(h/size[1])
            maps = resize(maps, (new_w, new_h))
            if i ==0: 
                self.chemical_maps_bin = maps
            else: 
                self.intensity_map_bin = maps

    def normalisation(self, norm_list:List=[]):
        self.normalised_elemental_data = self.chemical_maps_bin if self.chemical_maps_bin is not None else self.chemical_maps
        print("Normalise dataset using:")
        for i, norm_process in enumerate(norm_list):
            print(f"    {i+1}. {norm_process.__name__}")
            self.normalised_elemental_data = norm_process(
                self.normalised_elemental_data
            )

class PIXLDataset(IMAGEDataset):
    def __init__(self, file_path: Union[str, Path]):
        self.base_dataset = hs.load(file_path)
        self.chemical_maps = self.base_dataset.data.astype(np.float32)
        self.intensity_map = self.base_dataset.data.sum(axis=2).astype(np.float32)
        self.intensity_map = self.intensity_map / self.intensity_map.max()
        
        self.chemical_maps_bin = None
        self.intensity_map_bin = None
        
        self.feature_list = self.base_dataset.metadata.Signal.phases
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
