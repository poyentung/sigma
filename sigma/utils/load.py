import os
import numpy as np
import hyperspy.api as hs

from typing import Union, Tuple, List
from pathlib import Path
from PIL import Image
from os.path import isfile, join
from skimage.transform import resize
from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy._signals.signal2d import Signal2D

hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()


class SEMDataset(object):
    def __init__(self, file_path: Union[str, Path]):
        bcf_dataset = hs.load(file_path)
        self.bse = None
        for dataset in bcf_dataset:
            if (self.bse is None) and (type(dataset) is Signal2D):
                self.original_bse = dataset
                self.bse = dataset  # load BSE data
            elif (self.bse is not None) and (type(dataset) is Signal2D):
                old_w, old_h = self.bse.data.shape
                new_w, new_h = dataset.data.shape
                if (new_w + new_h) < (old_w + old_h):
                    self.original_bse = dataset
                    self.bse = dataset
            elif type(dataset) is EDSSEMSpectrum:
                self.original_edx = dataset
                self.edx = dataset  # load EDX data from bcf file

        self.edx.change_dtype("float32")  # change edx data from unit8 into float32

        self.edx_bin = None
        self.bse_bin = None
        
        # reserve a copy of the raw data for quantification
        self.edx_raw = self.edx.deepcopy()

        self.feature_list = self.edx.metadata.Sample.xray_lines
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}

    def set_feature_list(self, feature_list):
        self.feature_list = feature_list
        for s in [self.edx, self.edx_bin]:
            if s is not None:
                s.metadata.Sample.xray_lines = self.feature_list
        self.feature_dict = {el: i for (i, el) in enumerate(feature_list)}
        print(f"Set feature_list to {self.feature_list}")

    def rebin_signal(self, size=(2, 2)):
        print(f"Rebinning the intensity with the size of {size}")
        x, y = size[0], size[1]
        self.edx_bin = self.edx.rebin(scale=(x, y, 1))
        self.bse_bin = self.bse.rebin(scale=(x, y))
        self.edx_raw = self.edx_bin.deepcopy()
        return (self.edx_bin, self.bse_bin)

    def remove_fist_peak(self, end: float):
        print(
            f"Removing the fisrt peak by setting the intensity to zero until the energy of {end} keV."
        )
        for edx in (self.edx, self.edx_bin):
            if edx is None:
                continue
            else:
                scale = edx.axes_manager[2].scale
                offset = edx.axes_manager[2].offset
                end_ = int((end - offset) / scale)
                for i in range(end_):
                    edx.isig[i] = 0

    def peak_intensity_normalisation(self) -> EDSSEMSpectrum:
        print(
            "Normalising the chemical intensity along axis=2, so that the sum is wqual to 1 along axis=2."
        )
        if self.edx_bin:
            edx_norm = self.edx_bin
        else:
            edx_norm = self.edx
        edx_norm.data = edx_norm.data / edx_norm.data.sum(axis=2, keepdims=True)
        if np.isnan(np.sum(edx_norm.data)):
            edx_norm.data = np.nan_to_num(edx_norm.data)
        return edx_norm

    def peak_denoising_PCA(
        self, n_components_to_reconstruct=10, plot_results=True
    ) -> EDSSEMSpectrum:
        print("Peak denoising using PCA.")
        if self.edx_bin:
            edx_denoised = self.edx_bin
        else:
            edx_denoised = self.edx
        edx_denoised.decomposition(
            normalize_poissonian_noise=True,
            algorithm="SVD",
            random_state=0,
            output_dimension=n_components_to_reconstruct,
        )

        if plot_results == True:
            edx_denoised.plot_decomposition_results()
            edx_denoised.plot_explained_variance_ratio(log=True)
            edx_denoised.plot_decomposition_factors(comp_ids=4)

        return edx_denoised

    def get_feature_maps(self, feature_list=None) -> np:
        if feature_list is not None:
            self.set_feature_list(feature_list)

        num_elements = len(self.feature_list)

        if self.edx_bin is not None:
            lines = self.edx_bin.get_lines_intensity(self.feature_list)
        else:
            lines = self.edx.get_lines_intensity(self.feature_list)

        dims = lines[0].data.shape
        data_cube = np.zeros((dims[0], dims[1], num_elements))

        for i in range(num_elements):
            data_cube[:, :, i] = lines[i]

        return data_cube

    def normalisation(self, norm_list=[]):
        self.normalised_elemental_data = self.get_feature_maps(self.feature_list)
        print("Normalise dataset using:")
        for i, norm_process in enumerate(norm_list):
            print(f"    {i+1}. {norm_process.__name__}")
            self.normalised_elemental_data = norm_process(
                self.normalised_elemental_data
            )


class IMAGEDataset(object):
    def __init__(self, 
                 chemical_maps_dir: Union[str, Path], 
                 intensity_map_path: Union[str, Path]
                 ):

        chemical_maps_paths = [join(chemical_maps_dir, f) for f in os.listdir(chemical_maps_dir)]
        chemical_maps = [Image.open(p) for p in chemical_maps_paths]
        chemical_maps = [np.asarray(img) for img in chemical_maps]

        self.chemical_maps = np.stack(chemical_maps,axis=2).astype(np.float32)
        self.intensity_map = np.asarray(Image.open(intensity_map_path)).astype(np.float32)
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
                self.chemical_maps = maps
            else: 
                self.intensity_map = maps

    def normalisation(self, norm_list:List=[]):
        self.normalised_elemental_data = self.chemical_maps
        print("Normalise dataset using:")
        for i, norm_process in enumerate(norm_list):
            print(f"    {i+1}. {norm_process.__name__}")
            self.normalised_elemental_data = norm_process(
                self.normalised_elemental_data
            )
