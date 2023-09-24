from typing import List
import hyperspy.api as hs
import numpy as np
from sigma.utils.load import SEMDataset
from hyperspy.signals import Signal2D, Signal1D
from hyperspy._signals.signal2d import Signal2D, Signal2D
from hyperspy._signals.eds_tem import EDSTEMSpectrum
from .base import BaseDataset

class TEMDataset(BaseDataset):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        
        if type(self.base_dataset) == Signal2D:
            self.stem = self.base_dataset
        else:
            if type(self.base_dataset) == Signal1D:
                self.spectra = hs.load(file_path, signal_type="EDS_TEM")
                self.nav_img = Signal2D(self.spectra.data.sum(axis=2))
                self.spectra.change_dtype("float32")
                self.spectra_raw = self.spectra.deepcopy()

                self.spectra.metadata.set_item("Sample.xray_lines", [])
                self.spectra.axes_manager["Energy"].scale = 0.01 * 8.07 / 8.08
                self.spectra.axes_manager["Energy"].offset = -0.01
                self.spectra.axes_manager["Energy"].units = "keV"

                self.feature_list = []
            
            # if data format is .emd file
            elif type(self.base_dataset) is list: #file_path[-4:]=='.emd' and 
                emd_dataset = self.base_dataset
                self.nav_img = None
                for dataset in emd_dataset:
                    if (self.nav_img is None) and (dataset.metadata.General.title == "HAADF"):
                        self.original_nav_img = dataset
                        self.nav_img = dataset  # load HAADF data
                    elif type(dataset) is EDSTEMSpectrum:
                        self.original_spectra = dataset
                        self.spectra = dataset  # load spectra data from .emd file

                self.spectra.change_dtype("float32")  
                self.spectra_raw = self.spectra.deepcopy()

                elements = self.spectra.metadata.Sample.elements
                self.spectra.metadata.set_item("Sample.xray_lines", [e+'_Ka' for e in elements])
                self.feature_list = self.spectra.metadata.Sample.xray_lines
                self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}

    def set_xray_lines(self, xray_lines: List[str]):
        """
        Set the X-ray lines for the spectra analysis. 

        Parameters
        ----------
        xray_lines : List
            A list consisting of a series of elemental peaks. For example, ['Fe_Ka', 'O_Ka'].

        """
        self.feature_list = xray_lines
        self.spectra.set_lines(self.feature_list)
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        print(f"Set xray_lines to {self.feature_list}")

    def set_axes_scale(self, scale:float):
        """
        Set the scale for the energy axis. 

        Parameters
        ----------
        scale : float
            The scale of the energy axis. For example, given a data set with 1500 data points corresponding to 0-15 keV, the scale should be set to 0.01.

        """
        self.spectra.axes_manager["Energy"].scale = scale
    
    def set_axes_offset(self, offset:float):
        """
        Set the offset for the energy axis. 

        Parameters
        ----------
        offset : float
            the offset of the energy axis. 

        """
        self.spectra.axes_manager["Energy"].offset = offset

    def set_axes_unit(self, unit:str):
        """
        Set the unit for the energy axis. 

        Parameters
        ----------
        unit : float
            the unit of the energy axis. 

        """
        self.spectra.axes_manager["Energy"].unit = unit
    
    def remove_NaN(self):
        """
        Remove the pixels where no values are stored.
        """
        index_NaN = np.argwhere(np.isnan(self.spectra.data[:,0,0]))[0][0]
        self.nav_img.data = self.nav_img.data[:index_NaN-1,:]
        self.spectra.data = self.spectra.data[:index_NaN-1,:,:]

        if self.nav_img_bin is not None:
            self.nav_img_bin.data = self.nav_img_bin.data[:index_NaN-1,:]
        if self.spectra_bin is not None:
            self.spectra_bin.data = self.spectra_bin.data[:index_NaN-1,:,:]