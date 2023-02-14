from typing import List
import hyperspy.api as hs
import numpy as np
from sigma.utils.load import SEMDataset
from hyperspy.signals import Signal2D, Signal1D
from hyperspy._signals.eds_tem import EDSTEMSpectrum


class TEMDataset(SEMDataset):
    def __init__(self, file_path: str):
        hs_data = hs.load(file_path)
        if type(hs_data) == Signal2D:
            self.stem = hs.load(file_path)
        else:
            self.edx_bin = None
            self.bse_bin = None
            
            if type(hs_data) == Signal1D:
                self.edx = hs.load(file_path, signal_type="EDS_TEM")
                self.bse = Signal2D(self.edx.data.sum(axis=2))
                self.edx.change_dtype("float32")
                self.edx_raw = self.edx.deepcopy()

                self.edx.metadata.set_item("Sample.xray_lines", [])
                self.edx.axes_manager["Energy"].scale = 0.01 * 8.07 / 8.08
                self.edx.axes_manager["Energy"].offset = -0.01
                self.edx.axes_manager["Energy"].units = "keV"

                self.feature_list = []

            elif type(hs_data) is list:
                bcf_dataset = hs_data
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
                    elif type(dataset) is EDSTEMSpectrum:
                        self.original_edx = dataset
                        self.edx = dataset  # load EDX data from bcf file
                        
                self.edx.change_dtype("float32")  
                self.edx_raw = self.edx.deepcopy()

                self.feature_list = self.edx.metadata.Sample.xray_lines
                self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}

    def set_xray_lines(self, xray_lines: List[str]):
        """
        Set the X-ray lines for the edx analysis. 

        Parameters
        ----------
        xray_lines : List
            A list consisting of a series of elemental peaks. For example, ['Fe_Ka', 'O_Ka'].

        """
        self.feature_list = xray_lines
        self.edx.set_lines(self.feature_list)
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
        self.edx.axes_manager["Energy"].scale = scale
    
    def set_axes_offset(self, offset:float):
        """
        Set the offset for the energy axis. 

        Parameters
        ----------
        offset : float
            the offset of the energy axis. 

        """
        self.edx.axes_manager["Energy"].offset = offset

    def set_axes_unit(self, unit:str):
        """
        Set the unit for the energy axis. 

        Parameters
        ----------
        unit : float
            the unit of the energy axis. 

        """
        self.edx.axes_manager["Energy"].unit = unit
    
    def remove_NaN(self):
        """
        Remove the pixels where no values are stored.
        """
        index_NaN = np.argwhere(np.isnan(self.edx.data[:,0,0]))[0][0]
        self.bse.data = self.bse.data[:index_NaN-1,:]
        self.edx.data = self.edx.data[:index_NaN-1,:,:]

        if self.bse_bin is not None:
            self.bse_bin.data = self.bse_bin.data[:index_NaN-1,:]
        if self.edx_bin is not None:
            self.edx_bin.data = self.edx_bin.data[:index_NaN-1,:,:]