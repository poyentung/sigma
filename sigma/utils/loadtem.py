import hyperspy.api as hs
from sigma.utils.load import SEMDataset
from hyperspy.signals import Signal2D, Signal1D


class TEMDataset(SEMDataset):
    def __init__(self, file_path: str):
        if type(hs.load(file_path)) == Signal2D:
            self.stem = hs.load(file_path)

        elif type(hs.load(file_path)) == Signal1D:
            self.edx = hs.load(file_path, signal_type="EDS_TEM")
            self.bse = Signal2D(self.edx.data.sum(axis=2))
            self.edx.change_dtype("float32")

            self.edx.metadata.set_item("Sample.xray_lines", [])
            self.edx.axes_manager["Energy"].scale = 0.01 * 8.6326 / 8.62
            self.edx.axes_manager["Energy"].offset = 0.01
            self.edx.axes_manager["Energy"].units = "keV"

            self.edx_bin = None
            self.bse_bin = None
            self.feature_list = []

    def set_xray_lines(self, xray_lines: list):
        self.feature_list = xray_lines
        self.edx.set_lines(self.feature_list)
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        print(f"Set xray_lines to {self.feature_list}")
