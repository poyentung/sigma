#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sigma.utils.load import SEMDataset, IMAGEDataset
from sigma.utils.loadtem import TEMDataset
from sigma.utils.visualisation import make_colormap
from sigma.src.utils import k_factors_120kV

from typing import List, Dict, Union
import hyperspy.api as hs
import numpy as np
import pandas as pd
import hdbscan
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, Birch
from skimage import measure
from scipy import fftpack
from skimage.transform import resize

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


class PixelSegmenter(object):
    def __init__(
        self,
        latent: np.ndarray,
        dataset: Union[SEMDataset, TEMDataset, IMAGEDataset],
        method: str = "BayesianGaussianMixture",
        method_args: Dict = {"n_components": 8, "random_state": 4},
    ):

        self.latent = latent
        self.dataset = dataset
        self.dataset_norm = dataset.normalised_elemental_data
        self.method = method
        self.method_args = method_args
        self.height = self.dataset_norm.shape[0]
        self.width = self.dataset_norm.shape[1]

        # Set edx and bse signal to the corresponding ones
        if type(dataset) != IMAGEDataset:
            if self.dataset.edx_bin is not None:
                self.edx = self.dataset.edx_bin
            else:
                self.edx = self.dataset.edx

            if self.dataset.bse_bin is not None:
                self.bse = self.dataset.bse_bin
            else:
                self.bse = self.dataset.bse

            ### Get energy_axis ###
            size = self.edx.axes_manager[2].size
            scale = self.edx.axes_manager[2].scale
            offset = self.edx.axes_manager[2].offset
            self.energy_axis = [((a * scale) + offset) for a in range(0, size)]

        ### Train the model ###
        if self.method == "GaussianMixture":
            self.model = GaussianMixture(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_components"]
        elif self.method == "BayesianGaussianMixture":
            self.model = BayesianGaussianMixture(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_components"]
        elif self.method == "Kmeans":
            self.model = KMeans(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_clusters"]
        elif self.method == "Birch":
            self.model = Birch(**method_args).partial_fit(self.latent)
            self.n_components = self.method_args["n_clusters"]
        elif self.method == "HDBSCAN":
            self.model = hdbscan.HDBSCAN(**method_args)
            self.labels = self.model.fit_predict(latent)
            self.n_components = int(self.labels.max()) + 1 

        if self.method != "HDBSCAN":
            self.labels = self.model.predict(self.latent)

        ### calculate cluster probability maps ###
        means = []
        dataset_ravel = self.dataset_norm.reshape(-1, self.dataset_norm.shape[2])
        for i in range(self.n_components):
            mean = dataset_ravel[np.where(self.labels == i)[0]].mean(axis=0)
            means.append(mean.reshape(1, -1))
        mu = np.concatenate(means, axis=0)

        if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
            prob_map = self.model.predict_proba(self.latent)
            self.prob_map = prob_map

        self.mu = mu

        ### Calcuate peak_dict ###
        self.peak_dict = dict()
        for element in hs.material.elements:
            if element[0] == "Li":
                continue
            for character in element[1].Atomic_properties.Xray_lines:
                peak_name = element[0]
                char_name = character[0]
                key = f"{peak_name}_{char_name}"
                self.peak_dict[key] = character[1].energy_keV

        self.peak_list = self.dataset.feature_list

        # Set color for phase visualisation
        if self.n_components <= 10:
            self._color_palette = "tab10"
            self.color_palette = "tab10"
            self.color_norm = mpl.colors.Normalize(vmin=0, vmax=9)
        else:
            self._color_palette = "nipy_spectral"
            self.color_palette = "nipy_spectral"
            self.color_norm = mpl.colors.Normalize(vmin=0, vmax=self.n_components - 1)

    def set_color_palette(self, cmap):
        self.color_palette = cmap

    def set_feature_list(self, new_list):
        self.peak_list = new_list
        self.dataset.set_feature_list(new_list)

    @staticmethod
    def bic(
        latent,
        n_components=20,
        model="BayesianGaussianMixture",
        model_args={"random_state": 6},
    ):
        def _n_parameters(model):
            """Return the number of free parameters in the model."""
            _, n_features = model.means_.shape
            if model.covariance_type == "full":
                cov_params = model.n_components * n_features * (n_features + 1) / 2.0
            elif model.covariance_type == "diag":
                cov_params = model.n_components * n_features
            elif model.covariance_type == "tied":
                cov_params = n_features * (n_features + 1) / 2.0
            elif model.covariance_type == "spherical":
                cov_params = model.n_components
            mean_params = n_features * model.n_components
            return int(cov_params + mean_params + model.n_components - 1)

        bic_list = []
        for i in range(n_components):
            if model == "BayesianGaussianMixture":
                GMM = BayesianGaussianMixture(n_components=i + 1, **model_args).fit(
                    latent
                )
            elif model == "GaussianMixture":
                GMM = GaussianMixture(n_components=i + 1, **model_args).fit(latent)
            bic = -2 * GMM.score(latent) * latent.shape[0] + _n_parameters(
                GMM
            ) * np.log(latent.shape[0])
            bic_list.append(bic)
        return bic_list

    #################
    # Data Analysis #--------------------------------------------------------------
    #################

    def get_binary_map_edx_profile(
        self,
        cluster_num=1,
        use_label=False,
        threshold=0.8,
        denoise=False,
        keep_fraction=0.13,
        binary_filter_threshold=0.2,
    ):
        if use_label == False:
            if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
                phase = self.model.predict_proba(self.latent)[:, cluster_num]

                if denoise == False:
                    binary_map = np.where(phase > threshold, 1, 0).reshape(
                        self.height, self.width
                    )
                    binary_map_indices = np.where(
                        phase.reshape(self.height, self.width) > threshold
                    )

                else:
                    filtered_img = np.where(phase < threshold, 0, 1).reshape(
                        self.height, self.width
                    )
                    image_fft = fftpack.fft2(filtered_img)
                    image_fft2 = image_fft.copy()

                    # Set r and c to be the number of rows and columns of the array.
                    r, c = image_fft2.shape

                    # Set to zero all rows with indices between r*keep_fraction and
                    # r*(1-keep_fraction):
                    image_fft2[
                        int(r * keep_fraction) : int(r * (1 - keep_fraction))
                    ] = 0

                    # Similarly with the columns:
                    image_fft2[
                        :, int(c * keep_fraction) : int(c * (1 - keep_fraction))
                    ] = 0

                    # Transformed the filtered image back to real space
                    image_new = fftpack.ifft2(image_fft2).real

                    binary_map = np.where(image_new < binary_filter_threshold, 0, 1)
                    binary_map_indices = np.where(image_new > binary_filter_threshold)
            else:
                binary_map = (
                    self.model.labels_
                    * np.where(self.model.labels_ == cluster_num, 1, 0)
                ).reshape(self.height, self.width)
                binary_map_indices = np.where(
                    self.model.labels_.reshape(self.height, self.width) == cluster_num
                )
        else:
            binary_map = (
                self.labels * np.where(self.labels == cluster_num, 1, 0)
            ).reshape(self.height, self.width)
            binary_map_indices = np.where(
                self.labels.reshape(self.height, self.width) == cluster_num
            )

        # Get edx profile in the filtered phase region
        x_id = binary_map_indices[0].reshape(-1, 1)
        y_id = binary_map_indices[1].reshape(-1, 1)
        x_y = np.concatenate([x_id, y_id], axis=1)
        x_y_indices = tuple(map(tuple, x_y))

        if type(self.dataset) != IMAGEDataset:
            total_edx_profiles = list()
            for x_y_index in x_y_indices:
                total_edx_profiles.append(self.edx.data[x_y_index].reshape(1, -1))
            total_edx_profiles = np.concatenate(total_edx_profiles, axis=0)

            size = self.edx.axes_manager[2].size
            scale = self.edx.axes_manager[2].scale
            offset = self.edx.axes_manager[2].offset
            energy_axis = [((a * scale) + offset) for a in range(0, size)]

            element_intensity_sum = total_edx_profiles.sum(axis=0)
            edx_profile = pd.DataFrame(
                data=np.column_stack([energy_axis, element_intensity_sum]),
                columns=["energy", "intensity"],
            )
        else:
            total_edx_profiles = list()
            for x_y_index in x_y_indices:
                total_edx_profiles.append(self.dataset.chemical_maps[x_y_index].reshape(1, -1))
            total_edx_profiles = np.concatenate(total_edx_profiles, axis=0)

            energy_axis = self.dataset.feature_list

            element_intensity_sum = total_edx_profiles.sum(axis=0)
            edx_profile = pd.DataFrame(
                data=np.column_stack([energy_axis, element_intensity_sum]),
                columns=["energy", "intensity"],
            )
        return binary_map, binary_map_indices, edx_profile

    def get_all_edx_profile(self, normalised=True):
        edx_profiles = []
        for i in range(self.n_components):
            _, _, edx_profile = self.get_binary_map_edx_profile(
                cluster_num=i, use_label=True
            )
            edx_profiles.append(edx_profile["intensity"])
        edx_profiles = np.vstack(edx_profiles)
        if normalised == True:
            edx_profiles *= 1 / edx_profiles.max(axis=1, keepdims=True)
        return edx_profiles

    def get_unmixed_edx_profile(
        self,
        clusters_to_be_calculated="All",
        n_components="All",
        normalised=True,
        method="NMF",
        method_args={},
    ):

        if clusters_to_be_calculated != "All":
            num_inputs = len(clusters_to_be_calculated)
        else:
            num_inputs = self.n_components

        if n_components == "All":
            n_components = num_inputs

        assert method == "NMF"
        if method == "NMF":
            model = NMF(n_components=n_components, **method_args)

        edx_profiles = self.get_all_edx_profile(normalised)
        edx_profiles_ = pd.DataFrame(
            edx_profiles.T, columns=range(edx_profiles.shape[0])
        )

        if clusters_to_be_calculated != "All":
            edx_profiles_ = edx_profiles_[clusters_to_be_calculated]

        weights = model.fit_transform(edx_profiles_.to_numpy().T)
        components = model.components_
        self.NMF_recon_error = model.reconstruction_err_

        weights = pd.DataFrame(
            weights.round(3),
            columns=[f"w_{component_num}" for component_num in range(n_components)],
            index=[f"cluster_{cluster_num}" for cluster_num in edx_profiles_],
        )
        components = pd.DataFrame(
            components.T.round(3),
            columns=[f"cpnt_{component_num}" for component_num in range(n_components)],
        )

        return weights, components

    def get_masked_edx(
        self,
        cluster_num,
        threshold=0.8,
        denoise=False,
        keep_fraction=0.13,
        binary_filter_threshold=0.2,
        **binary_filter_args,
    ):

        phase = self.model.predict_proba(self.latent)[:, cluster_num]

        if denoise == False:
            binary_map_indices = np.where(
                phase.reshape(self.height, self.width) <= threshold
            )

        else:
            filtered_img = np.where(phase < threshold, 0, 1).reshape(
                self.height, self.width
            )
            image_fft = fftpack.fft2(filtered_img)
            image_fft2 = image_fft.copy()

            # Set r and c to be the number of rows and columns of the array.
            r, c = image_fft2.shape

            # Set to zero all rows with indices between r*keep_fraction and
            # r*(1-keep_fraction):
            image_fft2[int(r * keep_fraction) : int(r * (1 - keep_fraction))] = 0

            # Similarly with the columns:
            image_fft2[:, int(c * keep_fraction) : int(c * (1 - keep_fraction))] = 0

            # Transformed the filtered image back to real space
            image_new = fftpack.ifft2(image_fft2).real

            binary_map_indices = np.where(image_new > binary_filter_threshold)

        # Get edx profile in the filtered phase region
        x_id = binary_map_indices[0].reshape(-1, 1)
        y_id = binary_map_indices[1].reshape(-1, 1)
        x_y = np.concatenate([x_id, y_id], axis=1)
        x_y_indices = tuple(map(tuple, x_y))

        shape = self.edx.inav[0, 0].data.shape
        masked_edx = self.edx.deepcopy()
        for x_y_index in x_y_indices:
            masked_edx.data[x_y_index] = np.zeros(shape)

        return masked_edx

    def phase_statics(
        self, cluster_num, element_peaks=["Fe_Ka", "O_Ka"], binary_filter_args={}
    ):
        """

        Parameters
        ----------
        binary_map : np.array
            The filtered binary map for analysis.
        element_peaks : dict(), optional
            Determine whether the output includes the elemental intensity from 
            the origianl edx signal. The default is ['Fe_Ka','O_Ka'].
        binary_filter_args : dict()
            Determine the parameters to generate the binary for the analysis.

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe whcih contains all statistical inforamtion of phase distribution.
            These include 'area','equivalent_diameter', 'major_axis_length','minor_axis_length','min_intensity','mean_intensity','max_intensity'

        """

        if binary_filter_args == {}:
            use_label = True
        else:
            use_label = False
        binary_map, _, _ = self.get_binary_map_edx_profile(
            cluster_num, use_label=use_label, **binary_filter_args
        )
        pixel_to_um = self.edx.axes_manager[0].scale
        prop_list = [
            "area",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
            "min_intensity",
            "mean_intensity",
            "max_intensity",
        ]

        label_binary_map = measure.label(binary_map, connectivity=2)
        element_maps = self.dataset.get_feature_maps()

        # Create a dataframe to record all statical information
        stat_info = dict()

        # for each element, create an individual element intensity statics
        for i, element in enumerate(element_peaks):
            element_idx = self.dataset.feature_dict[element]
            clusters = measure.regionprops(
                label_image=label_binary_map,
                intensity_image=element_maps[:, :, element_idx],
            )

            # for the first iteration, record everything apart from elemental intensity i.e. area, length ...
            if i == 0:
                for prop in prop_list:
                    if prop == "area":
                        stat_info[f"{prop} (um^2)"] = [
                            cluster[prop] * pixel_to_um ** 2 for cluster in clusters
                        ]

                    elif prop in [
                        "equivalent_diameter",
                        "major_axis_length",
                        "minor_axis_length",
                    ]:
                        stat_info[f"{prop} (um)"] = [
                            cluster[prop] * pixel_to_um for cluster in clusters
                        ]

                    elif prop in ["min_intensity", "mean_intensity", "max_intensity"]:
                        stat_info[f"{prop}_{element}"] = [
                            cluster[prop] for cluster in clusters
                        ]

            # for the remaining iteration, only add elemental intensity into the dict()
            else:
                for prop in ["min_intensity", "mean_intensity", "max_intensity"]:
                    stat_info[f"{prop}_{element}"] = [
                        cluster[prop] for cluster in clusters
                    ]

        return pd.DataFrame(data=stat_info).round(3)
    
    def cluster_quantification(self,
                               cluster_num:int,
                               elements:List,
                               k_factors:List[float]=None,
                               composition_units:str='atomic',
                               use_label:bool=True)-> pd.DataFrame:
        
        # get indices of the specified cluster
        binary_map, binary_map_indices, _ = self.get_binary_map_edx_profile(cluster_num=cluster_num,use_label=use_label)
        indices = np.column_stack(binary_map_indices)
        indices = tuple(map(tuple, indices))
        
        # set elements for quantification
        edx_raw = self.dataset.edx_raw
        edx_raw.metadata.Sample.xray_lines = elements
        intensities = edx_raw.get_lines_intensity()
        
        if k_factors is None:
            try:
                k_factors = [k_factors_120kV[el] for el in elements]
            except KeyError:
                print('The k factor is not in the database.')
        
        compositions = edx_raw.quantification(intensities, method='CL',factors=k_factors,composition_units='atomic')
        cluster_element_intensities = [c.data[binary_map.astype(bool)] for c in compositions]
        cluster_element_intensities = np.column_stack(cluster_element_intensities)
        
        return pd.DataFrame(cluster_element_intensities, columns = [el.split('_')[0] for el in elements])

        
        
    #################
    # Visualization #--------------------------------------------------------------
    #################

    def plot_latent_space(self, color=True, cmap=None):
        cmap = self.color_palette if cmap is None else cmap

        fig, axs = plt.subplots(1, 1, figsize=(3, 3), dpi=150)
        label = self.labels

        if color:
            axs.scatter(
                self.latent[:, 0],
                self.latent[:, 1],
                c=label,
                s=2.0,
                zorder=2,
                alpha=0.15,
                linewidths=0,
                cmap=cmap,
                norm=self.color_norm,
            )

            if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
                i = 0
                for pos, covar, w in zip(
                    self.model.means_, self.model.covariances_, self.model.weights_
                ):
                    self.draw_ellipse(
                        pos,
                        covar,
                        alpha=0.14,
                        facecolor=plt.cm.get_cmap(cmap)(
                            i * (self.n_components - 1) ** -1
                        ),
                        edgecolor="None",
                        zorder=-10,
                    )
                    self.draw_ellipse(
                        pos,
                        covar,
                        alpha=0.0,
                        edgecolor=plt.cm.get_cmap(cmap)(
                            i * (self.n_components - 1) ** -1
                        ),
                        facecolor="None",
                        zorder=-9,
                        lw=0.25,
                    )
                    i += 1
        else:
            axs.scatter(
                self.latent[:, 0],
                self.latent[:, 1],
                c="k",
                s=1.0,
                zorder=2,
                alpha=0.15,
                linewidths=0,
            )

        for axis in ["top", "bottom", "left", "right"]:
            axs.spines[axis].set_linewidth(1.5)
        plt.show()
        return fig

    def draw_ellipse(self, position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 3):
            ax.add_patch(
                Ellipse(position, nsig * width, nsig * height, angle, **kwargs)
            )

    def plot_cluster_distribution(self, save=None, **kwargs):
        labels = self.model.predict(self.latent)
        means = []
        dataset_ravel = self.dataset_norm.reshape(-1, self.dataset_norm.shape[2])
        for i in range(self.n_components):
            mean = dataset_ravel[np.where(labels == i)[0]].mean(axis=0)
            means.append(mean.reshape(1, -1))
        mu = np.concatenate(means, axis=0)

        if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
            prob_map = self.model.predict_proba(self.latent)

        fig, axs = plt.subplots(
            self.n_components,
            2,
            figsize=(14, self.n_components * 4.2),
            dpi=96,
            **kwargs,
        )
        fig.subplots_adjust(hspace=0.35, wspace=0.1)

        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))

        for i in range(self.n_components):
            if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
                prob_map_i = prob_map[:, i]
            else:
                prob_map_i = np.where(labels == i, 1, 0)
            im = axs[i, 0].imshow(
                prob_map_i.reshape(self.height, self.width), cmap="viridis"
            )
            axs[i, 0].set_title("Probability of each pixel for cluster " + str(i))

            axs[i, 0].axis("off")
            cbar = fig.colorbar(im, ax=axs[i, 0], shrink=0.9, pad=0.025)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=10, size=0)

            if self.n_components <= 10:
                axs[i, 1].bar(
                    self.dataset.feature_list,
                    mu[i],
                    width=0.6,
                    color=plt.cm.get_cmap(self.color_palette)(i * 0.1),
                )
            else:
                axs[i, 1].bar(
                    self.dataset.feature_list,
                    mu[i],
                    width=0.6,
                    color=plt.cm.get_cmap(self.color_palette)(
                        i * (self.n_components - 1) ** -1
                    ),
                )

            axs[i, 1].set_title("Mean value for cluster " + str(i))

        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.show()

        if save is not None:
            fig.savefig(save, bbox_inches="tight", pad_inches=0.01)

    def plot_single_cluster_distribution(self, cluster_num, spectra_range=(0, 8)):
        if type(self.dataset) != IMAGEDataset:
            ncols, figsize = 3, (13, 2.5)
        else:
            ncols, figsize = 2, (5+0.4*len(self.dataset.feature_list),2.5) 
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, dpi=120)
        fig.subplots_adjust(hspace=0.35, wspace=0.1)

        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))

        if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
            prob_map_i = self.prob_map[:, cluster_num]
        else:
            prob_map_i = np.where(self.labels == cluster_num, 1, 0)
        im = axs[0].imshow(prob_map_i.reshape(self.height, self.width), cmap="viridis")
        axs[0].set_title("Pixel-wise probability for cluster " + str(cluster_num))

        axs[0].axis("off")
        cbar = fig.colorbar(im, ax=axs[0], shrink=0.9, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10, size=0)

        if self.n_components <= 10:
            axs[1].bar(
                self.dataset.feature_list,
                self.mu[cluster_num],
                width=0.6,
                color=plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1),
            )
        else:
            axs[1].bar(
                self.dataset.feature_list,
                self.mu[cluster_num],
                width=0.6,
                color=plt.cm.get_cmap(self.color_palette)(
                    cluster_num * (self.n_components - 1) ** -1
                ),
            )

        axs[1].set_xticklabels(self.dataset.feature_list, fontsize=8)
        axs[1].set_title("Mean value for cluster " + str(cluster_num))

        if type(self.dataset)!=IMAGEDataset:

            sum_spectrum = self.dataset.edx_bin if self.dataset.edx_bin else self.dataset.edx
            intensity_sum = sum_spectrum.sum().data / sum_spectrum.sum().data.max()

            try:
                edx_profile = self.get_binary_map_edx_profile(cluster_num)[2]
            except ValueError:
                print(f'warning: no pixel is assigned to cpnt_{cluster_num}')
                return
            
            intensity = edx_profile["intensity"].to_numpy() / edx_profile["intensity"].max()

            axs[2].plot(
                edx_profile["energy"],
                intensity_sum,
                alpha=1,
                linewidth=0.7,
                linestyle="dotted",
                color=sns.color_palette()[0],
                label="Normalised sum spectrum",
            )

            if self.n_components <= 10:
                axs[2].plot(
                    edx_profile["energy"],
                    intensity,
                    linewidth=1,
                    color=plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1),
                )
            else:
                axs[2].plot(
                    edx_profile["energy"],
                    intensity,
                    linewidth=1,
                    color=plt.cm.get_cmap(self.color_palette)(
                        cluster_num * (self.n_components - 1) ** -1
                    ),
                )

            axs[2].set_xticks(np.arange(0, 12, step=1))
            axs[2].set_yticks(np.arange(0, 1.1, step=0.2))

            axs[2].set_xticklabels(np.arange(0, 12, step=1).round(1), fontsize=8)
            axs[2].set_yticklabels(np.arange(0, 1.1, step=0.2).round(1), fontsize=8)

            axs[2].set_xlim(spectra_range[0], spectra_range[1])
            axs[2].set_ylim(None, intensity.max() * 1.35)
            axs[2].set_xlabel("Energy / keV", fontsize=10)
            axs[2].set_ylabel("Intensity / a.u.", fontsize=10)

            legend_properties = {"size": 7}
            axs[2].legend(
                loc="upper right", handletextpad=0.5, frameon=False, prop=legend_properties
            )

            if np.array(edx_profile["energy"]).min() <= 0:
                zero_energy_idx = np.where(np.array(edx_profile["energy"]).round(2) == 0)[
                    0
                ][0]
            else:
                zero_energy_idx = 0
            for el in self.dataset.feature_list:
                peak_sum = intensity_sum[zero_energy_idx:][
                    int(self.peak_dict[el] * 100) + 1
                ]
                peak_single = intensity[zero_energy_idx:][int(self.peak_dict[el] * 100) + 1]

                peak = max(peak_sum, peak_single)
                axs[2].vlines(
                    self.peak_dict[el],
                    0,
                    int(0.9 * peak),
                    linewidth=0.7,
                    color="grey",
                    linestyles="dashed",
                )
                axs[2].text(
                    self.peak_dict[el] - 0.1,
                    peak + (int(intensity.max()) / 20),
                    el,
                    rotation="vertical",
                    fontsize=7.5,
                )

        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        fig.set_tight_layout(True)
        plt.show()
        return fig

    def plot_phase_map(self, cmap=None):
        cmap = self.color_palette if cmap is None else cmap
        if type(self.dataset)!=IMAGEDataset:
            img = self.bse.data 
        else:
            img = resize(self.dataset.intensity_map, self.dataset.chemical_maps.shape[:2])

        phase = self.labels.reshape(self.height, self.width)

        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 4), dpi=100)

        axs[0].imshow(img, cmap="gray", interpolation="none")
        if type(self.dataset)==SEMDataset:
            axs[0].set_title("BSE")
        else:
            axs[0].set_title("Intensity")

        axs[0].axis("off")

        axs[1].imshow(img, cmap="gray", interpolation="none", alpha=1.0)

        if self.n_components <= 10:
            axs[1].imshow(
                phase,
                cmap=self.color_palette,
                interpolation="none",
                norm=self.color_norm,
                alpha=0.75,
            )
        else:
            axs[1].imshow(
                phase,
                cmap=self.color_palette,
                interpolation="none",
                alpha=0.6,
                norm=self.color_norm,
            )
        axs[1].axis("off")
        axs[1].set_title("Cluster map")

        fig.subplots_adjust(wspace=0.05, hspace=0.0)
        plt.show()
        return fig

    def plot_binary_map_edx_profile(
        self, cluster_num, normalisation=True, spectra_range=(0, 8), **kwargs
    ):

        binary_map, binary_map_indices, edx_profile = self.get_binary_map_edx_profile(
            cluster_num, use_label=False
        )

        if type(self.dataset) != IMAGEDataset:
            ncols, figsize, gridspec_kw = 3, (13, 3), {"width_ratios": [1, 1, 2]}
        else:
            ncols, figsize, gridspec_kw = 2, (6,3), None

        fig, axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=figsize,
            dpi=96,
            gridspec_kw=gridspec_kw,
            **kwargs,
        )

        phase_color = plt.cm.get_cmap(self.color_palette)(
            cluster_num * (self.n_components - 1) ** -1
        )
        c = mcolors.ColorConverter().to_rgb

        if (self.n_components > 10) and (cluster_num == 0):
            cmap = make_colormap([c("k"), c("w"), 1, c("w")])
        else:
            cmap = make_colormap([c("k"), phase_color[:3], 1, phase_color[:3]])

        axs[0].imshow(binary_map, cmap=cmap)
        axs[0].set_title(f"Binary map (cluster {cluster_num})", fontsize=10)
        axs[0].axis("off")
        axs[0].set_aspect("equal", "box")

        if type(self.dataset)!=IMAGEDataset:
            bse = self.dataset.bse_bin.data if self.dataset.bse_bin else self.dataset.bse.data
        else:
            if self.dataset.intensity_map.shape[:2]!= self.dataset.chemical_maps.shape[:2]: # if size of intensity map is different from chemical maps
                bse = resize(self.dataset.intensity_map, self.dataset.chemical_maps.shape[:2]) 
            else:
                bse = self.dataset.intensity_map
        axs[1].imshow(bse, cmap="gray", interpolation="none", alpha=0.9)
        axs[1].scatter(
            binary_map_indices[1], binary_map_indices[0], c="r", alpha=0.2, s=1.5
        )
        axs[1].grid(False)
        axs[1].axis("off")
        if type(self.dataset)==SEMDataset:
            axs[1].set_title("BSE + Binary Map", fontsize=10)
        else:
            axs[1].set_title("Intensity + Binary Map", fontsize=10)

        if type(self.dataset)!=IMAGEDataset:
            if normalisation:
                intensity = (
                    edx_profile["intensity"].to_numpy() / edx_profile["intensity"].max()
                )
            else:
                intensity = edx_profile["intensity"].to_numpy()

            if self.n_components <= 10:
                axs[2].plot(
                    edx_profile["energy"],
                    intensity,
                    linewidth=1,
                    color=plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1),
                )
            else:
                axs[2].plot(
                    edx_profile["energy"],
                    intensity,
                    linewidth=1,
                    color=plt.cm.get_cmap(self.color_palette)(
                        cluster_num * (self.n_components - 1) ** -1
                    ),
                )

            if np.array(edx_profile["energy"]).min() <= 0.0:
                zero_energy_idx = np.where(np.array(edx_profile["energy"]).round(2) == 0)[
                    0
                ][0]
            else:
                zero_energy_idx = 0
            for el in self.peak_list:
                peak = intensity[zero_energy_idx:][int(self.peak_dict[el] * 100) + 1]
                axs[2].vlines(
                    self.peak_dict[el],
                    0,
                    int(0.9 * peak),
                    linewidth=0.7,
                    color="grey",
                    linestyles="dashed",
                )
                axs[2].text(
                    self.peak_dict[el] - 0.1,
                    peak + (int(intensity.max()) / 20),
                    el,
                    rotation="vertical",
                    fontsize=8,
                )

            axs[2].set_xticks(np.arange(spectra_range[0], spectra_range[1], step=1))
            axs[2].set_xticklabels(
                np.arange(spectra_range[0], spectra_range[1], step=1), fontsize=8
            )

            if normalisation:
                axs[2].set_yticks(np.arange(0, 1.1, step=0.2))
                axs[2].set_yticklabels(np.arange(0, 1.1, step=0.2).round(1), fontsize=8)
            else:
                try:
                    axs[2].set_yticks(
                        np.arange(
                            0,
                            int(intensity.max().round()) + 1,
                            step=int((intensity.max().round() / 5)),
                        )
                    )
                    axs[2].set_yticklabels(
                        np.arange(
                            0,
                            int(intensity.max().round()) + 1,
                            step=int((intensity.max().round() / 5)),
                        ),
                        fontsize=8,
                    )
                except ZeroDivisionError:
                    pass
            
            axs[2].set_xlim(spectra_range[0], spectra_range[1])
            axs[2].set_ylim(None, intensity.max() * 1.2)
            axs[2].set_xlabel("Energy / keV", fontsize=10)
            axs[2].set_ylabel("X-rays / Counts", fontsize=10)

        # fig.subplots_adjust(left=0.1)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_binary_map(
        self,
        cluster_num,
        binary_filter_args={
            "threshold": 0.8,
            "denoise": False,
            "keep_fraction": 0.13,
            "binary_filter_threshold": 0.2,
        },
        save=None,
        **kwargs,
    ):

        binary_map, binary_map_indices, edx_profile = self.get_binary_map_edx_profile(
            cluster_num, **binary_filter_args
        )

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 2.5), dpi=96, **kwargs)

        axs[0].imshow(binary_map, interpolation="none", alpha=1)
        axs[0].set_title(f"Filtered Binary map (cluster {cluster_num})")
        axs[0].axis("off")
        axs[0].set_aspect("equal", "box")

        axs[1].imshow(self.dataset.bse_bin.data, cmap="gray", interpolation="none", alpha=1)
        axs[1].scatter(
            binary_map_indices[1], binary_map_indices[0], c="r", alpha=0.05, s=1.2
        )
        axs[1].grid(False)
        axs[1].axis("off")
        axs[1].set_title(f"BSE + Phase Map (cluster {cluster_num})")

        # fig.subplots_adjust(left=0.1)
        plt.tight_layout()

        if save is not None:
            fig.savefig(save, bbox_inches="tight", pad_inches=0.02)

        plt.show()

    def plot_unmixed_profile(self, components, peak_list=[]):
        if len(peak_list) == 0:
            peak_list = self.peak_list
        cpnt_num = len(components.columns.to_list())
        if cpnt_num > 4:
            n_rows = (cpnt_num + 3) // 4
            n_cols = 4
        else:
            n_rows = 1
            n_cols = cpnt_num

        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 2.6), dpi=150
        )
        for row in range(n_rows):
            for col in range(n_cols):
                cur_cpnt = (row * n_cols) + col
                if cur_cpnt > cpnt_num - 1:  # delete the extra subfigures
                    fig.delaxes(axs[row, col])
                else:
                    cpnt = f"cpnt_{cur_cpnt}"
                    if cpnt_num > 4:
                        axs_sub = axs[row, col]
                    else:
                        axs_sub = axs[col]
                    axs_sub.plot(self.energy_axis, components[cpnt], linewidth=1)
                    axs_sub.set_xlim(0, 8)
                    axs_sub.set_ylim(None, components[cpnt].max() * 1.3)
                    axs_sub.set_ylabel("Intensity")
                    axs_sub.set_xlabel("Energy (keV)")
                    axs_sub.set_title(f"cpnt_{cur_cpnt}")

                    if np.array(self.energy_axis).min() <= 0.0:
                        zero_energy_idx = np.where(
                            np.array(self.energy_axis).round(2) == 0
                        )[0][0]
                    else:
                        zero_energy_idx = 0
                    intensity = components[cpnt].to_numpy()
                    for el in peak_list:
                        peak = intensity[zero_energy_idx:][
                            int(self.peak_dict[el] * 100) + 1
                        ]
                        axs_sub.vlines(
                            self.peak_dict[el],
                            0,
                            0.9 * peak,
                            linewidth=1,
                            color="grey",
                            linestyles="dashed",
                        )
                        axs_sub.text(
                            self.peak_dict[el] - 0.18,
                            peak + (intensity.max() / 15),
                            el,
                            rotation="vertical",
                            fontsize=8,
                        )

        fig.subplots_adjust(hspace=0.3, wspace=0.0)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_edx_profile(self, cluster_num, peak_list, binary_filter_args):
        edx_profile = self.get_binary_map_edx_profile(
            cluster_num, **binary_filter_args
        )[2]
        intensity = edx_profile["intensity"].to_numpy()

        fig, axs = plt.subplots(1, 1, figsize=(4, 2), dpi=150)
        axs.set_xticks(np.arange(0, 12, step=1))
        axs.set_yticks(
            np.arange(0, int(intensity.max()) + 1, step=int((intensity.max() / 5)))
        )

        axs.set_xticklabels(np.arange(0, 12, step=1), fontsize=8)
        axs.set_yticklabels(
            np.arange(0, int(intensity.max()) + 1, step=int((intensity.max() / 5))),
            fontsize=8,
        )

        axs.set_xlim(0, 8)
        axs.set_ylim(None, intensity.max() * 1.25)
        axs.set_xlabel("Energy axis / keV", fontsize=10)
        axs.set_ylabel("X-rays / Counts", fontsize=10)

        if self.n_components <= 10:
            axs.plot(
                edx_profile["energy"],
                edx_profile["intensity"],
                linewidth=1,
                color=plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1),
            )
        else:
            axs.plot(
                edx_profile["energy"],
                edx_profile["intensity"],
                linewidth=1,
                color=plt.cm.get_cmap(self.color_palette)(
                    cluster_num * (self.n_components - 1) ** -1
                ),
            )

        zero_energy_idx = np.where(np.array(edx_profile["energy"]).round(2) == 0)[0][0]
        for el in peak_list:
            peak = intensity[zero_energy_idx:][int(self.peak_dict[el] * 100) + 1]
            axs.vlines(
                self.peak_dict[el],
                0,
                int(0.9 * peak),
                linewidth=0.7,
                color="grey",
                linestyles="dashed",
            )
            axs.text(
                self.peak_dict[el] - 0.075,
                peak + (int(intensity.max()) / 20),
                el,
                rotation="vertical",
                fontsize=7.5,
            )
        plt.show()
    
    def plot_ternary_composition(self, **kwargs): # see args for cluster_quantification
        cluster_element_intensities = self.cluster_quantification(**kwargs)
        
        fig = go.Figure(px.scatter_ternary(cluster_element_intensities, 
                                   *cluster_element_intensities.columns,
                                   template='none',
                                   opacity=0.5)
        )

        fig.update_layout(title="Ternary diagram in at.%",
                          title_x=0.5,
                          width=500,
                          height=500)
        
        fig.update_traces(marker=dict(size=3.0,
                                      line=dict(width=0)),
                          selector=dict(mode='markers'),
                         )
        fig.show()
