#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from load_dataset.exhaust import SEMDataset, peak_dict, make_colormap
import hyperspy.api as hs

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, Birch
from scipy import ndimage as ndi
from sklearn.cluster import MeanShift
from skimage import io, img_as_float, measure
from scipy import fftpack

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from  matplotlib import gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px



class PhaseClassifier(object):
    def __init__(self, 
                 latent:np, dataset_norm:np, sem:SEMDataset,
                 method='GaussianMixture', method_args={'n_components':8,
                                                        'random_state':4}):
        
        self.latent = latent
        self.dataset = dataset_norm
        self.sem = sem
        self.method = method
        self.method_args = method_args
        self.height = self.dataset.shape[0]
        self.width = self.dataset.shape[1]
        
        # Set edx and bse signal to the corresponding ones
        if self.sem.edx_bin is not None: self.edx=self.sem.edx_bin
        else: self.edx=self.sem.edx
        
        if self.sem.bse_bin is not None: self.bse=self.sem.bse_bin
        else: self.bse=self.sem.bse
        
        ### Get energy_axis ###
        size = self.edx.axes_manager[2].size
        scale = self.edx.axes_manager[2].scale
        offset = self.edx.axes_manager[2].offset
        self.energy_axis = [((a*scale) + offset) for a in range(0,size)]
        
        ### Train the model ###
        if self.method == 'GaussianMixture':
            self.model = GaussianMixture(**method_args).fit(self.latent)
            self.n_components = self.method_args['n_components']
        elif self.method == 'BayesianGaussianMixture':
            self.model = BayesianGaussianMixture(**method_args).fit(self.latent)
            self.n_components = self.method_args['n_components']
        elif self.method == 'Kmeans':
            self.model = KMeans(**method_args).fit(self.latent)
            self.n_components = self.method_args['n_clusters']
        elif self.method == 'Birch':
            self.model = Birch(**method_args).partial_fit(self.latent)
            self.n_components = self.method_args['n_clusters']
        
        ### calculate cluster probability maps ###
        labels = self.model.predict(self.latent)
        means = []
        dataset_ravel = self.dataset.reshape(-1,self.dataset.shape[2])
        for i in range(self.n_components):
            mean = dataset_ravel[np.where(labels==i)[0]].mean(axis=0)
            means.append(mean.reshape(1,-1))
        mu = np.concatenate(means,axis=0)
        
        if self.method in ['GaussianMixture', 'BayesianGaussianMixture']:
            prob_map = self.model.predict_proba(self.latent)
            
        self.mu = mu 
        self.labels = labels
        self.prob_map = prob_map
        
        ### Calcuate peak_dict ###
        self.peak_dict = dict()
        for element in hs.material.elements:
            if element[0]=='Li': continue
            for character in element[1].Atomic_properties.Xray_lines:
                peak_name = element[0]
                char_name = character[0]
                key = f'{peak_name}_{char_name}'
                self.peak_dict[key] = character[1].energy_keV
        
        self.peak_list = self.sem.feature_list
        
        # Set color for phase visualisation
        if self.n_components <= 10:
            self.color_palette = 'tab10'
            self.color_norm = mpl.colors.Normalize(vmin=0, vmax=9)
        else:
            self.color_palette = 'nipy_spectral'
            self.color_norm = mpl.colors.Normalize(vmin=0, vmax=self.n_components-1)
            
        
    def set_feature_list(self, new_list):
        self.peak_list = new_list
        self.sem.set_feature_list(new_list)

#################
# Data Analysis #--------------------------------------------------------------
#################
        
    def get_binary_map_edx_profile(self, cluster_num=1, threshold=0.8, 
                                   denoise=False,keep_fraction=0.13, 
                                   binary_filter_threshold=0.2):
        
        phase = self.model.predict_proba(self.latent)[:,cluster_num]
        
        if denoise == False:
            binary_map = np.where(phase>threshold,1,0).reshape(self.height,self.width)
            binary_map_indices = np.where(phase.reshape(self.height,self.width)>threshold)
        
        else:
            filtered_img = np.where(phase<threshold,0,1).reshape(self.height,self.width)
            image_fft = fftpack.fft2(filtered_img)
            image_fft2 = image_fft.copy()
            
            # Set r and c to be the number of rows and columns of the array.
            r, c = image_fft2.shape
        
            # Set to zero all rows with indices between r*keep_fraction and
            # r*(1-keep_fraction):
            image_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
        
            # Similarly with the columns:
            image_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
        
            # Transformed the filtered image back to real space
            image_new = fftpack.ifft2(image_fft2).real
        
            binary_map = np.where(image_new<binary_filter_threshold,0,1)
            binary_map_indices = np.where(image_new>binary_filter_threshold)
            
        # Get edx profile in the filtered phase region
        x_id = binary_map_indices[0].reshape(-1,1)
        y_id = binary_map_indices[1].reshape(-1,1)
        x_y = np.concatenate([x_id, y_id],axis=1)
        x_y_indices = tuple(map(tuple, x_y))
        
        total_edx_profiles = list()
        for x_y_index in x_y_indices:
            total_edx_profiles.append(self.edx.data[x_y_index].reshape(1,-1))
        total_edx_profiles = np.concatenate(total_edx_profiles,axis=0)
        
        size = self.edx.axes_manager[2].size
        scale = self.edx.axes_manager[2].scale
        offset = self.edx.axes_manager[2].offset
        energy_axis = [((a*scale) + offset) for a in range(0,size)]

        element_intensity_sum = total_edx_profiles.sum(axis=0)
        edx_profile = pd.DataFrame(data=np.column_stack([energy_axis, element_intensity_sum]),
                                   columns=['energy', 'intensity'])
        
        return binary_map, binary_map_indices, edx_profile
    
    def get_all_edx_profile(self, normalised=True, binary_filter_args={}):
        edx_profiles = []
        for i in range(self.n_components):
            _,_,edx_profile = self.get_binary_map_edx_profile(cluster_num=i, **binary_filter_args)
            edx_profiles.append(edx_profile['intensity'])
        edx_profiles = np.vstack(edx_profiles)
        if normalised==True:
            edx_profiles *= 1/edx_profiles.max(axis=1,keepdims=True)
        return edx_profiles

    def get_unmixed_edx_profile(self, 
                                clusters_to_be_calculated='All',
                                n_components='All',
                                normalised=True, method='NMF', 
                                method_args={},
                                binary_filter_args={}):
        
        if clusters_to_be_calculated != 'All':
            num_inputs = len(clusters_to_be_calculated)
        else:
            num_inputs = self.n_components
        
        if n_components=='All':
            n_components = num_inputs

        assert(method=='NMF')
        if method == 'NMF':
            model = NMF(n_components=n_components, **method_args)
        
        edx_profiles = self.get_all_edx_profile(normalised, binary_filter_args)
        edx_profiles_ = pd.DataFrame(edx_profiles.T, columns=range(edx_profiles.shape[0]))
        
        if clusters_to_be_calculated != 'All':
            edx_profiles_ = edx_profiles_[clusters_to_be_calculated]
            
        weights = model.fit_transform(edx_profiles_.to_numpy().T)
        components = model.components_
        self.NMF_recon_error = model.reconstruction_err_

        weights = pd.DataFrame(weights.round(3), columns=[f'w_{component_num}' for component_num in range(n_components)],
                               index=[f'cluster_{cluster_num}' for cluster_num in edx_profiles_])
        components = pd.DataFrame(components.T.round(3), columns=[f'cpnt_{component_num}' for component_num in range(n_components)])

        return weights, components
    
    def get_masked_edx(self, cluster_num, threshold=0.8, 
                       denoise=False,keep_fraction=0.13, 
                       
                       binary_filter_threshold=0.2, 
                       **binary_filter_args):
        
        phase = self.model.predict_proba(self.latent)[:,cluster_num]
        
        if denoise == False:
            binary_map_indices = np.where(phase.reshape(self.height,self.width)<=threshold)
        
        else:
            filtered_img = np.where(phase<threshold,0,1).reshape(self.height,self.width)
            image_fft = fftpack.fft2(filtered_img)
            image_fft2 = image_fft.copy()
            
            # Set r and c to be the number of rows and columns of the array.
            r, c = image_fft2.shape
        
            # Set to zero all rows with indices between r*keep_fraction and
            # r*(1-keep_fraction):
            image_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
        
            # Similarly with the columns:
            image_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
        
            # Transformed the filtered image back to real space
            image_new = fftpack.ifft2(image_fft2).real
        
            binary_map_indices = np.where(image_new>binary_filter_threshold)
            
        # Get edx profile in the filtered phase region
        x_id = binary_map_indices[0].reshape(-1,1)
        y_id = binary_map_indices[1].reshape(-1,1)
        x_y = np.concatenate([x_id, y_id],axis=1)
        x_y_indices = tuple(map(tuple, x_y))
        
        
        shape = self.edx.inav[0,0].data.shape
        masked_edx = self.edx.deepcopy()
        for x_y_index in x_y_indices:
            masked_edx.data[x_y_index] = np.zeros(shape)
        
        return masked_edx
    
    def phase_statics(self, cluster_num,element_peaks=['Fe_Ka','O_Ka'],binary_filter_args={}):
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
        binary_map, _, _ = self.get_binary_map_edx_profile(cluster_num, **binary_filter_args)
        pixel_to_um = self.edx.axes_manager[0].scale
        prop_list = ['area',
                     'equivalent_diameter', 
                     'major_axis_length',
                     'minor_axis_length',
                     'min_intensity',
                     'mean_intensity',
                     'max_intensity']
        
        label_binary_map = measure.label(binary_map,connectivity=2)
        element_maps = self.sem.get_feature_maps()
        
        # Create a dataframe to record all statical information
        stat_info = dict()
        
        # for each element, create an individual element intensity statics
        for i, element in enumerate(element_peaks):
            element_idx = self.sem.feature_dict[element]
            clusters = measure.regionprops(label_image=label_binary_map,intensity_image=element_maps[:,:,element_idx])
            
            # for the first iteration, record everything apart from elemental intensity i.e. area, length ...
            if i == 0: 
                for prop in prop_list:
                    if prop =='area':
                        stat_info[f'{prop} (um^2)'] = [cluster[prop]*pixel_to_um**2 for cluster in clusters]
                    
                    elif prop in ['equivalent_diameter','major_axis_length','minor_axis_length']:
                        stat_info[f'{prop} (um)'] = [cluster[prop]*pixel_to_um for cluster in clusters]
                    
                    elif prop in ['min_intensity','mean_intensity','max_intensity']:
                        stat_info[f'{prop}_{element}'] = [cluster[prop] for cluster in clusters]
            
            # for the remaining iteration, only add elemental intensity into the dict()
            else:
                for prop in ['min_intensity','mean_intensity','max_intensity']:
                    stat_info[f'{prop}_{element}'] = [cluster[prop] for cluster in clusters]
                
        return pd.DataFrame(data=stat_info).round(3)
        
        
    
#################
# Visualization #--------------------------------------------------------------
#################
    
    def plot_latent_space(self, ax=None, save=None, **kwargs):
        fig, axs = plt.subplots(1,1,figsize=(4,4),dpi=100, **kwargs)
        ax = axs or plt.gca()
        label = self.model.predict(self.latent)
        
        ax.scatter(self.latent[:, 0], self.latent[:, 1], 
                   c=label, s=1., zorder=2,alpha=.15,
                   linewidths=0, cmap=self.color_palette,
                   norm=self.color_norm)
        
        # ax.axis('equal')
        if self.method in ['GaussianMixture', 'BayesianGaussianMixture']:
            for pos, covar, w in zip(self.model.means_, self.model.covariances_, self.model.weights_):
                self.draw_ellipse(pos, covar, alpha= 0.12, facecolor='slategrey', zorder=-10)
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.01)
            
    def plot_latent_density(self, bins=50):
        z = np.histogram2d(x=self.latent[:,0], y=self.latent[:,1],bins=bins)[0]
        sh_0, sh_1 = z.shape
        x, y = np.linspace(self.latent[:,0].min(), self.latent[:,0].max(), sh_0), np.linspace(self.latent[:,1].min(), self.latent[:,1].max(), sh_1)
        fig = go.Figure(data=[go.Surface(z=z.T, 
                                         x=x,
                                         y=y,
                                         colorscale ='RdBu_r')])
        fig.update_layout(title='Density of pixels in latent space', autosize=True,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
    
    
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
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))
    
    def plot_cluster_distribution(self, save=None, **kwargs):
        labels = self.model.predict(self.latent)
        means = []
        dataset_ravel = self.dataset.reshape(-1,self.dataset.shape[2])
        for i in range(self.n_components):
            mean = dataset_ravel[np.where(labels==i)[0]].mean(axis=0)
            means.append(mean.reshape(1,-1))
        mu = np.concatenate(means,axis=0)
        
        if self.method in ['GaussianMixture', 'BayesianGaussianMixture']:
            prob_map = self.model.predict_proba(self.latent)
    
        fig, axs = plt.subplots(self.n_components, 2, figsize=(14, self.n_components*4.2),dpi=96, **kwargs)
        fig.subplots_adjust(hspace=0.35, wspace=0.1)
        
        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((-1,1))
        
        for i in range(self.n_components):
            if self.method in ['GaussianMixture', 'BayesianGaussianMixture']:
                prob_map_i = prob_map[:,i]
            else:
                prob_map_i = np.where(labels==i,1,0)
            im=axs[i,0].imshow(prob_map_i.reshape(self.height, self.width), cmap='viridis')
            axs[i,0].set_title('Probability of each pixel for cluster '+str(i))
            
            axs[i,0].axis('off')
            cbar = fig.colorbar(im,ax=axs[i,0], shrink=0.9, pad=0.025)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=10, size=0)
            
            if self.n_components <= 10:
                axs[i,1].bar(self.sem.feature_list, mu[i], width=0.6, 
                             color = plt.cm.get_cmap(self.color_palette)(i*0.1))
            else:
                axs[i,1].bar(self.sem.feature_list, mu[i], width=0.6, 
                             color = plt.cm.get_cmap(self.color_palette)(i*(self.n_components-1)**-1))
                             
            axs[i,1].set_title('Mean value for cluster '+str(i))
    
        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.show()
        
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.01)
    
    def plot_single_cluster_distribution(self, cluster_num, kwargs={}):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3),dpi=96, **kwargs)
        fig.subplots_adjust(hspace=0.35, wspace=0.1)
        
        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((-1,1))
        
        
        if self.method in ['GaussianMixture', 'BayesianGaussianMixture']:
            prob_map_i = self.prob_map[:,cluster_num]
        else:
            prob_map_i = np.where(self.labels==cluster_num,1,0)
        im=axs[0].imshow(prob_map_i.reshape(self.height, self.width), cmap='viridis')
        axs[0].set_title('Probability of each pixel for cluster '+str(cluster_num))

        axs[0].axis('off')
        cbar = fig.colorbar(im,ax=axs[0], shrink=0.9, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10, size=0)

        if self.n_components <= 10:
            axs[1].bar(self.sem.feature_list, self.mu[cluster_num], width=0.6, 
                         color = plt.cm.get_cmap(self.color_palette)(cluster_num*0.1))
        else:
            axs[1].bar(self.sem.feature_list, self.mu[cluster_num], width=0.6, 
                         color = plt.cm.get_cmap(self.color_palette)(cluster_num*(self.n_components-1)**-1))

        axs[1].set_title('Mean value for cluster '+str(cluster_num))
    
        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.show()
    
    
    
    def plot_phase_map(self, not_to_show=[],save=None, **kwargs):
        
        img = self.bse.data
        phase = self.model.predict(self.latent).reshape(self.height, self.width)
    
        fig, axs = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(8,4), dpi=100, **kwargs)
        
        for i in not_to_show:
            phase[np.where(phase==i)]=0
            
        axs[0].imshow(img,cmap='gray',interpolation='none')
        axs[0].set_title('SE')
        axs[0].axis('off') 
    
        axs[1].imshow(img,cmap='gray',interpolation='none',alpha=1.)
        
        if self.n_components <= 10:
            axs[1].imshow(phase,cmap=self.color_palette ,interpolation='none',
                          norm=self.color_norm, alpha=0.75)
        else:
            axs[1].imshow(phase,cmap=self.color_palette ,interpolation='none',
                          alpha=0.6, norm=self.color_norm)
        axs[1].axis('off')   
        axs[1].set_title('Cluster map')
    
        fig.subplots_adjust(wspace=0.05, hspace=0.)
        plt.show()
        
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.02)
    
    
    
    def plot_binary_map_edx_profile(self, 
                                    cluster_num, 
                                    binary_filter_args={'threshold':0.8, 
                                                        'denoise':False, 'keep_fraction':0.13, 
                                                        'binary_filter_threshold':0.2},
                                    save=None,
                                    **kwargs):
        
        binary_map, binary_map_indices, edx_profile = self.get_binary_map_edx_profile(cluster_num, 
                                                                                      **binary_filter_args)
        

        fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(10,3), dpi=96,
                                gridspec_kw={'width_ratios': [1, 1, 2]},**kwargs) 
        
        phase_color = plt.cm.get_cmap(self.color_palette)(cluster_num*(self.n_components-1)**-1)
        c = mcolors.ColorConverter().to_rgb
        cmap = make_colormap([c('k'), phase_color[:3], 1, phase_color[:3]])
        
        axs[0].imshow(binary_map, cmap=cmap)
        axs[0].set_title(f'Binary map (cluster {cluster_num})',fontsize=10)
        axs[0].axis('off')
        axs[0].set_aspect('equal', 'box')
    
        axs[1].imshow(self.sem.bse_bin.data, cmap='gray',interpolation='none', alpha=1)
        axs[1].scatter(binary_map_indices[1], binary_map_indices[0], c='r', alpha=0.05, s=1.2)
        axs[1].grid(False)
        axs[1].axis('off')
        axs[1].set_title('SE + Binary Map',fontsize=10)
        
        intensity = edx_profile['intensity'].to_numpy()
        axs[2].set_xticks(np.arange(0, 11, step=1))
        axs[2].set_yticks(np.arange(0, int(intensity.max())+1, 
                                    step=int((intensity.max()/5))))
        
        axs[2].set_xticklabels(np.arange(0, 11, step=1), fontsize=8)
        axs[2].set_yticklabels(np.arange(0,  int(intensity.max())+1, 
                                         step=int((intensity.max()/5))), fontsize=8)
        
        offset = self.edx.axes_manager[2].offset
        axs[2].set_xlim(0,8)
        axs[2].set_ylim(None,intensity.max()*1.2)
        axs[2].set_xlabel('Energy / keV', fontsize=10)
        axs[2].set_ylabel('X-rays / Counts', fontsize=10)
        
        if self.n_components <= 10:
            axs[2].plot(edx_profile['energy'], edx_profile['intensity'], 
                    linewidth=1,color=plt.cm.get_cmap(self.color_palette)(cluster_num*0.1))
        else:
            axs[2].plot(edx_profile['energy'], edx_profile['intensity'], 
                        linewidth=1,
                        color= plt.cm.get_cmap(self.color_palette)(cluster_num*(self.n_components-1)**-1))
        
        zero_energy_idx = np.where(np.array(edx_profile['energy']).round(2)==0)[0][0]
        for el in self.peak_list:
            peak = intensity[zero_energy_idx:][int(self.peak_dict[el]*100)+1]
            axs[2].vlines(self.peak_dict[el], 0, int(0.9*peak), linewidth=0.7, color = 'grey', linestyles='dashed')
            axs[2].text(self.peak_dict[el]-0.125, peak+(int(intensity.max())/10), el, rotation='vertical', fontsize=7.5)
        
        # fig.subplots_adjust(left=0.1)
        plt.tight_layout()
        
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.02)
            
        plt.show()
    
    def plot_binary_map(self, cluster_num, 
                        binary_filter_args={'threshold':0.8, 
                                            'denoise':False, 'keep_fraction':0.13, 
                                            'binary_filter_threshold':0.2},
                        save=None,
                        **kwargs):
        
        binary_map, binary_map_indices, edx_profile = self.get_binary_map_edx_profile(cluster_num, 
                                                                                      **binary_filter_args)
        
        fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(5,2.5), dpi=96,**kwargs) 

        axs[0].imshow(binary_map, interpolation='none', alpha=1)
        axs[0].set_title(f'Filtered Binary map (cluster {cluster_num})')
        axs[0].axis('off')
        axs[0].set_aspect('equal', 'box')
    
        axs[1].imshow(self.sem.bse_bin.data, cmap='gray',interpolation='none', alpha=1)
        axs[1].scatter(binary_map_indices[1], binary_map_indices[0], c='r', alpha=0.05, s=1.2)
        axs[1].grid(False)
        axs[1].axis('off')
        axs[1].set_title(f'BSE + Phase Map (cluster {cluster_num})')
        
        # fig.subplots_adjust(left=0.1)
        plt.tight_layout()
        
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.02)
            
        plt.show()
        
    def plot_unmixed_profile(self, components, peak_list = []):
        if len(peak_list) == 0:
            peak_list = self.peak_list
        cpnt_num = len(components.columns.to_list())
        if cpnt_num > 4:
            n_rows = (cpnt_num+3)//4
            n_cols = 4
        else:
            n_rows = 1
            n_cols = cpnt_num

        fig, axs = plt.subplots(n_rows, n_cols,figsize=(n_cols*3.6, n_rows*2.6),dpi=150)
        for row in range(n_rows):
            for col in range(n_cols):
                cur_cpnt = (row*n_cols)+col
                if cur_cpnt>cpnt_num-1: # delete the extra subfigures
                    fig.delaxes(axs[row,col]) 
                else:
                    cpnt = f'cpnt_{cur_cpnt}'
                    if cpnt_num > 4:
                        axs_sub = axs[row,col]
                    else:
                        axs_sub = axs[col]
                    axs_sub.plot(self.energy_axis, components[cpnt], linewidth=1)
                    axs_sub.set_xlim(0,8)
                    axs_sub.set_ylim(None,components[cpnt].max()*1.3)
                    axs_sub.set_ylabel('Intensity')
                    axs_sub.set_xlabel('Energy (keV)')
                    axs_sub.set_title(f'cpnt_{cur_cpnt}')

                    zero_energy_idx = np.where(np.array(self.energy_axis).round(2)==0)[0][0]
                    intensity = components[cpnt].to_numpy()
                    for el in peak_list:
                        peak = intensity[zero_energy_idx:][int(self.peak_dict[el]*100)+1]
                        axs_sub.vlines(self.peak_dict[el], 0, 0.9*peak, linewidth=1, color = 'grey', linestyles='dashed')
                        axs_sub.text(self.peak_dict[el]-0.18, peak+(intensity.max()/15), el, rotation='vertical', fontsize=8)

        fig.subplots_adjust(hspace=0.3, wspace=0.)
        plt.tight_layout()
        plt.show()
    
    
    def plot_edx_profile(self, cluster_num, peak_list, binary_filter_args):
        edx_profile = self.get_binary_map_edx_profile(cluster_num, **binary_filter_args)[2]
        intensity = edx_profile['intensity'].to_numpy()
        
        fig, axs = plt.subplots(1,1,figsize=(4,2),dpi=150)
        axs.set_xticks(np.arange(0, 12, step=1))
        axs.set_yticks(np.arange(0, int(intensity.max())+1, step=int((intensity.max()/5))))

        axs.set_xticklabels(np.arange(0, 12, step=1), fontsize=8)
        axs.set_yticklabels(np.arange(0, int(intensity.max())+1, step=int((intensity.max()/5))), fontsize=8)

        offset = self.edx.axes_manager[2].offset
        axs.set_xlim(0,8)
        axs.set_ylim(None, intensity.max()*1.25)
        axs.set_xlabel('Energy axis / keV', fontsize=10)
        axs.set_ylabel('X-rays / Counts', fontsize=10)

        if self.n_components <= 10:
            axs.plot(edx_profile['energy'], edx_profile['intensity'], 
                    linewidth=1,color=plt.cm.get_cmap(self.color_palette)(cluster_num*0.1))
        else:
            axs.plot(edx_profile['energy'], edx_profile['intensity'], 
                        linewidth=1,
                        color= plt.cm.get_cmap(self.color_palette)(cluster_num*(self.n_components-1)**-1))

        zero_energy_idx = np.where(np.array(edx_profile['energy']).round(2)==0)[0][0]
        for el in peak_list:
            peak = intensity[zero_energy_idx:][int(self.peak_dict[el]*100)+1]
            axs.vlines(self.peak_dict[el], 0, int(0.9*peak), linewidth=0.7, color = 'grey', linestyles='dashed')
            axs.text(self.peak_dict[el]-0.075, peak+(int(intensity.max())/20), el, rotation='vertical', fontsize=7.5)
        plt.show()
