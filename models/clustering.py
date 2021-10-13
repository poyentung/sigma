#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from load_dataset.exhaust import SEMDataset

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
from sklearn.cluster import MeanShift
from skimage import io, img_as_float, measure
from scipy import fftpack

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import seaborn as sns

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
        
        
        if self.method == 'GaussianMixture':
            self.model = GaussianMixture(**method_args).fit(self.latent)
            
        self.peak_dict = {'Al_Ka': 1.49, 'C_Ka' : 0.28, 'Ca_Ka': 3.69,
                          'Cr_Ka': 5.41, 'Fe_Ka': 6.40, 'Fe_La': 0.70, 
                          'Mg_Ka': 1.25, 'N_Ka': 0.39, 'O_Ka': 0.52, 
                          'P_Ka': 2.01, 'S_Ka': 2.31, 'Si_Ka': 1.74}
        
        self.peak_list = ['O_Ka','Fe_Ka','Mg_Ka','Ca_Ka', 'Al_Ka', 
                          'C_Ka', 'Si_Ka','S_Ka','Fe_La']
        

#################
# Data Analysus #--------------------------------------------------------------
#################

    def get_binary_map_edx_profile(self, cluster_num=0, threshold=0.8, 
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
    
    
    def phase_statics(self, cluster_num,element_peaks=['Fe_Ka','O_Ka'],**binary_filter_args):
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
        binary_map, _, _ = self.get_binary_map_edx_profile(cluster_num, 
                                                           **binary_filter_args)
        pixel_to_um = self.edx.axes_manager[0].scale
        prop_list = ['area',
                     'equivalent_diameter', #Added... verify if it works
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
                        stat_info[prop] = [cluster[prop]*pixel_to_um**2 for cluster in clusters]
                    
                    elif prop in ['equivalent_diameter','major_axis_length','minor_axis_length']:
                        stat_info[prop] = [cluster[prop]*pixel_to_um for cluster in clusters]
                    
                    elif prop in ['min_intensity','mean_intensity','max_intensity']:
                        stat_info[f'{prop}_{element}'] = [cluster[prop] for cluster in clusters]
            
            # for the remaining iteration, only add elemental intensity into the dict()
            else:
                for prop in ['min_intensity','mean_intensity','max_intensity']:
                    stat_info[f'{element}_{prop}'] = [cluster[prop] for cluster in clusters]
                
        return pd.DataFrame(data=stat_info)
        
        
    
#################
# Visualization #--------------------------------------------------------------
#################
    
    def plot_latent_space(self, ax=None, save=None, **kwargs):
        fig, axs = plt.subplots(1,1,figsize=(4,4),dpi=200, **kwargs)
        ax = axs or plt.gca()
        label = self.model.predict(self.latent)
        
        ax.scatter(self.latent[:, 0], self.latent[:, 1], 
                   c=label, s=1., zorder=2,alpha=.15,
                   linewidths=0, cmap='tab10',
                   norm=mpl.colors.Normalize(vmin=0, vmax=10))
        
        ax.axis('equal')
        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)
        
        for pos, covar, w in zip(self.model.means_, self.model.covariances_, self.model.weights_):
            self.draw_ellipse(pos, covar, alpha= 0.12, facecolor='slategrey', zorder=-10)
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.01)
    
    
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
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))
    
    def plot_phase_distribution(self, save=None, **kwargs):
        n_component = self.model.n_components
        labels = self.model.predict(self.latent)
        means = []
        dataset_ravel = self.dataset.reshape(-1,self.dataset.shape[2])
        for i in range(n_component):
            mean = dataset_ravel[np.where(labels==i)[0]].mean(axis=0)
            means.append(mean.reshape(1,-1))
        mu = np.concatenate(means,axis=0)
        prob_map = self.model.predict_proba(self.latent)
    
        fig, axs = plt.subplots(n_component, 2, figsize=(14, n_component*4.2),dpi=96, **kwargs)
        fig.subplots_adjust(hspace=0.35, wspace=0.1)
        
        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((-1,1))
        
        for i in range(n_component):
            im=axs[i,0].imshow(prob_map[:,i].reshape(self.height, self.width), cmap='viridis')
            axs[i,0].set_title('Probability of each pixel for cluster '+str(i+1))
            
            axs[i,0].axis('off')
            cbar = fig.colorbar(im,ax=axs[i,0], shrink=0.9, pad=0.025)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=10, size=0)
    
            axs[i,1].bar(self.sem.feature_list, mu[i],width=0.6, color=plt.cm.get_cmap('tab10')(i*0.1))
            axs[i,1].set_title('Mean value for cluster '+str(i+1))
    
        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.show()
        
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.01)
    
    
    
    def plot_phase_map(self, save=None, **kwargs):
        
        img = self.bse.data
        phase = self.model.predict(self.latent).reshape(self.height, self.width)
    
        fig, axs = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(8,16),**kwargs)
    
        axs[0].imshow(img,cmap='gray',interpolation='none')
        axs[0].set_title('BSE')
    
        axs[1].imshow(img,cmap='gray',interpolation='none',alpha=1.)
        axs[1].imshow(phase,cmap='tab10',interpolation='none',norm=mpl.colors.Normalize(vmin=0, vmax=10),alpha=0.75)
        axs[1].set_title('Phase map')
    
        fig.subplots_adjust(wspace=0.05, hspace=0.)
        plt.show()
        
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.02)
    
    
    
    def plot_binary_map_edx_profile(self, cluster_num, 
                                    binary_filter_args={'threshold':0.8, 
                                                        'denoise':False, 'keep_fraction':0.13, 
                                                        'binary_filter_threshold':0.2},
                                    save=None,
                                    **kwargs):
        
        binary_map, binary_map_indices, edx_profile = self.get_binary_map_edx_profile(cluster_num, 
                                                                                      **binary_filter_args)
        
        fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(9,3), dpi=150,
                                gridspec_kw={'width_ratios': [1, 1, 2]},**kwargs) 

        axs[0].imshow(binary_map, interpolation='none', alpha=1)
        axs[0].set_title('Filtered Binary map')
        axs[0].axis('off')
        axs[0].set_aspect('equal', 'box')
    
        axs[1].imshow(self.sem.bse_bin.data, cmap='gray',interpolation='none', alpha=1)
        axs[1].scatter(binary_map_indices[1], binary_map_indices[0], c='r', alpha=0.05, s=1.2)
        axs[1].grid(False)
        axs[1].axis('off')
        axs[1].set_title('BSE + Phase Map')
        
        intensity = edx_profile['intensity'].to_numpy()
        axs[2].set_xticks(np.arange(0, 11, step=1))
        axs[2].set_yticks(np.arange(0, int(intensity.max())+1, 
                                    step=int((intensity.max()/5))))
        
        axs[2].set_xticklabels(np.arange(0, 11, step=1), fontsize=8)
        axs[2].set_yticklabels(np.arange(0,  int(intensity.max())+1, 
                                         step=int((intensity.max()/5))), fontsize=8)
        
        offset = self.edx.axes_manager[2].offset
        axs[2].set_xlim(offset,8)
        axs[2].set_xlabel('Energy axis / keV', fontsize=10)
        axs[2].set_ylabel('X-rays / Counts', fontsize=10)

        axs[2].plot(edx_profile['energy'], edx_profile['intensity'], 
                    linewidth=1,color=sns.color_palette()[cluster_num])
        
        zero_energy_idx = np.where(np.array(edx_profile['energy']).round(2)==0)[0][0]
        for el in self.peak_list:
            peak = intensity[zero_energy_idx:][int(self.peak_dict[el]*100)+1]
            axs[2].vlines(self.peak_dict[el], 0, int(0.9*peak), linewidth=0.7, color = 'grey', linestyles='dashed')
            axs[2].text(self.peak_dict[el]-0.125, peak+(int(intensity.max())/20), el, rotation='vertical', fontsize=7.5)
        
        # fig.subplots_adjust(left=0.1)
        plt.tight_layout()
        
        if save is not None:
            fig.savefig(save, bbox_inches = 'tight', pad_inches=0.02)
            
        fig.show()
    
