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
from skimage import io, img_as_float
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
            self.model = GaussianMixture(**method_args).fit(self.dataset)
            
        self.peak_dict = {'Al_Ka': 1.49, 'C_Ka' : 0.28, 'Ca_Ka': 3.69,
                          'Cr_Ka': 5.41, 'Fe_Ka': 6.40, 'Fe_La': 0.70, 
                          'Mg_Ka': 1.25, 'N_Ka': 0.39, 'O_Ka': 0.52, 
                          'P_Ka': 2.01, 'S_Ka': 2.31, 'Si_Ka': 1.74}
        
        self.peak_list = ['O_Ka','Fe_Ka','Mg_Ka','Ca_Ka', 'Al_Ka', 
                          'C_Ka', 'Si_Ka','S_Ka','Fe_La']
        
    
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
        for i in range(n_component):
            mean = self.dataset[np.where(labels==i)[0]].mean(axis=0)
            means.append(mean.reshape(1,-1))
        mu = np.concatenate(means,axis=0)
        prob_map = self.model.predict_proba(self.latent)
    
        fig, axs = plt.subplots(n_component, 2, figsize=(14, n_component*4.2),dpi=96, **kwargs)
        fig.subplots_adjust(hspace=0.35, wspace=0.1)
        for i in range(n_component):
            im=axs[i,0].imshow(prob_map[:,i].reshape(self.height, self.width), cmap='viridis')
            axs[i,0].set_title('Probability of each pixel for cluster '+str(i+1))
            fig.colorbar(im,ax=axs[i,0], shrink=0.8)
    
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
    
    
    def get_binary_map_and_edx_profile(self, cluster_num=0, threshold=0.8, 
                                       denoise=False,keep_fraction=0.13, 
                                       binary_filter_threshold=0.2):
        
        phase = self.model.predict_proba(self.latent)[:,cluster_num]
        
        if denoise == False:
            phase_map = np.where(phase<threshold,0,1).reshape(self.height,self.width)
            phase_indices = np.where(phase_map>threshold)
        
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
        
            phase_map = np.where(image_new<binary_filter_threshold,0,1)
            phase_indices = np.where(image_new>binary_filter_threshold)
            
        # Get edx profile in the filtered phase region
        x_id = phase_indices[0].reshape(-1,1)
        y_id = phase_indices[1].reshape(-1,1)
        x_y = np.concatenate([x_id, y_id],axis=1)
        phase_indices = tuple(map(tuple, x_y))
        
        total_edx_profiles = list()
        for xy in phase_indices:
            total_edx_profiles.append(self.edx.data[xy].reshape(1,-1))
        total_edx_profiles = np.concatenate(total_edx_profiles,axis=0)
        
        size = self.edx.axes_manager[2].size
        scale = self.edx.axes_manager[2].scale
        offset = self.edx.axes_manager[2].offset
        energy_axis = [((a*scale) + offset) for a in range(0,size)]
        element_intensity_sum = total_edx_profiles.sum(axis=0)
        
        
        return phase_map, pd.DataFrame()
    
    
    def get_edx_profile(self, num_cluster=3):
        
        total = list()
        for i in x_y_idx:
            total.append(self.edx.data[i].reshape(1,-1))
        total = np.concatenate(total,axis=0)
        
        size = edx_2xbin.axes_manager[2].size
        scale = edx_2xbin.axes_manager[2].scale
        offset = edx_2xbin.axes_manager[2].offset
        x = [((a*scale) + offset) for a in range(0,size)]
        intensity = total.sum(axis=0)
    
    

    
    
    
    
        