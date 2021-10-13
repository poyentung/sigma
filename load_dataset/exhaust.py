# -*- coding: utf-8 -*-

import hyperspy.api as hs
from hyperspy._signals.eds_sem import EDSSEMSpectrum

import numpy as np
import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

class SEMDataset(object):
    def __init__(self, file_path:str):
        bcf_dataset = hs.load(file_path)
        self.bse = bcf_dataset[0] #load BSE data
        self.edx = bcf_dataset[2] #load EDX data from bcf file
        self.edx.change_dtype('float32') # change edx data from unit8 into float32
        
        self.edx_bin = None
        self.bse_bin = None
        
        self.feature_list = ['O_Ka', 'Fe_Ka', 'Mg_Ka', 'Ca_Ka', 
                             'Al_Ka', 'C_Ka', 'Si_Ka', 'S_Ka']
    
    def rebin_signal(self, size=(2,2)):
        print(f'Rebinning the intensity with the size of {size}')
        x, y = size[0], size[1]
        self.edx_bin = self.edx.rebin(scale=(x, y, 1))
        self.bse_bin = self.bse.rebin(scale=(x, y))
        return (self.edx_bin, self.bse_bin)
    
    def get_feature_maps(self) -> np:
        num_elements=len(self.feature_list)
        
        if self.edx_bin is not None:
            lines = self.edx_bin.get_lines_intensity(self.feature_list)
        else:
            lines = self.edx.get_lines_intensity(self.feature_list)
            
        dims = lines[0].data.shape
        data_cube = np.zeros((dims[0], dims[1], num_elements))
    
        for i in range(num_elements):
            data_cube[:, :, i] = lines[i]
    
        return data_cube
    
######################
# Data Preprocessing #----------------------------------------------------------
######################
  
def remove_fist_peak(edx:EDSSEMSpectrum, range_idx=58) -> EDSSEMSpectrum:
    print(f'Removing the fisrt peak by setting the first {range_idx} as zero')
    edx_cleaned = edx
    for i in range(range_idx):
        edx_cleaned.isig[i] = 0
    return edx_cleaned


def peak_intensity_normalisation(edx:EDSSEMSpectrum) -> EDSSEMSpectrum:
    edx_norm = edx
    edx_norm.data = edx_norm.data / edx_norm.data.sum(axis=2, keepdims=True)
    return edx_norm


def peak_denoising_PCA(edx:EDSSEMSpectrum, 
                       n_components_to_reconstruct=10, 
                       plot_results=True) -> EDSSEMSpectrum:
    edx_denoised = edx
    edx_denoised.decomposition(normalize_poissonian_noise=True, 
                               algorithm='SVD', 
                               random_state=0, 
                               output_dimension=n_components_to_reconstruct)

    if plot_results == True:
        edx_denoised.plot_decomposition_results()
        edx_denoised.plot_explained_variance_ratio(log=True)
        edx_denoised.plot_decomposition_factors(comp_ids=4)
        
    return edx_denoised


def z_score_normalisation(dataset:np) -> np:
    dataset_norm = dataset.copy()
    for i in range(dataset_norm.shape[2]):
        mean = dataset_norm[:,:,i].mean()
        std = dataset_norm[:,:,i].std()
        dataset_norm[:,:,i] = (dataset_norm[:,:,i] - mean) / std
    return dataset_norm
        
def avgerage_neighboring_signal(dataset:np) -> np:
    h, w = dataset.shape[0],  dataset.shape[1]   
    new_dataset = np.zeros(shape=dataset.shape) # create an empty np.array for new dataset
    
    for row in range(h): # for each row
        for col in range(w): # for each column
            row_idxs=[row-1, row, row+1] 
            col_idxs=[col-1, col, col+1] # get indices from the neighboring (num=3*3)
            
            for row_idx in row_idxs:
                if row_idx < 0 or row_idx >= h:
                    row_idxs.remove(row_idx) # remove the pixels which is out ofthe boundaries
                    
            for col_idx in col_idxs:
                if col_idx < 0 or col_idx >= w:
                    col_idxs.remove(col_idx) # remove the pixels which is out ofthe boundaries
            
            # get positions using indices after the removal of pixels out of the boundaries
            positions = [pos for pos in itertools.product(row_idxs, col_idxs)]
            background_signal = []
            
            for k in positions:
                background_signal.append(dataset[k])
            background_signal = np.stack(background_signal,axis=0)
            background_signal_avg = np.sum(background_signal,axis=0) / background_signal.shape[0]

            new_dataset[row,col,:] = background_signal_avg
         
    return new_dataset


#################
# Visualization #--------------------------------------------------------------
#################

def plot_intensity_maps(edx, element_list, grid_dims=(2,4), save=None):
    cmaps = ['Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r', 
             'YlOrBr_r', 'YlOrRd_r', 'Blues_r', 'YlOrBr_r', 'Greens_r', 'Reds_r', 
             'Purples_r', 'pink', 'bone', 'viridis']
    nrow = grid_dims[0]
    ncol = grid_dims[1]

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, 
                            figsize=(4*ncol,3.3*nrow))
    for i in range(nrow):
      for j in range(ncol):
        el = element_list[(i*ncol)+j]
        el_map = edx.get_lines_intensity([el])[0].data
        im = axs[i,j].imshow(el_map, cmap=cmaps[(i*ncol)+j])
        axs[i,j].set_yticks([])
        axs[i,j].set_xticks([])
        axs[i,j].set_title(el, fontsize=16)
        fig.colorbar(im, ax=axs[i,j], shrink=0.75)

    fig.subplots_adjust(wspace=0.11, hspace=0.)
    
    if save is not None:
        fig.savefig(save, bbox_inches = 'tight', pad_inches=0.01)
        
    plt.show()
    
def plot_intensity_np(dataset, feature_list, save=None, **kwargs):
    n_cols = len(feature_list)
    fig, axs = plt.subplots(1, ncols=n_cols, sharex=True, sharey=True, 
                            figsize=(n_cols*2,2), dpi=100, **kwargs)
    for col in range(len(feature_list)):
        axs[col].set_title(feature_list[col])
        axs[col].imshow(dataset[:,:,col], cmap='viridis')
    
    fig.subplots_adjust(wspace=0.11, hspace=0.)
    plt.show()
    if save is not None:
        fig.savefig(save, bbox_inches = 'tight', pad_inches=0.01)
  

def plotDist(sem:SEMDataset, idx=0, **kwargs):
    sns.set_style('ticks')
    dataset = sem.get_feature_maps()
    dataset_avg = avgerage_neighboring_signal(dataset)
    dataset_ins = z_score_normalisation(dataset_avg) 
    
    dataset_list= [dataset, dataset_avg, dataset_ins]
    dataset_lable=['Original', 'Neighbour Intensity Averging', 'Z-score Normalisation']
    
    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 

    fig, axs = plt.subplots(2,3, figsize=(10,5), dpi=100, gridspec_kw={'height_ratios': [2, 1.5]})
    #fig.suptitle(f'Intensity Distribution of {feature_list[idx]}', y = .93)
    
    for i in range(3):
        dataset = dataset_list[i]
        im = axs[0,i].imshow(dataset[:,:,idx].round(2),cmap='viridis')
        axs[0,i].set_aspect('equal')
        axs[0,i].set_title(f'{dataset_lable[i]}')
        
        axs[0,i].axis('off')
        cbar = fig.colorbar(im,ax=axs[0,i], shrink=0.83, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10, size=0)
    
    for j in range(3):
        dataset = dataset_list[j]
        sns.histplot(dataset[:,:,idx].ravel(),ax=axs[1,j], **kwargs)
        
        axs[1,j].set_xlabel('Element Intensity')
        axs[1,j].yaxis.set_major_formatter(formatter)
        if j!=0:
            axs[1,j].set_ylabel(' ')
    
    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    
