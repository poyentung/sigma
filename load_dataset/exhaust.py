# -*- coding: utf-8 -*-

import hyperspy.api as hs
from hyperspy._signals.eds_sem import EDSSEMSpectrum

import numpy as np
import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go

peak_dict = dict()
for element in hs.material.elements:
    if element[0]=='Li': continue
    for character in element[1].Atomic_properties.Xray_lines:
        peak_name = element[0]
        char_name = character[0]
        key = f'{peak_name}_{char_name}'
        peak_dict[key] = character[1].energy_keV

class SEMDataset(object):
    def __init__(self, file_path:str):
        bcf_dataset = hs.load(file_path)
        self.bse = bcf_dataset[0] #load BSE data
        self.edx = bcf_dataset[2] #load EDX data from bcf file
        self.edx.change_dtype('float32') # change edx data from unit8 into float32
        
        self.edx_bin = None
        self.bse_bin = None
        
        self.feature_list = self.edx.metadata.Sample.xray_lines
        
        self.feature_dict = {el:i for (i,el) in enumerate(self.feature_list)}
    
    def set_feature_list(self, feature_list):
        self.feature_list = feature_list
        self.feature_dict = {el:i for (i,el) in enumerate(self.feature_list)}
        for s in [self.edx, self.edx_bin]:
            if s is not None:
                s.metadata.Sample.xray_lines = self.feature_list
        print(f'Set feature_list as {self.feature_list}')
    
    def rebin_signal(self, size=(2,2)):
        print(f'Rebinning the intensity with the size of {size}')
        x, y = size[0], size[1]
        self.edx_bin = self.edx.rebin(scale=(x, y, 1))
        self.bse_bin = self.bse.rebin(scale=(x, y))
        return (self.edx_bin, self.bse_bin)
    
    
    def remove_fist_peak(self, end:float):
        print(f'Removing the fisrt peak by setting the intensity to zero until the energy of {end} keV.')
        for edx in (self.edx, self.edx_bin):
            scale = edx.axes_manager[2].scale
            offset = edx.axes_manager[2].offset
            end_ = int((end-offset)/scale)
            for i in range(end_):
                edx.isig[i] = 0
    
    def get_feature_maps(self, feature_list=None) -> np:
        if feature_list is not None:
            self.set_feature_list(feature_list)
            
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
  
<<<<<<< HEAD
def remove_fist_peak(edx:EDSSEMSpectrum, end=0.1197) -> EDSSEMSpectrum:
=======
def remove_fist_peak(edx:EDSSEMSpectrum, end=0.01197) -> EDSSEMSpectrum:
>>>>>>> 60ce5cd2dc24d3f9d6c62a37d84752d8bb0859fe
    print(f'Removing the fisrt peak by setting the intensity to zero until the energy of {end} keV.')
    edx_cleaned = edx
    scale = edx.axes_manager[2].scale
    offset = edx.axes_manager[2].offset
    end_ = int((end-offset)/scale)
    for i in range(end_):
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


def intensity_normalisation(dataset:np) -> np:
    dataset_norm = dataset.copy()
    dataset_norm = dataset_norm / dataset_norm.sum(axis=2)
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


def z_score_normalisation(dataset:np) -> np:
    dataset_norm = dataset.copy()
    for i in range(dataset_norm.shape[2]):
        mean = dataset_norm[:,:,i].mean()
        std = dataset_norm[:,:,i].std()
        dataset_norm[:,:,i] = (dataset_norm[:,:,i] - mean) / std
    return dataset_norm


def softmax(dataset:np) -> np:
    exp_dataset = np.exp(dataset)
    sum_exp = np.sum(exp_dataset,axis=2)
    sum_exp = np.expand_dims(sum_exp,axis=2)
    sum_exp = np.tile(sum_exp, (1,1,dataset.shape[2]))
    softmax_out = exp_dataset / sum_exp
    return softmax_out


#################
# Visualization #--------------------------------------------------------------
#################

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def plot_sum_spectrum(edx, xray_lines=True):       
    size = edx.axes_manager[2].size
    scale = edx.axes_manager[2].scale
    offset = edx.axes_manager[2].offset
    energy_axis = [((a*scale) + offset) for a in range(0,size)]
    
    fig = go.Figure(data=go.Scatter(x=energy_axis, y=edx.sum().data),
                    layout_xaxis_range=[offset,8],
                    layout=go.Layout(title="EDX Sum Spectrum",
                                     title_x=0.5,
                                     xaxis_title="Energy / keV",
                                     yaxis_title="Counts",
                                     width=900,
                                     height=500))
    
    if xray_lines:
        feature_list = edx.metadata.Sample.xray_lines
        zero_energy_idx = np.where(np.array(energy_axis).round(2)==0)[0][0]
        for el in feature_list:
            peak = edx.sum().data[zero_energy_idx:][int(peak_dict[el]*100)+1]
            fig.add_shape(type="line",
                          x0=peak_dict[el], y0=0, x1=peak_dict[el], y1=int(0.9*peak),
                          line=dict(color="black",
                                    width=2,
                                    dash="dot")
                          )
        
            fig.add_annotation(x=peak_dict[el], y=peak,
                               text=el,
                               showarrow=False,
                               arrowhead=2,
                               yshift=30,
                               textangle=270
                               )
        
    fig.update_layout(showlegend=False)
    fig.update_layout(template='simple_white')
    fig.show()

def plot_intensity_maps(edx, element_list, grid_dims=(2,4), save=None):
    cmaps = []
    c = mcolors.ColorConverter().to_rgb
    for i in range(len(element_list)):
        rvb = make_colormap([c('k'),sns.color_palette('bright')[i], 0.7, sns.color_palette('bright')[i]])
        cmaps.append(rvb)
    nrow = grid_dims[0]
    ncol = grid_dims[1]

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True, 
                            figsize=(4*ncol,3.3*nrow))
    for i in range(nrow):
      for j in range(ncol):
        el = element_list[(i*ncol)+j]
        el_map = edx.get_lines_intensity([el])[0].data
        im = axs[i,j].imshow(el_map, cmap='viridis')#cmaps[(i*ncol)+j])
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
  

def plot_pixel_distributions(sem:SEMDataset, peak='Fe_Ka', **kwargs):
    idx = sem.feature_dict[peak]
    sns.set_style('ticks')
    dataset = sem.get_feature_maps()
    dataset_avg = avgerage_neighboring_signal(dataset)
    dataset_ins = z_score_normalisation(dataset_avg) 
    dataset_softmax = softmax(dataset_ins)
    
    dataset_list= [dataset, dataset_avg, dataset_ins, dataset_softmax]
    dataset_lable=['Original', 'Neighbour Intensity Averging', 'Z-score Normalisation', 'Softmax']
    
    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 

    fig, axs = plt.subplots(2,len(dataset_list), figsize=(4*len(dataset_list),6), dpi=100, gridspec_kw={'height_ratios': [2, 1.5]})
    #fig.suptitle(f'Intensity Distribution of {feature_list[idx]}', y = .93)
    
    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        im = axs[0,i].imshow(dataset[:,:,idx].round(2),cmap='viridis')
        axs[0,i].set_aspect('equal')
        axs[0,i].set_title(f'{dataset_lable[i]}')
        
        axs[0,i].axis('off')
        cbar = fig.colorbar(im,ax=axs[0,i], shrink=0.83, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10, size=0)
    
    for j in range(len(dataset_list)):
        dataset = dataset_list[j]
        sns.histplot(dataset[:,:,idx].ravel(),ax=axs[1,j], bins=50, **kwargs)
        
        axs[1,j].set_xlabel('Element Intensity')
        axs[1,j].yaxis.set_major_formatter(formatter)
        if j!=0:
            axs[1,j].set_ylabel(' ')
    
    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    

def plot_profile(energy, intensity, peak_list):
    fig = go.Figure(data=go.Scatter(x=energy, y=intensity),
                        layout_xaxis_range=[0,8],
                        layout=go.Layout(title="",
                                        title_x=0.5,
                                        xaxis_title="Energy / keV",
                                        yaxis_title="Intensity",
                                        width=900,
                                        height=500))
    zero_energy_idx = np.where(np.array(energy).round(2)==0)[0][0]
    for el in peak_list:
        peak = intensity[zero_energy_idx:][int(peak_dict[el]*100)+1]
        fig.add_shape(type="line",
                    x0=peak_dict[el], y0=0, x1=peak_dict[el], y1=0.9*peak,
                    line=dict(color="black",
                                width=2,
                                dash="dot")
                        )

        fig.add_annotation(x=peak_dict[el], y=peak,
                            text=el,
                            showarrow=False,
                            arrowhead=2,
                            yshift=30,
                            textangle=270
                            )  
    fig.update_layout(showlegend=False)
    fig.update_layout(template='simple_white')
    fig.show()