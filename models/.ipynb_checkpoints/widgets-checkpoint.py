#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from load_dataset.exhaust import plot_profile
from models.clustering import PhaseClassifier

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from  matplotlib import pyplot as plt

ALL = 'ALL'

def unique_sorted_values_plus_ALL(array):
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique
    

def show_unmixed_weights(weights:pd.DataFrame):
    dropdown_cluster = widgets.Dropdown(options=unique_sorted_values_plus_ALL(weights.index))
    output_cluster = widgets.Output()
    plots_output = widgets.Output()
    
    with output_cluster:
        display(weights)
    def dropdown_cluster_eventhandler(change):
        output_cluster.clear_output()
        with output_cluster:
            if (change.new == ALL):
                display(weights)
            else:
                display(weights[weights.index == change.new])
    
        plots_output.clear_output()
        with plots_output:
            if (change.new != ALL):
                num_cpnt = len(weights.columns.to_list())
                fig, axs = plt.subplots(1,1,figsize=(4,3),dpi=96)
                axs.bar(np.arange(0,num_cpnt), weights[weights.index == change.new].to_numpy().ravel(), width=0.6)
                axs.set_xticks(np.arange(0,num_cpnt))
                axs.set_ylabel('weight of component')
                axs.set_xlabel('component number')
                plt.show()
    
    dropdown_cluster.observe(dropdown_cluster_eventhandler, names='value')
    
    display(dropdown_cluster)
    display(output_cluster)
    display(plots_output)
    
def show_unmixed_components(PC:PhaseClassifier, components:pd.DataFrame):
    dropdown_cluster = widgets.Dropdown(options=unique_sorted_values_plus_ALL(components.columns))
    plots_output = widgets.Output()
    
    with plots_output:
        PC.plot_unmixed_profile(components)
    def dropdown_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            if (change.new == ALL):
                PC.plot_unmixed_profile(components)
            else:
                plot_profile(PC.energy_axis, components[change.new], PC.peak_list)
    
    dropdown_cluster.observe(dropdown_cluster_eventhandler, names='value')
    
    display(dropdown_cluster)
    display(plots_output)

def show_clusters(PC:PhaseClassifier,binary_filter_args):
    cluster_options = [f'cluster_{n}' for n in range(PC.n_components)]
    dropdown_cluster = widgets.Dropdown(options=['ALL']+cluster_options)
    plots_output = widgets.Output()
    
    with plots_output:
        for i in range(PC.n_components):
            PC.plot_binary_map(cluster_num=i, binary_filter_args=binary_filter_args)
        
    def dropdown_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            if (change.new == ALL):
                pass
            else:
                PC.plot_binary_map(cluster_num=int(change.new.split('_')[1]), binary_filter_args=binary_filter_args)
        
    dropdown_cluster.observe(dropdown_cluster_eventhandler, names='value')
    display(dropdown_cluster)
    display(plots_output)
    
def show_clusters(PC:PhaseClassifier,binary_filter_args):
    cluster_options = [f'cluster_{n}' for n in range(PC.n_components)]
    multi_select = widgets.SelectMultiple(options=cluster_options)
    plots_output = widgets.Output()
    profile_output = widgets.Output()
        
    def eventhandler(change):
        plots_output.clear_output()
        profile_output.clear_output()
        
        with plots_output:
            for cluster in change.new:
                PC.plot_binary_map_edx_profile(cluster_num=int(cluster.split('_')[1]), binary_filter_args=binary_filter_args)
                
        with profile_output:
            ### X-ray profile ###
            for cluster in change.new:
                _,_, edx_profile = PC.get_binary_map_edx_profile(cluster_num=int(cluster.split('_')[1]),
                                                                 **binary_filter_args)
                plot_profile(edx_profile['energy'], edx_profile['intensity'], PC.peak_list)
        
    multi_select.observe(eventhandler, names='value')
    
    display(multi_select)
    tab = widgets.Tab([plots_output, profile_output])
    tab.set_title(0, 'clusters + edx')
    tab.set_title(1, 'edx')
    display(tab)
    # display(plots_output)
    # display(profile_output)