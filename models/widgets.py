#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    

def show_unmixed_components(weights:pd.DataFrame):
    dropdown_cluster = widgets.Dropdown(options=unique_sorted_values_plus_ALL(weights.index))
    output_cluster = widgets.Output()
    plots_output = widgets.Output()
    
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