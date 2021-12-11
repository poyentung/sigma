#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from load_dataset.exhaust import plot_profile
from models.clustering import PhaseClassifier

import numpy as np
import pandas as pd
import random
import hyperspy.api as hs
import seaborn as sns
import altair as alt
import ipywidgets as widgets
from IPython.display import display
from matplotlib import cm
from  matplotlib import pyplot as plt

def search_energy_peak():
    text = widgets.BoundedFloatText(value=1.4898,step=0.1,description='Energy (keV):', continuous_update=True)
    button = widgets.Button(description='Search')
    out = widgets.Output()

    def button_evenhandler(_):
        out.clear_output()
        with out:
            print('Candidates:')
            print(hs.eds.get_xray_lines_near_energy(energy=text.value, only_lines=['a', 'b']))

    button.on_click(button_evenhandler)
    widget_set = widgets.HBox([text, button])
    display(widget_set)
    display(out)

def check_latent_space(PC:PhaseClassifier, ratio_to_be_shown=0.25):
    # create color codes
    phase_colors = []
    for i in range(PC.n_components):
          r,g,b = cm.get_cmap('nipy_spectral')(i*(PC.n_components-1)**-1)[:3]
          r,g,b = int(r*255),int(g*255), int(b*255)
          color = "#{:02x}{:02x}{:02x}".format(r,g,b)
          phase_colors.append(color)
    domain = [i for i in range(PC.n_components)]
    range_ = phase_colors
    
    latent, dataset, feature_list, labels = PC.latent, PC.dataset, PC.sem.feature_list, PC.labels
    combined = np.concatenate([latent,dataset.reshape(-1,dataset.shape[-1]).round(2), labels.reshape(-1,1)],axis=1)
    sampled_combined = random.choices(combined, k=int(latent.shape[0]//(ratio_to_be_shown**-1)))
    sampled_combined = np.array(sampled_combined)

    source = pd.DataFrame(sampled_combined, columns=['x','y']+feature_list+['Cluster_id'], index=pd.RangeIndex(0, sampled_combined.shape[0], name='pixel'))
    alt.data_transformers.disable_max_rows()
    
    # Brush
    brush = alt.selection(type='interval')
    
    # Points
    points=alt.Chart(source).mark_circle(
            size=4,
            opacity=0.2
        ).encode(
            x='x:Q',
            y='y:Q', # use min extent to stabilize axis title placement
            color=alt.condition(brush, alt.Color('Cluster_id:N', scale=alt.Scale(domain=domain,range=range_)), alt.value('grey'))
        ).properties( 					
            width=400,
            height=400
        ).properties(title=alt.TitleParams(text='Latent space')
        ).add_selection(brush)
    # Base chart for data tables
    ranked_text = alt.Chart(source).mark_bar().transform_filter(
        brush
    )

    # Data Bars
    columns=list()
    for item in feature_list:
      columns.append(ranked_text.encode(y=alt.Y(f'mean({item}):Q', scale=alt.Scale(domain=(0, 1)))
                      ).properties(title=alt.TitleParams(text=item)))
    text = alt.hconcat(*columns) # Combine bars

    # Build chart
    chart = alt.hconcat(
                points,
                text
            ).resolve_legend(
                color="independent"
            ).configure_view(strokeWidth=0)
    
    return chart
    
def show_cluster_distribution(PC:PhaseClassifier):
    cluster_options = [f'cluster_{n}' for n in range(PC.n_components)]
    multi_select_cluster = widgets.SelectMultiple(options=['All']+cluster_options)
    plots_output = widgets.Output()
    
    with plots_output:
        for i in range(PC.n_components):
            PC.plot_single_cluster_distribution(cluster_num=i)
        
    def eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            if change.new == ('All',):
                for i in range(PC.n_components):
                    PC.plot_single_cluster_distribution(cluster_num=i)
            else:
                for cluster in change.new:
                    PC.plot_single_cluster_distribution(cluster_num=int(cluster.split('_')[1]))
                
    multi_select_cluster.observe(eventhandler, names='value')
    display(multi_select_cluster)
    display(plots_output)
    

def show_unmixed_weights(weights:pd.DataFrame):
    weights_options = weights.index
    multi_select_cluster = widgets.SelectMultiple(options=weights_options)
    plots_output = widgets.Output()
    all_output = widgets.Output()
    
    with all_output:
        display(weights)
        
    def multi_select_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            row_index = [cluster for cluster in change.new]
            display(weights.loc[row_index])
            for cluster in change.new:
                num_cpnt = len(weights.columns.to_list())
                fig, axs = plt.subplots(1,1,figsize=(4,3),dpi=96)
                axs.bar(np.arange(0,num_cpnt), weights[weights.index == cluster].to_numpy().ravel(), width=0.6)
                axs.set_xticks(np.arange(0,num_cpnt))
                axs.set_ylabel('weight of component')
                axs.set_xlabel('component number')
                plt.show()
    
    multi_select_cluster.observe(multi_select_cluster_eventhandler, names='value')
    
    display(multi_select_cluster)
    tab = widgets.Tab([all_output, plots_output])
    tab.set_title(0, 'All weights')
    tab.set_title(1, 'Single weight')
    display(tab)
    
def show_unmixed_components(PC:PhaseClassifier, components:pd.DataFrame):
    components_options = components.columns
    dropdown_cluster = widgets.Dropdown(options=components_options)
    plots_output = widgets.Output()
    all_output = widgets.Output()
    
    with all_output:
        PC.plot_unmixed_profile(components)
    def dropdown_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            plot_profile(PC.energy_axis, components[change.new], PC.peak_list)
    
    dropdown_cluster.observe(dropdown_cluster_eventhandler, names='value')
    
    display(dropdown_cluster)
    tab = widgets.Tab([all_output, plots_output])
    tab.set_title(0, 'All cpnt')
    tab.set_title(1, 'Single cpnt')
    display(tab)

def show_unmixed_weights_and_compoments(PC:PhaseClassifier, weights:pd.DataFrame, components:pd.DataFrame):
    # weights
    weights_options = weights.index
    multi_select_cluster = widgets.SelectMultiple(options=weights_options)
    plots_output = widgets.Output()
    all_output = widgets.Output()
    
    with all_output:
        display(weights)
        
    def multi_select_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            row_index = [cluster for cluster in change.new]
            display(weights.loc[row_index])
            for cluster in change.new:
                num_cpnt = len(weights.columns.to_list())
                fig, axs = plt.subplots(1,1,figsize=(4,3),dpi=96)
                axs.bar(np.arange(0,num_cpnt), weights[weights.index == cluster].to_numpy().ravel(), width=0.6)
                axs.set_xticks(np.arange(0,num_cpnt))
                axs.set_ylabel('weight of component')
                axs.set_xlabel('component number')
                plt.show()
    
    multi_select_cluster.observe(multi_select_cluster_eventhandler, names='value')


    # compoments
    components_options = components.columns
    dropdown_cluster = widgets.Dropdown(options=components_options)
    plots_output_cpnt = widgets.Output()
    all_output_cpnt = widgets.Output()
    
    with all_output_cpnt:
        PC.plot_unmixed_profile(components)
    def dropdown_cluster_eventhandler(change):
        plots_output_cpnt.clear_output()
        with plots_output_cpnt:
            plot_profile(PC.energy_axis, components[change.new], PC.peak_list)
    
    dropdown_cluster.observe(dropdown_cluster_eventhandler, names='value')
    
    widget_set = widgets.HBox([multi_select_cluster, dropdown_cluster])
    display(widget_set)


    tab = widgets.Tab([all_output, plots_output, all_output_cpnt, plots_output_cpnt])
    tab.set_title(0, 'All weights')
    tab.set_title(1, 'Single weight')    
    tab.set_title(2, 'All components')
    tab.set_title(3, 'Single component')
    display(tab)

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
    

def save_csv(df):
    text = widgets.Text(value='stat_info.csv',
                    placeholder='Type something',
                    description='Save as:',
                    disabled=False,
                    continuous_update=True
                   )

    button = widgets.Button(description='Save')
    out = widgets.Output()
    def save_to(_):
        out.clear_output()
        with out:
            df.to_csv(text.value)
            print('save the csv to', text.value)

    button.on_click(save_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)
    
def show_cluster_stats(PC:PhaseClassifier,binary_filter_args):
    columns = ['area (um^2)','equivalent_diameter (um)', 
           'major_axis_length (um)','minor_axis_length (um)']

    for item in ('min_intensity','mean_intensity','max_intensity'):
            columns += [f'{item}_{peak}' for peak in PC.peak_list]

    properties = widgets.Dropdown(options=columns,description='property:')
    clusters = widgets.SelectMultiple(options=[f'cluster_{i}' for i in range(PC.n_components)],description='cluster:')
    bound_bins = widgets.BoundedIntText(value=40,min=5,max=100,step=1,description='num_bins:')
    output=widgets.Output()

    def plot_output(clusters, properties, bound_bins):
        output.clear_output()
        with output:
            df_list = []
            for cluster in clusters:
                df_stats = PC.phase_statics(cluster_num=int(cluster.split('_')[1]),
                                            element_peaks=PC.peak_list,
                                            binary_filter_args=binary_filter_args)
                df_list.append(df_stats[properties])
                fig, axs = plt.subplots(1,1,figsize=(4,3),dpi=96)
                sns.set_style('ticks')
                sns.histplot(df_stats[properties],bins=bound_bins)
                plt.title(cluster)
                plt.show()
            df_list = pd.concat(df_list, axis=1, keys=clusters)
            save_csv(df_list)
            
            

    def cluster_handler(change):
        plot_output(change.new, properties.value, bound_bins.value)

    def properties_handler(change):
        plot_output(clusters.value, change.new, bound_bins.value)

    def bins_handler(change):
        plot_output(clusters.value, properties.value, change.new)

    clusters.observe(cluster_handler,names='value')
    properties.observe(properties_handler,names='value')
    bound_bins.observe(bins_handler,names='value')

    all_widgets = widgets.HBox([clusters,properties,bound_bins])
    display(all_widgets)
    display(output) 
