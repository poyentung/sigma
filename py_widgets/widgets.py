#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import visualisation as visual 
from src.segmentation import PixelSegmenter

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

def view_bcf_dataset(sem, search_energy=True):
    if search_energy == True:
        search_energy_peak()
    bse_out = widgets.Output()
    with bse_out:
        sem.bse.plot(colorbar=False)
        plt.show()

    sum_spec_out = widgets.Output()
    with sum_spec_out:
        visual.plot_sum_spectrum(sem.edx)

    elemental_map_out = widgets.Output()
    with elemental_map_out:
        visual.plot_intensity_maps(sem.edx, sem.feature_list)

    if sem.edx_bin is not None:
        elemental_map_out_bin = widgets.Output()
        with elemental_map_out_bin:
            visual.plot_intensity_maps(sem.edx_bin, sem.feature_list)

        
    layout = widgets.Layout(width='600px', height='40px')
    text = widgets.Text(value='Al_Ka, C_Ka, Ca_Ka, Fe_Ka, K_Ka, O_Ka, Si_Ka, Ti_Ka, Zn_La',
                        placeholder='Type something',
                        description='Feature list:',
                        disabled=False,
                        continuous_update=True,
                        display='flex',
                        flex_flow='column',
                        align_items='stretch', 
                        layout=layout
                    )

    button = widgets.Button(description='Set')
    out = widgets.Output()

    def set_to(_):
        out.clear_output()
        with out:
            feature_list = text.value.replace(" ", "").split(',')
            sem.set_feature_list(feature_list)

        sum_spec_out.clear_output()
        with sum_spec_out:
            visual.plot_sum_spectrum(sem.edx)
        
        elemental_map_out.clear_output()
        with elemental_map_out:
            visual.plot_intensity_maps(sem.edx, sem.feature_list)

        if sem.edx_bin is not None:
            elemental_map_out_bin.clear_output()
            with elemental_map_out_bin:
                visual.plot_intensity_maps(sem.edx_bin, sem.feature_list)

    button.on_click(set_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)

    tab_list = [bse_out, sum_spec_out, elemental_map_out]
    if sem.edx_bin is not None:
        tab_list += [elemental_map_out_bin]

    tab = widgets.Tab(tab_list)
    tab.set_title(0, 'BSE image')
    tab.set_title(1, 'EDX sum spectrum')
    tab.set_title(2, 'Elemental maps (raw)')
    if sem.edx_bin is not None:
        tab.set_title(3, 'Elemental maps (binned)')
    display(tab)

def check_latent_space(ps:PixelSegmenter, ratio_to_be_shown=0.25, show_map=False):
    # create color codes 
    phase_colors = []
    for i in range(ps.n_components):
        r,g,b = cm.get_cmap(ps.color_palette)(i*(ps.n_components-1)**-1)[:3]
        r,g,b = int(r*255),int(g*255), int(b*255)
        color = "#{:02x}{:02x}{:02x}".format(r,g,b)
        phase_colors.append(color)
    domain = [i for i in range(ps.n_components)]
    range_ = phase_colors
    
    latent, dataset, feature_list, labels = ps.latent, ps.dataset, ps.sem.feature_list, ps.labels
    x_id, y_id = np.meshgrid(range(ps.width), range(ps.height))
    x_id = x_id.ravel().reshape(-1,1)
    y_id = y_id.ravel().reshape(-1,1)
    z_id = (ps.bse.data/ps.bse.data.max()).reshape(-1,1)

    combined = np.concatenate([x_id, y_id, z_id, latent, dataset.reshape(-1,dataset.shape[-1]).round(2), labels.reshape(-1,1)],axis=1)

    sampled_combined = random.choices(combined, k=int(latent.shape[0]//(ratio_to_be_shown**-1)))
    sampled_combined = np.array(sampled_combined)

    source = pd.DataFrame(sampled_combined, columns=['x_id','y_id','z_id','x','y']+feature_list+['Cluster_id'], index=pd.RangeIndex(0, sampled_combined.shape[0], name='pixel'))
    alt.data_transformers.disable_max_rows()
    
    # Brush
    brush = alt.selection(type='interval')
    
    # Points
    points=alt.Chart(source).mark_circle(
            size=2,
        ).encode(
            x='x:Q',
            y='y:Q', # use min extent to stabilize axis title placement
            color=alt.Color('Cluster_id:N', scale=alt.Scale(domain=domain,range=range_)),
            opacity=alt.condition(brush, alt.value(0.7), alt.value(0.25)),
            tooltip=['Cluster_id:N', alt.Tooltip('x:Q', format=',.2f'), alt.Tooltip('y:Q', format=',.2f')]
        ).properties( 					
            width=450,
            height=450
        ).properties(title=alt.TitleParams(text='Latent space')
        ).add_selection(brush)

    # Base chart for data tables
    ranked_text = alt.Chart(source).mark_bar().transform_filter(
        brush
    )

    # Data Bars
    columns=list()
    domain_barchart=(0,1) if ps.dataset.max()<1.0 else (-4,4)
    for item in feature_list:
        columns.append(ranked_text.encode(y=alt.Y(f'mean({item}):Q', scale=alt.Scale(domain=domain_barchart))
                      ).properties(title=alt.TitleParams(text=item)))
    text = alt.hconcat(*columns) # Combine bars

    # Heatmap
    if show_map == True:
        bse_df = pd.DataFrame({'x_bse': x_id.ravel(),'y_bse':y_id.ravel(), 'z_bse':z_id.ravel()})
        bse = alt.Chart(bse_df).mark_circle(size=3).encode(
                    x=alt.X('x_bse:O',axis=None),
                    y=alt.Y('y_bse:O',axis=None),
                    color= alt.Color('z_bse:Q', scale=alt.Scale(scheme='greys', domain=[1.0,0.0]))
                    ).properties(
                        width=ps.width,
                        height=ps.height
                    )
        heatmap = alt.Chart(source).mark_circle(size=3).encode(
                    x=alt.X('x_id:O',axis=None),
                    y=alt.Y('y_id:O',axis=None),
                    color= alt.Color('Cluster_id:N', scale=alt.Scale(domain=domain,range=range_)),
                    opacity=alt.condition(brush, alt.value(1), alt.value(0))
                    ).properties(
                        width=ps.width,
                        height=ps.height
                    ).add_selection(brush)
        heatmap_bse = bse + heatmap
    
    final_widgets = [points,heatmap_bse,text] if show_map == True else [points,text]

    # Build chart
    chart = alt.hconcat(
                *final_widgets
            ).resolve_legend(
                color="independent"
            ).configure_view(strokeWidth=0)
    
    return chart
    
def show_cluster_distribution(ps:PixelSegmenter):
    cluster_options = [f'cluster_{n}' for n in range(ps.n_components)]
    multi_select_cluster = widgets.SelectMultiple(options=['All']+cluster_options)
    plots_output = widgets.Output()
    
    with plots_output:
        for i in range(ps.n_components):
            ps.plot_single_cluster_distribution(cluster_num=i)
        
    def eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            if change.new == ('All',):
                for i in range(ps.n_components):
                    ps.plot_single_cluster_distribution(cluster_num=i)
            else:
                for cluster in change.new:
                    ps.plot_single_cluster_distribution(cluster_num=int(cluster.split('_')[1]))
                
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
    
def show_unmixed_components(ps:PixelSegmenter, components:pd.DataFrame):
    components_options = components.columns
    dropdown_cluster = widgets.Dropdown(options=components_options)
    plots_output = widgets.Output()
    all_output = widgets.Output()
    
    with all_output:
        ps.plot_unmixed_profile(components)
    def dropdown_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            visual.plot_profile(ps.energy_axis, components[change.new], ps.peak_list)
    
    dropdown_cluster.observe(dropdown_cluster_eventhandler, names='value')
    
    display(dropdown_cluster)
    tab = widgets.Tab([all_output, plots_output])
    tab.set_title(0, 'All cpnt')
    tab.set_title(1, 'Single cpnt')
    display(tab)

def show_unmixed_weights_and_compoments(ps:PixelSegmenter, weights:pd.DataFrame, components:pd.DataFrame):
    # weights
    weights.loc['Sum'] = weights.sum()
    weights = weights.round(3)
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
        ps.plot_unmixed_profile(components)
    def dropdown_cluster_eventhandler(change):
        plots_output_cpnt.clear_output()
        with plots_output_cpnt:
            visual.plot_profile(ps.energy_axis, components[change.new], ps.peak_list)
    
    dropdown_cluster.observe(dropdown_cluster_eventhandler, names='value')
    
    widget_set = widgets.HBox([multi_select_cluster, dropdown_cluster])
    display(widget_set)


    tab = widgets.Tab([all_output, plots_output, all_output_cpnt, plots_output_cpnt])
    tab.set_title(0, 'All weights')
    tab.set_title(1, 'Single weight')    
    tab.set_title(2, 'All components')
    tab.set_title(3, 'Single component')
    display(tab)

# def show_clusters(ps:PixelSegmenter,binary_filter_args):
#     cluster_options = [f'cluster_{n}' for n in range(ps.n_components)]
#     dropdown_cluster = widgets.Dropdown(options=['ALL']+cluster_options)
#     plots_output = widgets.Output()
    
#     with plots_output:
#         for i in range(ps.n_components):
#             ps.plot_binary_map(cluster_num=i, binary_filter_args=binary_filter_args)
        
#     def dropdown_cluster_eventhandler(change):
#         plots_output.clear_output()
#         with plots_output:
#             if (change.new == ALL):
#                 pass
#             else:
#                 ps.plot_binary_map(cluster_num=int(change.new.split('_')[1]), binary_filter_args=binary_filter_args)
        
#     dropdown_cluster.observe(dropdown_cluster_eventhandler, names='value')
#     display(dropdown_cluster)
#     display(plots_output)
    
def show_clusters(ps:PixelSegmenter, normalisation=True, spectra_range=(0,8)):
    cluster_options = [f'cluster_{n}' for n in range(ps.n_components)]
    multi_select = widgets.SelectMultiple(options=cluster_options)
    plots_output = widgets.Output()
    profile_output = widgets.Output()
        
    def eventhandler(change):
        plots_output.clear_output()
        profile_output.clear_output()
        
        with plots_output:
            for cluster in change.new:
                ps.plot_binary_map_edx_profile(cluster_num=int(cluster.split('_')[1]),normalisation=normalisation, spectra_range=spectra_range)
                
        with profile_output:
            ### X-ray profile ###
            for cluster in change.new:
                _,_, edx_profile = ps.get_binary_map_edx_profile(cluster_num=int(cluster.split('_')[1]),use_label=True)
                visual.plot_profile(edx_profile['energy'], edx_profile['intensity'], ps.peak_list)
        
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
    
def show_cluster_stats(ps:PixelSegmenter,binary_filter_args={}):
    columns = ['area (um^2)','equivalent_diameter (um)', 
           'major_axis_length (um)','minor_axis_length (um)']

    for item in ('min_intensity','mean_intensity','max_intensity'):
            columns += [f'{item}_{peak}' for peak in ps.peak_list]

    properties = widgets.Dropdown(options=columns,description='property:')
    clusters = widgets.SelectMultiple(options=[f'cluster_{i}' for i in range(ps.n_components)],description='cluster:')
    bound_bins = widgets.BoundedIntText(value=40,min=5,max=100,step=1,description='num_bins:')
    output=widgets.Output()

    def plot_output(clusters, properties, bound_bins):
        output.clear_output()
        with output:
            df_list = []
            for cluster in clusters:
                df_stats = ps.phase_statics(cluster_num=int(cluster.split('_')[1]),
                                            element_peaks=ps.peak_list,
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
