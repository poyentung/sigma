from sigma.utils import visualisation as visual
from sigma.src.segmentation import PixelSegmenter
from sigma.utils.load import SEMDataset, IMAGEDataset
from sigma.utils.loadtem import TEMDataset
from sigma.src.utils import k_factors_120kV

import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import hyperspy.api as hs
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors
from skimage.transform import resize
import seaborn as sns
import altair as alt
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display

# to make sure the plot function works
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)


def search_energy_peak():
    text = widgets.BoundedFloatText(
        value=1.4898, step=0.1, description="Energy (keV):", continuous_update=True
    )
    button = widgets.Button(description="Search")
    out = widgets.Output()

    def button_evenhandler(_):
        out.clear_output()
        with out:
            print("Candidates:")
            print(
                hs.eds.get_xray_lines_near_energy(
                    energy=text.value, only_lines=["a", "b"]
                )
            )

    button.on_click(button_evenhandler)
    widget_set = widgets.HBox([text, button])
    display(widget_set)
    display(out)


def save_fig(fig):
    file_name = widgets.Text(
        value="figure_name.tif",
        placeholder="Type something",
        description="Save as:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    folder_name = widgets.Text(
        value="results",
        placeholder="Type something",
        description="Folder name:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    dpi = widgets.BoundedIntText(
        value="96",
        min=0,
        max=300,
        step=1,
        description="Set dpi:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    pad = widgets.BoundedFloatText(
        value="0.01",
        min=0.0,
        description="Set pad:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    button = widgets.Button(description="Save")
    out = widgets.Output()

    def save_to(_):
        out.clear_output()
        with out:
            if not os.path.isdir(folder_name.value):
                os.mkdir(folder_name.value)
            if isinstance(fig, mpl.figure.Figure):
                save_path = os.path.join(folder_name.value, file_name.value)
                fig.savefig(
                    save_path, dpi=dpi.value, bbox_inches="tight", pad_inches=pad.value
                )
                print("save figure to", file_name.value)
            else:
                initial_file_name = file_name.value.split(".")
                folder_for_fig = os.path.join(folder_name.value, initial_file_name[0])
                if not os.path.isdir(folder_for_fig):
                    os.mkdir(folder_for_fig)
                for i, single_fig in enumerate(fig):
                    save_path = os.path.join(
                        folder_for_fig,
                        f"{initial_file_name[0]}_{i:02}.{initial_file_name[1]}",
                    )
                    single_fig.savefig(
                        save_path,
                        dpi=dpi.value,
                        bbox_inches="tight",
                        pad_inches=pad.value,
                    )
                print("save all figure to folder:", folder_for_fig)

    button.on_click(save_to)
    all_widgets = widgets.HBox(
        [folder_name, file_name, dpi, pad, button], layout=Layout(width="auto")
    )
    display(all_widgets)
    display(out)


def pick_color(plot_func, *args, **kwargs):
    # Create initial color codes
    hsv = plt.get_cmap("hsv")
    colors = []
    for i in range(len(kwargs["element_list"])):
        colors.append(mpl.colors.to_hex(hsv(i / len(kwargs["element_list"]))[:3]))

    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = []
    for element, color in zip(kwargs["element_list"], colors):
        color_pickers.append(
            widgets.ColorPicker(value=color, description=element, layout=layout_format)
        )
    # Create an ouput object
    out = widgets.Output()
    with out:
        fig = visual.plot_intensity_maps(**kwargs)
        save_fig(fig)

    def change_color(_):
        out.clear_output()
        with out:
            color_for_map = []
            for color_picker in color_pickers:
                if not isinstance(color_picker, widgets.Button):
                    color_for_map.append(mpl.colors.to_rgb(color_picker.value)[:3])
            fig = visual.plot_intensity_maps(**kwargs, colors=color_for_map)
            save_fig(fig)

    button = widgets.Button(description="Set", layout=Layout(width="auto"))
    button.on_click(change_color)

    # Set cmap = viridis for all maps
    text_color = widgets.Text(
        value="viridis",
        placeholder="Type something",
        description="Color map:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )

    def set_single_cmap(_):
        out.clear_output()
        with out:
            fig = visual.plot_intensity_maps(**kwargs, colors=text_color.value)
            save_fig(fig)

    button2 = widgets.Button(description="Set color map", layout=Layout(width="auto"))
    button2.on_click(set_single_cmap)

    # Reset button
    def reset(_):
        out.clear_output()
        with out:
            fig = visual.plot_intensity_maps(**kwargs, colors=[])
            save_fig(fig)

    button3 = widgets.Button(description="Reset", layout=Layout(width="auto"))
    button3.on_click(reset)

    color_list = []
    for row in range((len(color_pickers) // 5) + 1):
        color_list.append(widgets.HBox(color_pickers[5 * row : (5 * row + 5)]))

    color_list.append(button)
    colorpicker_col = widgets.VBox(
        color_list, layout=Layout(flex="8 1 0%", width="80%")
    )
    button_col = widgets.VBox(
        [text_color, button2, button3], layout=Layout(flex="2 1 0%", width="20%")
    )
    color_box = widgets.HBox([colorpicker_col, button_col])

    out_box = widgets.Box([out])

    final_box = widgets.VBox([color_box, out_box])
    display(final_box)


def view_bcf_dataset(dataset:SEMDataset, search_energy=True):
    if search_energy == True:
        search_energy_peak()

    bse_out = widgets.Output()
    with bse_out:
        dataset.bse.plot(colorbar=False)
        plt.show()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(dataset.bse.data, cmap="gray")
        axs.axis("off")
        save_fig(fig)
        plt.close()

    sum_spec_out = widgets.Output()
    with sum_spec_out:
        visual.plot_sum_spectrum(dataset.edx)

    elemental_map_out = widgets.Output()
    with elemental_map_out:
        pick_color(
            visual.plot_intensity_maps, edx=dataset.edx, element_list=dataset.feature_list
        )
        # fig = visual.plot_intensity_maps(sem.edx, sem.feature_list)
        # save_fig(fig)

    if dataset.edx_bin is not None:
        elemental_map_out_bin = widgets.Output()
        with elemental_map_out_bin:
            pick_color(
                visual.plot_intensity_maps,
                edx=dataset.edx_bin,
                element_list=dataset.feature_list,
            )
            # fig = visual.plot_intensity_maps(sem.edx_bin, sem.feature_list)
            # save_fig(fig)

    default_elements = ""
    for i, element in enumerate(dataset.feature_list):
        if i == len(dataset.feature_list) - 1:
            default_elements += element
        else:
            default_elements += element + ", "

    layout = widgets.Layout(width="600px", height="40px")
    text = widgets.Text(
        value=default_elements,
        placeholder="Type something",
        description="Feature list:",
        disabled=False,
        continuous_update=True,
        # display='flex',
        # flex_flow='column',
        align_items="stretch",
        layout=layout,
    )

    button = widgets.Button(description="Set")
    out = widgets.Output()

    def set_to(_):
        out.clear_output()
        with out:
            feature_list = text.value.replace(" ", "").split(",")
            dataset.set_feature_list(feature_list)

        sum_spec_out.clear_output()
        with sum_spec_out:
            visual.plot_sum_spectrum(dataset.edx)

        elemental_map_out.clear_output()
        with elemental_map_out:
            visual.plot_intensity_maps(dataset.edx, dataset.feature_list)

        if dataset.edx_bin is not None:
            elemental_map_out_bin.clear_output()
            with elemental_map_out_bin:
                visual.plot_intensity_maps(dataset.edx_bin, dataset.feature_list)

    button.on_click(set_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)

    tab_list = [bse_out, sum_spec_out, elemental_map_out]
    if dataset.edx_bin is not None:
        tab_list += [elemental_map_out_bin]

    tab = widgets.Tab(tab_list)
    tab.set_title(0, "BSE image")
    tab.set_title(1, "EDX sum spectrum")
    tab.set_title(2, "Elemental maps (raw)")
    i = 2
    if dataset.edx_bin is not None:
        tab.set_title(i + 1, "Elemental maps (binned)")
    display(tab)

def view_im_dataset(im):
    intensity_out = widgets.Output()
    with intensity_out:
        fig, axs = plt.subplots(1, 1)
        axs.imshow(im.intensity_map, cmap="gray")
        axs.axis("off")
        save_fig(fig)
        plt.show()

    elemental_map_out = widgets.Output()
    with elemental_map_out:
        pick_color(
            visual.plot_intensity_maps, 
            edx=im.chemical_maps, 
            element_list=im.feature_list
        )
        # fig = visual.plot_intensity_maps(sem.edx, sem.feature_list)
        # save_fig(fig)

    default_elements = ""
    for i, element in enumerate(im.feature_list):
        if i == len(im.feature_list) - 1:
            default_elements += element
        else:
            default_elements += element + ", "

    layout = widgets.Layout(width="600px", height="40px")
    text = widgets.Text(
        value=default_elements,
        placeholder="Type something",
        description="Feature list:",
        disabled=False,
        continuous_update=True,
        # display='flex',
        # flex_flow='column',
        align_items="stretch",
        layout=layout,
    )

    button = widgets.Button(description="Set")
    out = widgets.Output()

    def set_to(_):
        out.clear_output()
        with out:
            feature_list = text.value.replace(" ", "").split(",")
            im.set_feature_list(feature_list)

        elemental_map_out.clear_output()
        with elemental_map_out:
            visual.plot_intensity_maps(im.chemical_maps, im.feature_list)


    button.on_click(set_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)

    tab_list = [intensity_out, elemental_map_out]

    tab = widgets.Tab(tab_list)
    tab.set_title(0, "Intensity image")
    tab.set_title(1, "Elemental maps (raw)")
    display(tab)


def view_rgb(dataset:Union[SEMDataset,TEMDataset,IMAGEDataset]):
    option_dict = {}
    if isinstance(dataset.normalised_elemental_data, np.ndarray):
        option_dict["normalised"] = dataset.normalised_elemental_data

    if type(dataset) != IMAGEDataset:
        option_dict["binned"] = dataset.edx_bin.data
        option_dict["raw"] = dataset.edx.data
    else:
        option_dict["raw"] = dataset.chemical_maps

    dropdown_option = widgets.Dropdown(
        options=list(option_dict.keys()), description="Data:"
    )
    dropdown_r = widgets.Dropdown(options=dataset.feature_list, description="Red:")
    dropdown_g = widgets.Dropdown(options=dataset.feature_list, description="Green:")
    dropdown_b = widgets.Dropdown(options=dataset.feature_list, description="Blue:")

    plots_output = widgets.Output()

    def dropdown_option_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_rgb(
                dataset,
                elemental_maps=option_dict[change.new],
                elements=[dropdown_r.value, dropdown_g.value, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_r_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_rgb(
                dataset,
                elemental_maps=option_dict[dropdown_option.value],
                elements=[change.new, dropdown_g.value, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_g_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_rgb(
                dataset,
                elemental_maps=option_dict[dropdown_option.value],
                elements=[dropdown_r.value, change.new, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_b_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_rgb(
                dataset,
                elemental_maps=option_dict[dropdown_option.value],
                elements=[dropdown_r.value, dropdown_g.value, change.new],
            )
            save_fig(fig)

    dropdown_option.observe(dropdown_option_eventhandler, names="value")
    dropdown_r.observe(dropdown_r_eventhandler, names="value")
    dropdown_g.observe(dropdown_g_eventhandler, names="value")
    dropdown_b.observe(dropdown_b_eventhandler, names="value")
    color_box = widgets.VBox([dropdown_r, dropdown_g, dropdown_b])
    box = widgets.HBox([dropdown_option, color_box])

    display(box)
    display(plots_output)


def view_pixel_distributions(dataset:Union[SEMDataset, TEMDataset, IMAGEDataset], norm_list:List=[], cmap:str="viridis"):
    peak_options = dataset.feature_list
    dropdown_peaks = widgets.Dropdown(options=peak_options, description="Element:")
    
    plots_output = widgets.Output()
    
    with plots_output:
        fig = visual.plot_pixel_distributions(
            dataset=dataset, norm_list=norm_list, peak=dropdown_peaks.value, cmap=cmap
            )
        plt.show()
        save_fig(fig)
            
    def dropdown_option_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_pixel_distributions(
            dataset=dataset, norm_list=norm_list, peak=dropdown_peaks.value, cmap=cmap
            )
            plt.show()
            save_fig(fig)
    
    dropdown_peaks.observe(dropdown_option_eventhandler, names="value")
    out_box = widgets.VBox([dropdown_peaks, plots_output])    
    display(out_box)


def view_intensity_maps(edx, element_list):
    pick_color(visual.plot_intensity_maps, edx=edx, element_list=element_list)


def view_bic(
    latent: np.ndarray,
    n_components: int = 20,
    model: str = "BayesianGaussianMixture",
    model_args: Dict = {"random_state": 6},
):
    bic_list = PixelSegmenter.bic(latent, n_components, model, model_args)
    fig = go.Figure(
        data=go.Scatter(
            x=np.arange(1, n_components + 1, dtype=int),
            y=bic_list,
            mode="lines+markers",
        ),
        layout=go.Layout(
            title="",
            title_x=0.5,
            xaxis_title="Number of component",
            yaxis_title="BIC",
            width=800,
            height=600,
        ),
    )

    fig.update_layout(showlegend=False)
    fig.update_layout(template="simple_white")
    fig.update_traces(marker_size=12)
    fig.show()
    save_csv(pd.DataFrame(data={"bic": bic_list}))


def view_latent_space(ps: PixelSegmenter, color=True):
    colors = []
    cmap = plt.get_cmap(ps.color_palette)
    for i in range(ps.n_components):
        colors.append(mpl.colors.to_hex(cmap(i * (ps.n_components - 1) ** -1)[:3]))

    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = []
    for i, c in enumerate(colors):
        color_pickers.append(
            widgets.ColorPicker(
                value=c, description=f"cluster_{i}", layout=layout_format
            )
        )

    newcmp = mpl.colors.ListedColormap(colors, name="new_cmap")
    out = widgets.Output()
    with out:
        fig = ps.plot_latent_space(color=color, cmap=None)
        plt.show()
        save_fig(fig)

    def change_color(_):
        out.clear_output()
        with out:
            color_for_map = []
            for color_picker in color_pickers:
                color_for_map.append(mpl.colors.to_rgb(color_picker.value)[:3])
            newcmp = mpl.colors.ListedColormap(color_for_map, name="new_cmap")
            ps.set_color_palette(newcmp)
            fig = ps.plot_latent_space(color=color, cmap=newcmp)
            save_fig(fig)

    button = widgets.Button(
        description="Set", layout=Layout(flex="8 1 0%", width="auto")
    )
    button.on_click(change_color)

    # Reset button
    def reset(_):
        out.clear_output()
        with out:
            fig = ps.plot_latent_space(color=color, cmap=ps._color_palette)
            save_fig(fig)

    button2 = widgets.Button(
        description="Reset", layout=Layout(flex="2 1 0%", width="auto")
    )
    button2.on_click(reset)

    color_list = []
    for row in range((len(color_pickers) // 5) + 1):
        color_list.append(widgets.HBox(color_pickers[5 * row : (5 * row + 5)]))

    button_box = widgets.HBox([button, button2])
    color_box = widgets.VBox(
        [widgets.VBox(color_list), button_box],
        layout=Layout(flex="2 1 0%", width="auto"),
    )
    out_box = widgets.Box([out], layout=Layout(flex="8 1 0%", width="auto"))
    final_box = widgets.VBox([color_box, out_box])
    display(final_box)


def plot_latent_density(ps: PixelSegmenter, bins=50):
    z = np.histogram2d(x=ps.latent[:, 0], y=ps.latent[:, 1], bins=bins)[0]
    sh_0, sh_1 = z.shape
    x, y = (
        np.linspace(ps.latent[:, 0].min(), ps.latent[:, 0].max(), sh_0),
        np.linspace(ps.latent[:, 1].min(), ps.latent[:, 1].max(), sh_1),
    )
    fig = go.Figure(data=[go.Surface(z=z.T, x=x, y=y, colorscale="RdBu_r")])
    fig.update_layout(
        title="Density of pixels in latent space",
        autosize=True,
        width=500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    fig.show()


def check_latent_space(ps: PixelSegmenter, ratio_to_be_shown=0.25, show_map=False):
    # create color codes
    phase_colors = []
    for i in range(ps.n_components):
        r, g, b = cm.get_cmap(ps.color_palette)(i * (ps.n_components - 1) ** -1)[:3]
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        phase_colors.append(color)

   
    domain = [i for i in range(ps.n_components)]
    range_ = phase_colors

    latent, dataset, feature_list, labels = (
        ps.latent,
        ps.dataset.normalised_elemental_data,
        ps.dataset.feature_list,
        ps.labels,
    )
    x_id, y_id = np.meshgrid(range(ps.width), range(ps.height))
    x_id = x_id.ravel().reshape(-1, 1)
    y_id = y_id.ravel().reshape(-1, 1)

    if type(ps.dataset) != IMAGEDataset:
        z_id = (ps.bse.data / ps.bse.data.max()).reshape(-1, 1)
    else:
        intensity_map = resize(ps.dataset.intensity_map, ps.dataset.chemical_maps.shape[:2])
        z_id = (intensity_map / intensity_map.max()).reshape(-1, 1)

    combined = np.concatenate(
        [
            x_id,
            y_id,
            z_id,
            latent,
            dataset.reshape(-1, dataset.shape[-1]).round(2),
            labels.reshape(-1, 1),
        ],
        axis=1,
    )

    if ratio_to_be_shown != 1.0:
        sampled_combined = random.choices(
            combined, k=int(latent.shape[0] // (ratio_to_be_shown ** -1))
        )
        sampled_combined = np.array(sampled_combined)
    else:
        sampled_combined = combined

    source = pd.DataFrame(
        sampled_combined,
        columns=["x_id", "y_id", "z_id", "x", "y"] + feature_list + ["Cluster_id"],
        index=pd.RangeIndex(0, sampled_combined.shape[0], name="pixel"),
    )
    alt.data_transformers.disable_max_rows()

    # Brush
    brush = alt.selection(type="interval")
    interaction = alt.selection(
        type="interval",
        bind="scales",
        on="[mousedown[event.shiftKey], mouseup] > mousemove",
        translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
        zoom="wheel![event.shiftKey]",
    )

    # Points
    points = (
        alt.Chart(source)
        .mark_circle(size=3,)
        .encode(
            x="x:Q",
            y="y:Q",  # use min extent to stabilize axis title placement
            color=alt.Color(
                "Cluster_id:N", scale=alt.Scale(domain=domain, range=range_)
            ),
            opacity=alt.condition(brush, alt.value(0.9), alt.value(0.25)),
            tooltip=[
                "Cluster_id:N",
                alt.Tooltip("x:Q", format=",.2f"),
                alt.Tooltip("y:Q", format=",.2f"),
            ],
        )
        .properties(width=450, height=450)
        .properties(title=alt.TitleParams(text="Latent space"))
        .add_selection(
            brush,
            interaction
            )
    )

    # Base chart for data tables
    ranked_text = alt.Chart(source).mark_bar().transform_filter(brush)

    # Data Bars
    columns = list()
    domain_barchart = (0, 1) if ps.dataset_norm.max() < 1.0 else (-4, 4)
    for item in feature_list:
        columns.append(
            ranked_text.encode(
                y=alt.Y(f"mean({item}):Q", scale=alt.Scale(domain=domain_barchart))
            ).properties(title=alt.TitleParams(text=item))
        )
    text = alt.hconcat(*columns)  # Combine bars

    # Heatmap
    if show_map == True:
        bse_df = pd.DataFrame(
            {"x_bse": x_id.ravel(), "y_bse": y_id.ravel(), "z_bse": z_id.ravel()}
        )
        bse = (
            alt.Chart(bse_df)
            .mark_square(size=6)
            .encode(
                x=alt.X("x_bse:O", axis=None),
                y=alt.Y("y_bse:O", axis=None),
                color=alt.Color(
                    "z_bse:Q", scale=alt.Scale(scheme="greys", domain=[1.0, 0.0])
                ),
            )
            .properties(width=ps.width*2, height=ps.height*2)
        )
        heatmap = (
            alt.Chart(source)
            .mark_square(size=5)
            .encode(
                x=alt.X("x_id:O", axis=None),
                y=alt.Y("y_id:O", axis=None),
                color=alt.Color(
                    "Cluster_id:N", scale=alt.Scale(domain=domain, range=range_)
                ),
                opacity=alt.condition(brush, alt.value(0.75), alt.value(0)),
            )
            .properties(width=ps.width*2, height=ps.height*2)
            .add_selection(brush)
        )
        heatmap_bse = bse + heatmap

    final_widgets = [points, heatmap_bse, text] if show_map == True else [points, text]

    # Build chart
    chart = (
        alt.hconcat(*final_widgets)
        .resolve_legend(color="independent")
        .configure_view(strokeWidth=0)
    )

    return chart


def show_cluster_distribution(ps: PixelSegmenter, **kwargs):
    cluster_options = [f"cluster_{n}" for n in range(ps.n_components)]
    multi_select_cluster = widgets.SelectMultiple(options=["All"] + cluster_options)
    plots_output = widgets.Output()

    all_fig = []
    with plots_output:
        for i in range(ps.n_components):
                fig = ps.plot_single_cluster_distribution(cluster_num=i, **kwargs)
                all_fig.append(fig)

    def eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            if change.new == ("All",):
                for i in range(ps.n_components):
                        fig = ps.plot_single_cluster_distribution(cluster_num=i, **kwargs)
            else:
                for cluster in change.new:
                        fig = ps.plot_single_cluster_distribution(
                            cluster_num=int(cluster.split("_")[1]), **kwargs
                        )

    multi_select_cluster.observe(eventhandler, names="value")
    display(multi_select_cluster)
    save_fig(all_fig)
    display(plots_output)


def view_phase_map(ps):
    colors = []
    cmap = plt.get_cmap(ps.color_palette)
    for i in range(ps.n_components):
        colors.append(mpl.colors.to_hex(cmap(i * (ps.n_components - 1) ** -1)[:3]))

    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = []
    for i, c in enumerate(colors):
        color_pickers.append(
            widgets.ColorPicker(
                value=c, description=f"cluster_{i}", layout=layout_format
            )
        )

    newcmp = mpl.colors.ListedColormap(colors, name="new_cmap")
    out = widgets.Output()
    with out:
        fig = ps.plot_phase_map(cmap=None)
        plt.show()
        save_fig(fig)

    def change_color(_):
        out.clear_output()
        with out:
            color_for_map = []
            for color_picker in color_pickers:
                color_for_map.append(mpl.colors.to_rgb(color_picker.value)[:3])
            newcmp = mpl.colors.ListedColormap(color_for_map, name="new_cmap")
            ps.set_color_palette(newcmp)
            fig = ps.plot_phase_map(cmap=newcmp)
            save_fig(fig)

    button = widgets.Button(description="Set", layout=Layout(width="auto"))
    button.on_click(change_color)

    color_list = []
    for row in range((len(color_pickers) // 5) + 1):
        color_list.append(widgets.HBox(color_pickers[5 * row : (5 * row + 5)]))

    color_box = widgets.VBox(
        [widgets.VBox(color_list), button], layout=Layout(flex="2 1 0%", width="auto")
    )
    out_box = widgets.Box([out], layout=Layout(flex="8 1 0%", width="auto"))
    final_box = widgets.VBox([color_box, out_box])
    display(final_box)


def show_unmixed_weights(weights: pd.DataFrame):
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
                fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=96)
                axs.bar(
                    np.arange(0, num_cpnt),
                    weights[weights.index == cluster].to_numpy().ravel(),
                    width=0.6,
                )
                axs.set_xticks(np.arange(0, num_cpnt))
                axs.set_ylabel("weight of component")
                axs.set_xlabel("component number")
                plt.show()

    multi_select_cluster.observe(multi_select_cluster_eventhandler, names="value")

    display(multi_select_cluster)
    tab = widgets.Tab([all_output, plots_output])
    tab.set_title(0, "All weights")
    tab.set_title(1, "Single weight")
    display(tab)


def show_unmixed_components(ps: PixelSegmenter, components: pd.DataFrame):
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

    dropdown_cluster.observe(dropdown_cluster_eventhandler, names="value")

    display(dropdown_cluster)
    tab = widgets.Tab([all_output, plots_output])
    tab.set_title(0, "All cpnt")
    tab.set_title(1, "Single cpnt")
    display(tab)


def show_unmixed_weights_and_compoments(
    ps: PixelSegmenter, weights: pd.DataFrame, components: pd.DataFrame
):
    # weights
    # weights.loc["Sum"] = weights.sum()
    # weights = weights.round(3)
    # weights_options = weights.index
    multi_select_cluster = widgets.SelectMultiple(options=weights.index)
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
                fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=96)
                axs.bar(
                    np.arange(0, num_cpnt),
                    weights[weights.index == cluster].to_numpy().ravel(),
                    width=0.6,
                )
                axs.set_xticks(np.arange(0, num_cpnt))
                axs.set_ylabel("Abundance coefficient")
                axs.set_xlabel("NMF component ID")
                plt.show()
                save_fig(fig)

    multi_select_cluster.observe(multi_select_cluster_eventhandler, names="value")

    # compoments
    components_options = components.columns
    dropdown_cluster = widgets.Dropdown(options=components_options)
    plots_output_cpnt = widgets.Output()
    all_output_cpnt = widgets.Output()

    with all_output_cpnt:
        fig = ps.plot_unmixed_profile(components)
        save_fig(fig)

    def dropdown_cluster_eventhandler(change):
        plots_output_cpnt.clear_output()
        with plots_output_cpnt:
            visual.plot_profile(ps.energy_axis, components[change.new], ps.peak_list)

    dropdown_cluster.observe(dropdown_cluster_eventhandler, names="value")

    widget_set = widgets.HBox([multi_select_cluster, dropdown_cluster])
    display(widget_set)

    tab = widgets.Tab([all_output, plots_output, all_output_cpnt, plots_output_cpnt])
    tab.set_title(0, "All weights")
    tab.set_title(1, "Single weight")
    tab.set_title(2, "All components")
    tab.set_title(3, "Single component")
    display(tab)


def view_clusters_sum_spectra(
    ps: PixelSegmenter, normalisation=True, spectra_range=(0, 8)
):
    cluster_options = [f"cluster_{n}" for n in range(ps.n_components)]
    multi_select = widgets.SelectMultiple(options=cluster_options)
    plots_output = widgets.Output()
    profile_output = widgets.Output()

    figs = []
    with plots_output:
        for cluster in cluster_options:
            fig = ps.plot_binary_map_edx_profile(
                cluster_num=int(cluster.split("_")[1]),
                normalisation=normalisation,
                spectra_range=spectra_range,
            )
            figs.append(fig)

    def eventhandler(change):
        plots_output.clear_output()
        profile_output.clear_output()

        with plots_output:
            for cluster in change.new:
                fig = ps.plot_binary_map_edx_profile(
                    cluster_num=int(cluster.split("_")[1]),
                    normalisation=normalisation,
                    spectra_range=spectra_range,
                )

        with profile_output:
            ### X-ray profile ###
            for cluster in change.new:
                _, _, edx_profile = ps.get_binary_map_edx_profile(
                    cluster_num=int(cluster.split("_")[1]), use_label=True
                )
                visual.plot_profile(
                    edx_profile["energy"], edx_profile["intensity"], ps.peak_list
                )

    multi_select.observe(eventhandler, names="value")

    display(multi_select)
    save_fig(figs)
    tab = widgets.Tab([plots_output, profile_output])
    tab.set_title(0, "clusters + edx")
    tab.set_title(1, "edx")
    display(tab)


def save_csv(df):
    text = widgets.Text(
        value="file_name.csv",
        placeholder="Type something",
        description="Save as:",
        disabled=False,
        continuous_update=True,
    )

    button = widgets.Button(description="Save")
    out = widgets.Output()

    def save_to(_):
        out.clear_output()
        with out:
            df.to_csv(text.value)
            print("save the csv to", text.value)

    button.on_click(save_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)


def show_cluster_stats(ps: PixelSegmenter, binary_filter_args={}):
    columns = [
        "area (um^2)",
        "equivalent_diameter (um)",
        "major_axis_length (um)",
        "minor_axis_length (um)",
    ]

    for item in ("min_intensity", "mean_intensity", "max_intensity"):
        columns += [f"{item}_{peak}" for peak in ps.peak_list]

    properties = widgets.Dropdown(options=columns, description="property:")
    clusters = widgets.SelectMultiple(
        options=[f"cluster_{i}" for i in range(ps.n_components)], description="cluster:"
    )
    bound_bins = widgets.BoundedIntText(
        value=40, min=5, max=100, step=1, description="num_bins:"
    )
    output = widgets.Output()

    def plot_output(clusters, properties, bound_bins):
        output.clear_output()
        df_list = []
        fig_list = []
        with output:
            for cluster in clusters:
                df_stats = ps.phase_statics(
                    cluster_num=int(cluster.split("_")[1]),
                    element_peaks=ps.peak_list,
                    binary_filter_args=binary_filter_args,
                )
                df_list.append(df_stats[properties])
                fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=96)
                sns.set_style("ticks")
                sns.histplot(df_stats[properties], bins=bound_bins)
                plt.title(cluster)
                plt.show()
                fig_list.append(fig)
            df_list = pd.concat(df_list, axis=1, keys=clusters)

        save_csv(df_list)
        fig_list = fig_list[0] if len(fig_list) == 1 else fig_list
        save_fig(fig_list)

    def cluster_handler(change):
        plot_output(change.new, properties.value, bound_bins.value)

    def properties_handler(change):
        plot_output(clusters.value, change.new, bound_bins.value)

    def bins_handler(change):
        plot_output(clusters.value, properties.value, change.new)

    clusters.observe(cluster_handler, names="value")
    properties.observe(properties_handler, names="value")
    bound_bins.observe(bins_handler, names="value")

    all_widgets = widgets.HBox([clusters, properties, bound_bins])
    display(all_widgets)
    display(output)


def view_emi_dataset(tem, search_energy=True):
    if search_energy == True:
        search_energy_peak()

    bse_out = widgets.Output()
    with bse_out:
        tem.bse.plot(colorbar=False)
        plt.show()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(tem.bse.data, cmap="gray")
        axs.axis("off")
        save_fig(fig)
        plt.close()

    sum_spec_out = widgets.Output()
    with sum_spec_out:
        visual.plot_sum_spectrum(tem.edx)

    elemental_map_out = widgets.Output()
    with elemental_map_out:
        if len(tem.feature_list) != 0:
            pick_color(
                visual.plot_intensity_maps, edx=tem.edx, element_list=tem.feature_list
            )
            # fig = visual.plot_intensity_maps(sem.edx, sem.feature_list)
            # save_fig(fig)

    if tem.edx_bin is not None:
        elemental_map_out_bin = widgets.Output()
        with elemental_map_out_bin:
            pick_color(
                visual.plot_intensity_maps,
                edx=tem.edx_bin,
                element_list=tem.feature_list,
            )
            # fig = visual.plot_intensity_maps(sem.edx_bin, sem.feature_list)
            # save_fig(fig)

    default_elements = ""
    for i, element in enumerate(tem.feature_list):
        if i == len(tem.feature_list) - 1:
            default_elements += element
        else:
            default_elements += element + ", "

    layout = widgets.Layout(width="600px", height="40px")
    text = widgets.Text(
        value=default_elements,
        placeholder="Type something",
        description="Feature list:",
        disabled=False,
        continuous_update=True,
        # display='flex',
        # flex_flow='column',
        align_items="stretch",
        layout=layout,
    )

    button = widgets.Button(description="Set")
    out = widgets.Output()

    def set_to(_):
        out.clear_output()
        with out:
            feature_list = text.value.replace(" ", "").split(",")
            tem.set_feature_list(feature_list)

        sum_spec_out.clear_output()
        with sum_spec_out:
            visual.plot_sum_spectrum(tem.edx)

        if len(tem.feature_list) != 0:
            elemental_map_out.clear_output()
            with elemental_map_out:
                pick_color(
                    visual.plot_intensity_maps,
                    edx=tem.edx,
                    element_list=tem.feature_list,
                )
                visual.plot_intensity_maps(tem.edx, tem.feature_list)

        if tem.edx_bin is not None:
            elemental_map_out_bin.clear_output()
            with elemental_map_out_bin:
                visual.plot_intensity_maps(tem.edx_bin, tem.feature_list)

    button.on_click(set_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)

    tab_list = [bse_out, sum_spec_out, elemental_map_out]
    if tem.edx_bin is not None:
        tab_list += [elemental_map_out_bin]

    tab = widgets.Tab(tab_list)
    tab.set_title(0, "EDX sum intensity map")
    tab.set_title(1, "EDX sum spectrum")
    tab.set_title(2, "Elemental maps (raw)")
    i = 2
    if tem.edx_bin is not None:
        tab.set_title(i + 1, "Elemental maps (binned)")
    display(tab)


def show_abundance_map(ps:PixelSegmenter, weights:pd.DataFrame, components: pd.DataFrame):
    def plot_rgb(ps, phases:List):
        shape = ps.get_binary_map_edx_profile(0)[0].shape
        img = np.zeros((shape[0], shape[1], 3))
        
        # make abundance map
        for i, phase in enumerate(phases):
            if phase!='None':
                cpnt_weights = weights[phase]/weights[phase].max()
                tmp = np.zeros(shape)
                for j in range(ps.n_components):
                    try:
                        idx = ps.get_binary_map_edx_profile(j)[1]
                        tmp[idx] = cpnt_weights[j]
                    except ValueError:
                        pass
                        # print(f'warning: no pixel is assigned to cpnt_{j}.')
                img[:, :, i] = tmp
            else:
                img[:, :, i] = np.zeros(shape)

        fig, axs = plt.subplots(1, 1, dpi=96)
        axs.imshow(img, alpha=0.95)
        axs.axis("off")
        plt.show()
        return fig
    
    cpnt_names = [f'cpnt_{i}' for i in range(len(weights.columns))] 
    cpnt_options = [x for x in zip(cpnt_names, weights.columns)] + [('None', 'None')]
    dropdown_r = widgets.Dropdown(options=cpnt_options, value='None', description="Red:")
    dropdown_g = widgets.Dropdown(options=cpnt_options, value='None', description="Green:")
    dropdown_b = widgets.Dropdown(options=cpnt_options, value='None', description="Blue:")

    plots_output = widgets.Output()
    with plots_output:
        fig = plot_rgb(
            ps,
            phases=['None', 'None', 'None'],
        )
        save_fig(fig)

    def dropdown_r_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(
                ps,
                phases=[change.new, dropdown_g.value, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_g_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(
                ps,
                phases=[dropdown_r.value, change.new, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_b_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(
                ps,
                phases=[dropdown_r.value, dropdown_g.value, change.new],
            )
            save_fig(fig)

    dropdown_r.observe(dropdown_r_eventhandler, names="value")
    dropdown_g.observe(dropdown_g_eventhandler, names="value")
    dropdown_b.observe(dropdown_b_eventhandler, names="value")
    color_box = widgets.VBox([dropdown_r, dropdown_g, dropdown_b])

    display(color_box)
    display(plots_output)

    
def plot_ternary_composition(ps:PixelSegmenter):
    # cluster_num:int,
    # elements:List,
    # k_factors:List[float]=None,
    # composition_units:str='atomic',
    cluster_options = range(ps.n_components)
    dropdown_cluster = widgets.Dropdown(options=cluster_options, description='cluster',value=None, )
    
    element_options = []
    for el in ps.dataset.feature_list:
        if el in k_factors_120kV.keys():
            element_options.append(el)
    
    dropdown_element1 = widgets.Dropdown(options=element_options, description='element1',value=None)
    dropdown_element2 = widgets.Dropdown(options=element_options, description='element2',value=None)
    dropdown_element3 = widgets.Dropdown(options=element_options, description='element3',value=None)
    
    plots_output = widgets.Output()
    
    button = widgets.Button(description="Calculate")
    out = widgets.Output()

    def button_evenhandler(_):
        plots_output.clear_output()
        with plots_output:
            try:
                ps.plot_ternary_composition(
                    cluster_num=int(dropdown_cluster.value),
                    elements=[dropdown_element1.value, dropdown_element2.value, dropdown_element3.value],
                )
            except ValueError:
                print('Oops. Please try different combiniations of elements')
            except TypeError:
                print('Please select cluster number')
            except AttributeError:
                print('Please select elements')

    button.on_click(button_evenhandler)
    
    option_box = widgets.HBox([dropdown_cluster, 
                               dropdown_element1, 
                               dropdown_element2,
                               dropdown_element3, 
                               button],
                              layout=Layout(flex="flex-start", width="80%",align_items='center'),)
    display(option_box)
    display(plots_output)
    
            
    