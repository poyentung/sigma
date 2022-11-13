from sigma.utils.load import SEMDataset, IMAGEDataset
from sigma.utils.loadtem import TEMDataset

import hyperspy.api as hs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
from typing import Union, List, Tuple
from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy.signals import EDSTEMSpectrum
from matplotlib import pyplot as plt

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

peak_dict = dict()
for element in hs.material.elements:
    if element[0] == "Li":
        continue
    for character in element[1].Atomic_properties.Xray_lines:
        peak_name = element[0]
        char_name = character[0]
        key = f"{peak_name}_{char_name}"
        peak_dict[key] = character[1].energy_keV


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {"red": [], "green": [], "blue": []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap("CustomMap", cdict)


def plot_sum_spectrum(edx, xray_lines=True):
    size = edx.axes_manager[2].size
    scale = edx.axes_manager[2].scale
    offset = edx.axes_manager[2].offset
    energy_axis = [((a * scale) + offset) for a in range(0, size)]

    fig = go.Figure(
        data=go.Scatter(x=energy_axis, y=edx.sum().data),
        layout_xaxis_range=[offset, 10],
        layout=go.Layout(
            title="EDX Sum Spectrum",
            title_x=0.5,
            xaxis_title="Energy / keV",
            yaxis_title="Counts",
            width=700,
            height=400,
        ),
    )

    if xray_lines:
        feature_list = edx.metadata.Sample.xray_lines
        if np.array(energy_axis).min() <= 0:
            zero_energy_idx = np.where(np.array(energy_axis).round(2) == 0)[0][0]
        else:
            zero_energy_idx = 0
        for el in feature_list:
            peak = edx.sum().data[zero_energy_idx:][int(peak_dict[el] * 100) + 1]
            fig.add_shape(
                type="line",
                x0=peak_dict[el],
                y0=0,
                x1=peak_dict[el],
                y1=int(0.9 * peak),
                line=dict(color="black", width=2, dash="dot"),
            )

            fig.add_annotation(
                x=peak_dict[el],
                y=peak,
                text=el,
                showarrow=False,
                arrowhead=2,
                yshift=30,
                textangle=270,
            )

    fig.update_layout(showlegend=False)
    fig.update_layout(template="simple_white")
    fig.show()


def plot_intensity_maps(edx, element_list, colors=[], save=None):
    feature_dict = {el: i for (i, el) in enumerate(element_list)}
    num_peak = len(element_list)
    if num_peak > 4:
        n_rows = (num_peak + 3) // 4
        n_cols = 4
    else:
        n_rows = 1
        n_cols = num_peak

    c = mcolors.ColorConverter().to_rgb
    # color = sns.color_palette("Spectral", as_cmap=True)
    hsv = plt.get_cmap("hsv")

    # Create cmap for individual maps
    cmaps = []
    if type(colors) == str:
        pass
    elif not colors:
        for i in range(num_peak):
            cmaps.append(
                make_colormap([c("k"), hsv(i / num_peak)[:3], 1, hsv(i / num_peak)[:3]])
            )
    else:
        assert len(colors) == num_peak
        for color in colors:
            cmaps.append(make_colormap([c("k"), color, 1, color]))

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        sharey=True,
        figsize=(4 * n_cols, 3.0 * n_rows),
    )
    for i in range(n_rows):
        for j in range(n_cols):
            cur_peak = (i * n_cols) + j
            if cur_peak > num_peak - 1:  # delete the extra subfigures
                fig.delaxes(axs[i, j])
            else:
                if num_peak > 4:
                    axs_sub = axs[i, j]
                elif num_peak == 1:
                    axs_sub = axs
                else:
                    axs_sub = axs[j]

                el = element_list[cur_peak]
                if (type(edx) is EDSSEMSpectrum) or (type(edx) is EDSTEMSpectrum):
                    el_map = edx.get_lines_intensity([el])[0].data
                else:
                    el_map = edx[:, :, feature_dict[el]]

                if type(colors) == str:
                    c = colors
                else:
                    c = cmaps[(i * n_cols) + j]
                im = sns.heatmap(el_map, cmap=c, square=True, ax=axs_sub)
                axs_sub.set_yticks([])
                axs_sub.set_xticks([])
                axs_sub.set_title(el, fontsize=15)
                # fig.colorbar(im, ax=axs_sub, shrink=0.75)

    fig.subplots_adjust(wspace=0.11, hspace=0.0)
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, bbox_inches="tight", pad_inches=0.01)

    plt.show()
    return fig


def plot_rgb(dataset:Union[SEMDataset, TEMDataset, IMAGEDataset], elemental_maps:np.ndarray, elements:List=[]):
    assert len(elements) < 4
    if not elements:
        elements = dataset.feature_list[:3]
    if not isinstance(elemental_maps, np.ndarray):
        elemental_maps = elemental_maps.data
    shape = elemental_maps.shape[:2]
    img = np.zeros((shape[0], shape[1], 3))

    for i, element in enumerate(elements):
        idx = dataset.feature_dict[element]
        img[:, :, i] = elemental_maps[:, :, idx]

    fig, axs = plt.subplots(1, 1, dpi=96)
    axs.imshow(img, alpha=0.95)
    axs.axis("off")
    plt.show()
    return fig


def plot_pixel_distributions(
    dataset:Union[SEMDataset, TEMDataset, IMAGEDataset], norm_list:List=[], peak:str="Fe_Ka", cmap:str="viridis"
):
    idx = dataset.feature_dict[peak]
    sns.set_style("ticks")
    if type(dataset)!=IMAGEDataset:
        normalised_elemental_data = dataset.get_feature_maps()  
    else:
        normalised_elemental_data = dataset.chemical_maps
    num_norm_process = len(norm_list) + 1
    norm_dataset_labels = ["raw_data"]
    norm_datasets_list = [normalised_elemental_data]

    for i, norm_process in enumerate(norm_list):
        normalised_elemental_data = norm_process(normalised_elemental_data)
        norm_datasets_list.append(normalised_elemental_data)
        norm_dataset_labels.append(norm_process.__name__)

    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    fig, axs = plt.subplots(
        2,
        num_norm_process,
        figsize=(4 * num_norm_process, 6),
        dpi=100,
        gridspec_kw={"height_ratios": [2, 1.5]},
    )

    for i in range(num_norm_process):
        dataset = norm_datasets_list[i]
        im = axs[0, i].imshow(dataset[:, :, idx].round(2), cmap=cmap)
        axs[0, i].set_aspect("equal")
        axs[0, i].set_title(f"{norm_dataset_labels[i]}")

        axs[0, i].axis("off")
        cbar = fig.colorbar(im, ax=axs[0, i], shrink=0.83, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10, size=0)

    for j in range(num_norm_process):
        dataset = norm_datasets_list[j]
        sns.histplot(dataset[:, :, idx].ravel(), ax=axs[1, j], bins=50)

        axs[1, j].set_xlabel("Element Intensity")
        axs[1, j].yaxis.set_major_formatter(formatter)
        if j != 0:
            axs[1, j].set_ylabel(" ")

    fig.subplots_adjust(wspace=0.2, hspace=0.1)
    return fig


def plot_profile(energy, intensity, peak_list):
    if type(intensity) is pd.core.series.Series:
        intensity = intensity.to_list()
    fig = go.Figure(
        data=go.Scatter(x=energy, y=intensity),
        layout_xaxis_range=[0, 8],
        layout=go.Layout(
            title="",
            title_x=0.5,
            xaxis_title="Energy / keV",
            yaxis_title="Intensity",
            width=900,
            height=500,
        ),
    )
    zero_energy_idx = np.where(np.array(energy).round(2) == 0)[0][0]
    for el in peak_list:
        peak = intensity[zero_energy_idx:][int(peak_dict[el] * 100) + 1]
        fig.add_shape(
            type="line",
            x0=peak_dict[el],
            y0=0,
            x1=peak_dict[el],
            y1=0.9 * peak,
            line=dict(color="black", width=2, dash="dot"),
        )

        fig.add_annotation(
            x=peak_dict[el],
            y=peak,
            text=el,
            showarrow=False,
            arrowhead=2,
            yshift=30,
            textangle=270,
        )
    fig.update_layout(showlegend=False)
    fig.update_layout(template="simple_white")
    fig.show()
