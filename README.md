## Description

**Spectral Interpretation using Gaussian Mixtures and Autoencoder (SIGMA)** is an open-source Python library for phase identification and spectrum analysis for energy dispersive x-ray spectroscopy (EDS) datasets. The library mainly builds on the [**Hyperspy**](https://hyperspy.org/), [**Pytorch**](https://pytorch.org/), and [**Scikit-learn**](https://scikit-learn.org/stable/). The current version only supports `.bcf` and `.emi` files. The publication is available [**here**](https://doi.org/10.1002/essoar.10511396.1).<br />

**(UPDATE)** Now SIGMA (version=0.1.31) can load `individual images` (elemental intensity maps, e.g., `*.tif`).

**Try your dataset on SIGMA with Colab in the cloud:** 
| Type of data  | Colab link    
 :---: | :---: 
| SEM | <a href="https://colab.research.google.com/github/poyentung/sigma/blob/master/tutorial/colab/tutorial_colab_sem.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|
| TEM | <a href="https://colab.research.google.com/github/poyentung/sigma/blob/master/tutorial/colab/tutorial_colab_tem.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Images | <a href="https://colab.research.google.com/github/poyentung/sigma/blob/master/tutorial/colab/tutorial_colab_image.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>|

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101005611: [**The EXCITE Network**](https://excite-network.eu/). If analysis using SIGMA forms a part of published work please cite the [manuscript](https://doi.org/10.1029/2022GC010530).

## Installation
1. Create a [**Python>=3.7.0**](https://www.python.org/) environment with [**conda**](https://docs.conda.io/en/latest/):
```bash
conda create -n sigma python=3.7 anaconda
conda activate sigma
```

2. Install **SIGMA** with [**pip**](https://pypi.org/project/pip/):
```bash
pip install emsigma
```

3. Use the notebook in the tutorial folder to run **SIGMA**.

## Workflow of SIGMA
1. A neural network autoencoder is trained to learn good representations of elemental pixels in the 2D latent space. <br />
<div align="left">
  <img width="650" alt="Autoencoder" src="https://user-images.githubusercontent.com/29102746/163899500-34ac68e2-9a38-44d9-a869-e40c024c420b.png">
</div><br />

2. The trained encoder is then used to transform high-dimensional elemental pixels into low-dimensional representations, followed by clustering using Gaussian mixture modeling (GMM) in the informative latent space.<br />
<div align="left">
  <img width="650" alt="GMM" src="https://user-images.githubusercontent.com/29102746/163899758-6bd61544-fa91-44ac-8647-d249982b6607.png"> 
</div><br />

3. Non-negative matrix factorization (NMF) is applied to unmix the single-phase spectra for all clusters.<br />
<div align="left">
  <img width="650" alt="NMF" src="https://user-images.githubusercontent.com/29102746/163899763-0fb4f835-3380-4504-9f3a-bb33089421f8.png">  
</div><br />

In such a way, the algorithm not only identifies the locations of all unknown phases but also isolates the background-subtracted EDS spectra of individual phases.

## User-friendly GUI
### Check .bcf file
An example of checking the EDS dataset and the sum spectrum.
<details open>
<summary>Demo with Colab</summary>

![Demo-check_EDS_dataset](https://user-images.githubusercontent.com/29102746/159283425-00a6e8a6-3274-4495-9ab6-ca0e9a844277.gif)

</details>

### Dimensionality reduction and clustering
An example of analysing the latent space using the graphical widget.
<details open>
<summary>Demo with Colab</summary>

![Screen Recording 2022-02-22 at 12 09 38 PM](https://user-images.githubusercontent.com/29102746/159275323-45ad978a-7dcf-40d9-839b-d58979bb0101.gif)

</details>

### Factor analysis on cluster-wise spectra
A demo of acquiring Background-substracted spectrum using Factor Analysis (FA).
<details open>
<summary>Demo with Colab</summary>
  
![Demo-NMF](https://user-images.githubusercontent.com/29102746/159292227-1e82402c-2429-4c81-8245-8798c426ea0f.gif)

</details>
