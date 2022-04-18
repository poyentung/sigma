## Description

**Spectral Interpretation using Gaussian Mixtures and Autoencoder (SIGMA)** is an open-source Python library for phase identification and spectrum analysis for energy dispersive x-ray spectroscopy (EDS) datasets. The library mainly builds on the Hyperspy, Pytorch, and Scikit-learn. 


### Try SIGMA with Colab:

<a href="https://colab.research.google.com/github/poyentung/unmix/blob/final/tutorial/full_tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a href="https://zenodo.org/badge/latestdoi/415443021"><img src="https://zenodo.org/badge/415443021.svg" alt="DOI"></a>

## Installation
1. Create a [**Python>=3.7.0**](https://www.python.org/) environment with [**conda**](https://docs.conda.io/en/latest/):
```bash
conda create -n sigma python=3.7 anaconda
conda activate sigma
```

2. Install SIGMA with [**pip**](https://pypi.org/project/pip/):
```bash
pip install emsigma
```


## Check EDS dataset with GUI
An example of checking the EDS dataset and the sum spectrum.
<details open>
<summary>Demo with Colab</summary>

![Demo-check_EDS_dataset](https://user-images.githubusercontent.com/29102746/159283425-00a6e8a6-3274-4495-9ab6-ca0e9a844277.gif)

</details>

## Dimensionality reduction and clustering
An example of analysing the latent space using the graphical widget.
<details open>
<summary>Demo with Colab</summary>

![Screen Recording 2022-02-22 at 12 09 38 PM](https://user-images.githubusercontent.com/29102746/159275323-45ad978a-7dcf-40d9-839b-d58979bb0101.gif)

</details>

## Factor analysis on cluster-wise spectra
A demo of acquiring Background-substracted spectrum using Factor Analysis (FA).
<details open>
<summary>Demo with Colab</summary>
  
![Demo-NMF](https://user-images.githubusercontent.com/29102746/159292227-1e82402c-2429-4c81-8245-8798c426ea0f.gif)

</details>
