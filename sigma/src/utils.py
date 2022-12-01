# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FeatureDataset(Dataset):
    def __init__(self, dataset: np, setname: str, noise=None):

        if len(dataset.shape) != 2:
            dataset = dataset.reshape(-1, dataset.shape[-1])

        # split train/test set if in training mode
        assert setname in ["train", "test", "all"]
        train_data, test_data = train_test_split(
            dataset, test_size=0.2, random_state=42
        )
        if setname == "train":
            self.dataset = train_data

        elif setname == "test":
            self.dataset = test_data

        elif setname == "all":
            self.dataset = dataset

        self.std = self.dataset.std(axis=0)
        self.noise = noise

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        if self.noise is not None:
            out = self.dataset[idx] + np.random.normal(0, (self.noise * self.std) ** 2)
            return torch.Tensor(out)
        else:
            return torch.Tensor(self.dataset[idx])


# Ref: https://www.globalsino.com/EM/page4624.html
k_factors_120kV = dict(Mg_Ka=1.02,
                       Al_Ka=0.86,
                       Si_Ka=0.76,
                       P_Ka=0.77,
                       S_Ka=0.83,
                       K_Ka=0.86,
                       Ca_Ka=0.88,
                       Ti_Ka=0.86,
                       Cr_Ka=0.9,
                       Mn_Ka=1.04,
                       Fe_Ka=1.0,
                       Co_Ka=0.98,
                       Ni_Ka=1.07,
                       Cu_Ka=1.17,
                       Zn_Ka=1.19,
                       Nb_Ka=2.14,
                       Mo_Ka=3.8,
                       Ag_Ka=9.52)