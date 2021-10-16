# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class FeatureDataset(Dataset):
    def __init__(self, dataset:np, setname:str, noise=None):
        
        if len(dataset.shape) != 2:
            dataset = dataset.reshape(-1, dataset.shape[-1])
        
        # split train/test set if in training mode
        assert setname in ['train','test', 'all']
        train_data, test_data = train_test_split(dataset, test_size=0.1,
                                                 random_state=42)
        if setname == 'train':
            self.dataset = train_data
            
        elif setname == 'test':
            self.dataset = test_data
        
        elif setname == 'all':
            self.dataset = dataset
            
        self.std = self.dataset.std(axis=0)
        self.noise = noise
        

    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        if self.noise is not None:
            out = self.dataset[idx] + np.random.normal(0, (self.noise*self.std)**2)
            return torch.Tensor(out)
        else:
            return torch.Tensor(self.dataset[idx])