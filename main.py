# -*- coding: utf-8 -*-

import timeit

from src.run_experiment import Experiment
from models.autoencoder import AutoEncoder
from load_dataset.exhaust import *

if __name__ == '__main__':
    
    file_path='/Users/andrewtung/Documents/Github/bcf_files/XLI_exhaust_011.bcf'
    # /Users/andrewtung/Documents/Github/bcf_files/XLI_exhaust_011.bcf
    
    sem = SEMDataset(file_path)
        
    sem.rebin_signal(size=(2,2))
    remove_fist_peak(sem.edx_bin)
    peak_intensity_normalisation(sem.edx_bin)
    peak_denoising_PCA(sem.edx_bin, plot_results=False)

    dataset = sem.get_feature_maps()
    dataset_processed = avgerage_neighboring_signal(dataset)
    dataset_norm = z_score_normalisation(dataset_processed)

    
    general_results_dir='results'

    Ex = Experiment(descriptor='AE_unmix',
                     general_results_dir=general_results_dir,
                     model = AutoEncoder, 
                     model_args={'hidden_layer_sizes':(512,256,128)},
                     chosen_dataset = dataset_norm,
                     num_epochs=10, patience = 5,
                     batch_size = 64)
    

