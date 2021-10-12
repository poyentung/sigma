# -*- coding: utf-8 -*-

import timeit

from src import run_experiment
from models.autoencoder import AutoEncoder
from load_dataset.exhaust import *

if __name__ == '__main__':
    
    sem = SEMDataset('XLI_exhaust_011.bcf')
        
    sem.rebin_signal(size=(2,2))
    remove_fist_peak(sem.edx_bin)
    peak_intensity_normalisation(sem.edx_bin)
    peak_denoising_PCA(sem.edx_bin, plot_results=False)

    dataset = sem.get_feature_maps()
    dataset_processed = avgerage_neighboring_signal(dataset)
    dataset_norm = z_score_normalisation(dataset_processed)

    
    general_results_dir='results'
    
    tot0 = timeit.default_timer()
    run_experiment.DoExperiment(descriptor='AE_unmix',
                                general_results_dir=general_results_dir,
                                autoencoder = AutoEncoder,
                                chosen_dataset = dataset_norm,
                                num_epochs=10, patience = 5,
                                batch_size = 64)
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
