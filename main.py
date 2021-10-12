# -*- coding: utf-8 -*-

import timeit

from src import run_experiment
from models import autoencoder
from load_dataset import exhaust

if __name__ == '__main__':
    
    sem = exhaust.SEMDataset('/Users/andrewtung/Documents/Github/bcf_files/XLI_exhaust_011.bcf')
    
    sem.rebin_signal()
    exhaust.remove_fist_peak(sem.edx_bin)
    exhaust.peak_intensity_normalisation(sem.edx_bin)
    exhaust.peak_denoising_PCA(sem.edx_bin, plot_results=False)
    
    dataset = sem.get_feature_maps()
    dataset_processed = exhaust.avgerage_neighboring_signal(dataset)
    dataset_norm = exhaust.z_score_normalisation(dataset_processed)

    
    general_results_dir='results'
    
    tot0 = timeit.default_timer()
    run_experiment.DoExperiment(descriptor='AE_unmix',
            general_results_dir=general_results_dir,
            autoencoder = autoencoder.AutoEncoder,
            chosen_dataset = dataset_norm,
            num_epochs=1, patience = 50,
            batch_size = 64)
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
