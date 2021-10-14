# -*- coding: utf-8 -*-

from src.run_experiment import Experiment
from models.autoencoder import AutoEncoder
from models.clustering import PhaseClassifier
from load_dataset.exhaust import *


if __name__ == '__main__':
    
    file_path='/Users/andrewtung/Documents/Github/bcf_files/XLI_exhaust_011.bcf'
    # /Users/andrewtung/Documents/Github/bcf_files/XLI_exhaust_011.bcf
    # D:/Github/bcf_files/XLI_exhaust_011.bcf
    
    sem = SEMDataset(file_path)
        
    sem.rebin_signal(size=(2,2))
    remove_fist_peak(sem.edx_bin)
    peak_intensity_normalisation(sem.edx_bin)
    peak_denoising_PCA(sem.edx_bin, plot_results=False)

    dataset = sem.get_feature_maps()
    dataset_processed = avgerage_neighboring_signal(dataset)
    dataset_norm = z_score_normalisation(dataset_processed)
    dataset_softmax = softmax(dataset_norm)
    plot_pixel_distributions(sem, peak='Fe_Ka')

    
    general_results_dir='results'
    
    # Set up the experiment, e.g. determining the model structure
    Ex = Experiment(descriptor='AE_unmix',
                     general_results_dir=general_results_dir,
                     model = AutoEncoder, 
                     model_args={'hidden_layer_sizes':(512,256,128)},
                     chosen_dataset = dataset_softmax)
    
    # Train the model
    Ex.run_model(num_epochs=10, 
                 patience=10, 
                 batch_size=64,
                 learning_rate=1e-4, 
                 weight_decay=0.0, 
                 task='train_eval',
                 noise_added=0.02,
                 criterion='MSE'
                 )
    
    # Load the trained model file 
    Ex.load_trained_model('results/2021-10-12_Model-AE_unmix/params/Model-AE_unmix_epoch49')
    latent = Ex.get_latent()
    
    # Set up an object for GM clustering
    PC = PhaseClassifier(latent, dataset_norm, sem, method='GaussianMixture', 
                         method_args={'n_components':8,
                                      'random_state':4})
    
    # Plot latent sapce (2-dimensional) with corresponding Gaussian models
    PC.plot_latent_space()
    
    # Plot probability of each pixel for each cluster
    PC.plot_phase_distribution()
    
    # Plot phase map using the corresponding GM model
    PC.plot_phase_map()
    
    # Given a cluster, plot the binary map and the x-ray profile from the corresponding
    # pixels in the binary map.
    PC.plot_binary_map_edx_profile(cluster_num=1, 
                                   binary_filter_args={'threshold':0.6, 
                                                       'denoise':False, 
                                                       'keep_fraction':0.13, 
                                                       'binary_filter_threshold':0.2},
                                    save=None)
    
    # Given a cluster, output a pandas dataframe containing statistical information
    stat_info = PC.phase_statics(cluster_num=1,element_peaks=['Fe_Ka', 'O_Ka'],
                                 binary_filter_args={'threshold':0.6, 
                                                       'denoise':False, 
                                                       'keep_fraction':0.13, 
                                                       'binary_filter_threshold':0.2})
    