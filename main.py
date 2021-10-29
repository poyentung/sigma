# -*- coding: utf-8 -*-

from src.run_experiment import Experiment, same_seeds
from models.autoencoder import AutoEncoder
from models.clustering import PhaseClassifier
from load_dataset.exhaust import *

import plotly.io as pio
pio.renderers.default='browser'


if __name__ == '__main__':
    
    # file_path='/home/tung/Github/bcf_files/XLI_exhaust_011.bcf'
    file_path="/Users/andrewtung/Documents/Github/bcf_files/XLI_exhaust_011.bcf"
    # D:/Github/bcf_files/XLI_exhaust_011.bcf
    # /home/tung/Github/bcf_files
    
    sem = SEMDataset(file_path)
    sem.set_feature_list(['O_Ka', 'Fe_Ka', 'Mg_Ka', 'Ca_Ka', 
                          'Al_Ka', 'C_Ka', 'Si_Ka', 'S_Ka'])
    
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
    
    
    # Fix random seed 
    same_seeds(1)
    
    # Set up the experiment, e.g. determining the model structure
    Ex = Experiment(descriptor='AE_unmix',
                    general_results_dir=general_results_dir,
                    model = AutoEncoder, 
                    model_args={'hidden_layer_sizes':(512,256,128)},
                    chosen_dataset = dataset_softmax,
                    save_model_every_epoch=True)
    
    # Train the model
    Ex.run_model(num_epochs=100, 
                 patience=50, 
                 batch_size=64,
                 learning_rate=1e-4, 
                 weight_decay=0.0, 
                 task='train_all',
                 noise_added=0.0,
                 criterion='MSE',
                 print_latent=True,
                 lr_scheduler_args={'factor':0.5,
                                    'patience':5, 
                                    'threshold':1e-2, 
                                    'min_lr':1e-7,
                                    'verbose':True}
                 )
    
    # Load the trained model file 
    Ex.load_trained_model('results/Model-AE_unmix_best')
    latent = Ex.get_latent()
    
    # Set up an object for GM clustering
    PC = PhaseClassifier(latent, dataset_softmax, sem, 
                         method='BayesianGaussianMixture', 
                         method_args={'n_components':15,
                                      'random_state':4})
    
    # Plot latent sapce (2-dimensional) with corresponding Gaussian models
    PC.plot_latent_space()
    
    # Plot probability of each pixel for each cluster
    PC.plot_phase_distribution()
    
    # Plot phase map using the corresponding GM model
    PC.plot_phase_map()
    
    # Given a cluster, plot the binary map and the x-ray profile from the corresponding
    # pixels in the binary map.
    binary_filter_args={'threshold':0.8, 
                        'denoise':False, 
                        'keep_fraction':0.13, 
                        'binary_filter_threshold':0.5}
    PC.plot_binary_map_edx_profile(cluster_num=10,
                                   binary_filter_args=binary_filter_args,
                                    save=None)
    
    # Given a cluster, output a pandas dataframe containing statistical information
    stat_info = PC.phase_statics(cluster_num=10,element_peaks=['Fe_Ka', 'O_Ka'],
                                 binary_filter_args=binary_filter_args)
    
    # Get x-ray energy profiles from all phases
    W, H = PC.get_unmixed_edx_profile(normalised=True,
                                      method_args={'init':'nndsvd'},
                                      binary_filter_args=binary_filter_args)
    