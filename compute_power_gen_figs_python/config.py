import numpy as np

root_dir = '/home/stepeter/AJILE/stepeter_sandbox/ECoG_mvmt_init/data_release_scripts/'

constants_compute_tfr = {'pad_val' : 0.5,
                         'alpha' : 0.05,
                         'n_perms' : 2000,
                         'freqs' : np.arange(1, 124, 5),
                         'decim' : 50,
                         'baseline_vals' : [-1.5,-1]
                        } #units: pad_val (sec), freqs (Hz), baseline_vals (sec)

constants_plot_tfr = {'epoch_times' : [-1.5,1.51],
                      'vscale_val' : 1,
                      'atlas' : 'aal',
                      'alpha' : 0.05,
                      'n_perms' : 2000,
                      'stat_method' : 'fdr_bh',
                      'baseline_vals' : [-1.5,-1],
                      'epoch_times_plot' : [-1.5,1.51],
                      'vscale_val' : 1,
                      'roi_title_cols' : [[1,0,0],[0,0,1],[0,1,0],[1,0,1],[0,1,1],[.8,.8,0],[1,.5,0],[.5,0,1],[0,.5,1],[0,1,.5],[1,0,.5],[.6,.3,.2]],
                      'roi_label_dict' : {'Frontal_Mid_L' : 'Middle Frontal','Precentral_L' : 'Precentral',
                                          'Postcentral_L' : 'Postcentral','Temporal_Sup_L' : 'Superior Temporal',
                                          'SupraMarginal_L' : 'Supramarginal','Parietal_Inf_L' : 'Inferior Parietal',
                                          'Temporal_Mid_L' : 'Middle Temporal','Temporal_Inf_L' : 'Inferior Temporal'}
                     } #units: epoch_times (sec), vscale_val (dB)


constants_hand_clench_tfr = {'metadata_pos_min_time_clenches' : -2.5,
                            'pwr_decim_clenches' : 120,
                            'fs_ecog' : 1220.703125,
                            'baseline_vals' : [-1.1,-0.5]
                            } #units: pwr_decim_clenches (Hz), fs_ecog (Hz), others (sec)

constants_plot_chan_tfr_compare = {'vscale_val' : 2,
                                   'cmap_label' : 'RdBu_r'
                                  }
#Data sampling rates
fs_video = 30 #fps
fs_ecog = 500 #Hz

#Start time of metadata position traces
metadata_pos_min_time = -1.5 #sec
baseline_vals_movement = [-1.5,-1] #[-0.5,0] #sec, baseline interval for behavior traces

#Regression constants
constants_regress = {'chunksize' : 1,
                     'n_freqs' : 2,
                     'max_chan_num' : 200,
                     't_ave' : [0,.5],
                     'atlas' : 'aal',
                     'dipole_dens_thresh' : 3,
                     'n_estimators' : 150,
                     'max_depth' : 8,
                     'vscale_r2' : [0,0.1],
                     'cmap_r2' : 'YlGn',
                     'vscale_coef' : [-1,1],
                     'vscale_coef_sd' : [-1,1],
                     'vscale_intercept' : [-1,1],
                     'cmap_coef' : 'RdBu_r',
                     'r2_thresh' : 0,
                     'chan_labels' : 'allgood',
                     'zero_rem_thresh' : .99,
                     'coef_plt_titles' : ['Intercept','Reach\nDuration','Reach\nMagnitude',
                                          'Reach\nAngle','Onset\nVelocity','Audio ratio',
                                          'Bimanual\nratio','Bimanual lag','Bimanual','Day (SD)',
                                          'Time of\nday (SD)'],
                     'r2_plt_titles' : ['','Reach\nDuration','Reach\nMagnitude','Reach\nAngle',
                                        'Onset\nVelocity','Audio ratio','Bimanual\nratio',
                                        'Bimanual lag','Bimanual','Day','Time of\nday']
                    } #units: t_ave (sec), vscale_coef (dB), vscale_intercept (dB)


