import os, mne, sys, pdb, argparse, h5py, glob, natsort
os.environ["OMP_NUM_THREADS"] = "1" #avoid multithreading if have Anaconda numpy
import numpy as np
import pandas as pd
from datetime import datetime as dt
import scipy.io as sio

#Import custom scripts
import config
from tfr_utils import compute_tfr,tfr_subtract_baseline,tfr_boot_sig_mask,str2bool

#Set parameters
def main(ecog_lp,savefolder,averagePower,stat_method): 
    '''
    Use MNE to compute spectrograms from ECoG trials (with false positive trials removed, if desired).
    '''
    
    #Load parameters
    alpha = config.constants_compute_tfr['alpha']
    n_perms = config.constants_compute_tfr['n_perms']
    pad_val = config.constants_compute_tfr['pad_val'] #(sec) used for padding each side of epochs and cropped off later
    freqs = config.constants_compute_tfr['freqs']
    decim = config.constants_compute_tfr['decim']
    baselineT = config.constants_compute_tfr['baseline_vals'] B#aseline times to subtract off (in sec, centered around events)
    
    #Find all epoch files
    fnames_all = natsort.natsorted(glob.glob(ecog_lp+'*_epo.fif'))
    
    for fname in fnames_all:
        #Load epoch data
        ep_dat = mne.read_epochs(fname)
        epoch_times = [ep_dat.times.min()+pad_val, ep_dat.times.max()-pad_val]
        bad_chans = ep_dat.info['bads'].copy() #store for later
        ep_dat.info['bads'] = []
        
        #Remove false positive events
        if 'false_pos' in ep_dat.metadata.columns:
            bad_ev_inds = np.nonzero(ep_dat.metadata['false_pos'].values)[0]
            ep_dat.drop(bad_ev_inds)
        
        #Compute TFR
        power = compute_tfr(ep_dat,epoch_times,freqs = freqs,crop_val=pad_val,decim=decim)
        power._metadata = ep_dat.metadata.copy() #add in metadata
        
        #Parameter updates
        power_ave_masked=power.copy()
        if averagePower:
            power_ave_masked = power_ave_masked.average() #remove epoch dimension for output
        else:
            alpha = np.nan #no stats if not averaging 

        #Calculate and subtract baseline (for each channel), computing stats if desired
        baseidx = np.nonzero(np.logical_and(power.times>=baselineT[0],power.times<=baselineT[1]))[0]
        for chan in range(power.data.shape[1]):
            curr_power = tfr_subtract_baseline(power,chan,baseidx,compute_mean=True)
            curr_masked_power_ave = tfr_boot_sig_mask(curr_power,baseidx,n_perms,alpha,stat_method,averagePower)

            #Return masked data to original variable
            if averagePower:
                power_ave_masked.data[chan,...]=curr_masked_power_ave
            else:
                curr_masked_power_ave = np.moveaxis(curr_masked_power_ave,-1,0)
                power_ave_masked.data[:,chan,...]=curr_masked_power_ave
            del curr_masked_power_ave, curr_power

        #Save result
        file_prefix = fname.split('/')[-1][:-8]
        if not averagePower:
            saveName = file_prefix+'_allEpochs_tfr.h5'
        else:
            saveName = file_prefix+'_ave_tfr.h5'
        power_ave_masked.info['bads'] = bad_chans.copy() #add bad channel list back in
        power_ave_masked.save(savefolder+saveName, overwrite=True)
        del power
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute ECoG power spectrograms')
    parser.add_argument('-eclp','--ecog_lp', required=True,help='Folder to load ECoG trials from')
    parser.add_argument('-sp','--savePath',required=True,help='Folder to save spectrograms')
    parser.add_argument('-avp','--averagePower', type=str2bool, nargs='?', required=False, default='False',
                        help='whether to average power across trials (True) or save single trials (False)')
    parser.add_argument('-stm','--stat_method', required=False, default='fdr_bh',
                        help='Which statistical correction to use (see statsmodels.stats.multitest.multipletests)')
    args = parser.parse_args()
            
    main(args.ecog_lp, args.savePath, args.baseline_vals, args.averagePower,args.stat_method)
    