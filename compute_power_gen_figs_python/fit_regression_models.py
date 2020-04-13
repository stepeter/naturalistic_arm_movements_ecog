import os,pdb,argparse
os.environ["OMP_NUM_THREADS"] = "1" #avoid multithreading if have Anaconda numpy
import numpy as np

#Packages for multiprocessing
from multiprocessing import Pool
from functools import partial

#Custom packages
import config
from regression_utils import single_subj_regression


def main(tfr_lp,save_folder,feats2use,n_perms,n_subjs,n_proc,model_type):
    '''
    Perform robust linear regression between behavioral features and spectral power.
    Allows for running subjects in parallel to speed up computation time (see n_proc variable).
    The only required inputs are the load/save pathnames and directory pathname for the projection matrices.
    '''
    #Load config params
    max_chan_num = config.constants_regress['max_chan_num'] #more than maximum number of channels for any subject
    n_freqs = config.constants_regress['n_freqs'] #number of spectral feature frequency bands
    chunksize = config.constants_regress['chunksize'] #used for parallel processes (values other than 1 may decrease run time)

    #Initialize coefficient, R2, and delta R2 variables
    coefs_pats_out = np.empty([n_subjs,max_chan_num,len(feats2use),n_freqs,n_perms])
    coefs_pats_out[:] = np.nan
    del_r2_test_reduced_pats_out = np.empty([n_subjs,max_chan_num,len(feats2use)-1,n_freqs,n_perms])
    del_r2_train_reduced_pats_out = del_r2_test_reduced_pats_out.copy()
    r2_train_full_pats_out = np.empty([n_subjs,max_chan_num,n_freqs,n_perms])
    r2_train_full_pats_out[:] = np.nan
    r2_test_full_pats_out = r2_train_full_pats_out.copy()
    feat_select_prob = del_r2_test_reduced_pats_out.copy()

    #Run regressions separately for each patient 
    if n_proc>1:
        #Run parallel processes
        pool = Pool(n_proc)
        for s, res in enumerate(pool.imap(partial(single_subj_regression,tfr_lp,max_chan_num,feats2use,
                                                  n_perms,model_type),range(n_subjs), chunksize)):
            #Store function output
            coefs_pats_out[s,...],r2_test_full_pats_out[s,...],r2_train_full_pats_out[s,...],\
            del_r2_test_reduced_pats_out[s,...],del_r2_train_reduced_pats_out[s,...],feat_select_prob[s,...] = res
    else:
        for s in range(n_subjs):
            res = single_subj_regression(tfr_lp,max_chan_num,feats2use,n_perms,model_type,s)
            #Store function output
            coefs_pats_out[s,...],r2_test_full_pats_out[s,...],r2_train_full_pats_out[s,...],\
            del_r2_test_reduced_pats_out[s,...],del_r2_train_reduced_pats_out[s,...],feat_select_prob[s,...] = res

    #Save results
    np.save(save_folder+'reg_coefs.npy',coefs_pats_out)
    np.save(save_folder+'reg_r2_train.npy',r2_train_full_pats_out)
    np.save(save_folder+'reg_r2_test.npy',r2_test_full_pats_out)
    np.save(save_folder+'del_r2_train.npy',del_r2_train_reduced_pats_out)
    np.save(save_folder+'del_r2_test.npy',del_r2_test_reduced_pats_out)
    np.save(save_folder+'feat_select_prob.npy',feat_select_prob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute ECoG power spectrograms')
    parser.add_argument('-tflp','--tfr_lp', required=True,help='Folder to load ECoG spectral power from')
    parser.add_argument('-sp','--save_folder',required=True,help='Folder to save regression results')
    parser.add_argument('-fts','--feats2use', type=str, nargs='+', required=False,
                        default=['day','tod','reach_duration','reach_r','reach_a','onset_velocity','audio_ratio',
                         'I_over_C_ratio','other_reach_overlap','bimanual','patient_id'],
                        help='Behavioral feature independent variables (and patient_id)')
    parser.add_argument('-nper','--n_perms', required=False, default=200, type=int,
                        help='Number of random train/test splits per subject')
    parser.add_argument('-nsbj','--n_subjs', required=False, default=12, type=int,
                        help='Number of subjects to run regressions over')
    parser.add_argument('-npc','--n_proc', required=False, default=4, type=int,
                        help='Number of processes to run at once (how many CPU cores to use)')
    parser.add_argument('-mod','--model_type', required=False, default='linear', type=str,
                        help='Type of model to fit (linear or rf [random forest])')
    args = parser.parse_args()
            
    main(args.tfr_lp,args.save_folder,args.feats2use,args.n_perms,args.n_subjs,args.n_proc,args.model_type)