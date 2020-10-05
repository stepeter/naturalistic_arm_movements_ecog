'''
Functions for creating context encoding models
'''

import os,mne,pdb,glob,natsort
os.environ["OMP_NUM_THREADS"] = "1" #avoid multithreading if have Anaconda numpy
import numpy as np
import pandas as pd
from datetime import datetime as dt
import statsmodels.formula.api as smf
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize,LinearSegmentedColormap

import config

def user_def_f_aves(sbj_id):
    '''
    Define subject-specific frequency bands for power features used in regression
    '''
    f_lo,f_hi = [8,32],[76,100]
#     if sbj_id==0:
#         f_lo,f_hi = [5,32],[96,120]
#     elif sbj_id==1:
#         f_lo,f_hi = [5,21],[70,90]
#     elif sbj_id==2:
#         f_lo,f_hi = [5,21],[80,120]
#     elif sbj_id==3:
#         f_lo,f_hi = [5,11],[80,120]
#     elif sbj_id==4:
#         f_lo,f_hi = [5,21],[70,90]
#     elif sbj_id==5:
#         f_lo,f_hi = [11,26],[80,120]
#     elif sbj_id==6:
#         f_lo,f_hi = [5,21],[80,120]
#     elif sbj_id==7:
#         f_lo,f_hi = [26,31],[80,120]
#     elif sbj_id==8:
#         f_lo,f_hi = [5,21],[60,80]
#     elif sbj_id==9:
#         f_lo,f_hi = [5,21],[70,90]
#     elif sbj_id==10:
#         f_lo,f_hi = [5,21],[80,120]
#     elif sbj_id==11:
#         f_lo,f_hi = [5,21],[60,80]
    return f_lo,f_hi


def load_power_metadata(s,tfr_loadpath):
    '''
    Create combined training and validation power and metadata for regression model.
    '''
    first_pass = 0
    #Load and combine power/metadata
    fname_tfr = natsort.natsorted(glob.glob(tfr_loadpath+'subj_'+str(s+1).zfill(2)+'*_allEpochs_tfr.h5'))

    inds_2_sel_all,df_metadata = [],[]
    for i,fname_curr in enumerate(fname_tfr):
        pow_temp = mne.time_frequency.read_tfrs(fname_curr)[0]
        keep_inds = np.nonzero(~np.isnan(np.mean(np.reshape(pow_temp.data,
                               (pow_temp.data.shape[0],-1)),axis=1)))[0] #Drop any bad epochs first
        pow_temp.data = pow_temp.data[keep_inds,...]
        if i==0:
            power = pow_temp.copy()
            df_metadata = power.metadata.copy()
        else:
            power.data = np.concatenate((power.data,pow_temp.data),axis=0)
            df_metadata = pd.concat([df_metadata, pow_temp.metadata],
                                    axis=0, ignore_index=True,sort=False)

    #Combine power data and metadata
    if first_pass == 0:
        power_all = power.copy()
        df_metadata_all = df_metadata.copy()
        first_pass = 1
    else:
        pow_tmp = power.data
        #Concatenate along epoch dimension
        power_all.data = np.concatenate((power_all.data,pow_tmp),axis=0)
        df_metadata_all = pd.concat([df_metadata_all, df_metadata.copy()],
                                     axis=0, ignore_index=True,sort=False)
    
    return power_all,df_metadata_all


def update_events_dataframe(df_in,log_cols=['run_1'],log_cols1=['reach_r','onset_velocity'],bimanual_thresh=None):
    '''
    Format events dataframe for context modelling. Can take absolute value or log of specific features.
    '''
    df_out = df_in.copy()
    df_out['patient_id'] = df_in.apply(lambda row: row['vid_name'][0:8], axis=1)
    df_out['tod'] = pd.to_timedelta(df_in['time'], unit='ms').dt.total_seconds()
    df_out['I_over_C_ratio'].fillna(0.5, inplace=True) #replace NaN's with 0.5
    for col in log_cols1:
        df_out[col] = np.log(df_in[col]+1)
    for col in log_cols:
        df_out[col] = np.log(df_in[col])
    if bimanual_thresh:
        df_in2 = df_in.copy()
        tmp_oll = df_in2['other_lead_lag'].values
        tmp_oll2 = tmp_oll.copy()
        tmp_oll2[np.isnan(tmp_oll)] = 0
        tmp_oll[np.abs(tmp_oll2)>bimanual_thresh] = np.nan
        bimanual = (1-np.isnan(tmp_oll)).tolist()
        df_out['bimanual'] = bimanual
    else:
        df_in2 = df_in.copy()
        tmp_oll = df_in2['other_lead_lag'].values
        bimanual = (1-np.isnan(tmp_oll)).tolist()
        df_out['bimanual'] = bimanual
    df_out['intercept'] = 1
    return df_out


def standardize_feats(df_in):
    '''
    Standardizes all features (subtract mean, divide by SD) excluding patient_id column.
    '''
    #Convert columns to numeric
    for col_name in list(df_in.columns)[:-1]:
        df_in[col_name] = pd.to_numeric(df_in[col_name])
    
    #Standardize feature columns
    for col_name in list(df_in.columns)[:-1]:
        df_in[col_name] = (df_in[col_name]-\
                           df_in[col_name].mean())/df_in[col_name].std()
    return df_in


def train_test_split_power_inds(n_evs,per_train):
    '''
    Computes random train/test split across events.
    '''
    len_arr = np.arange(n_evs)
    np.random.shuffle(len_arr)
    train_per = int(per_train*n_evs)
    train_inds = len_arr[:train_per]
    test_inds = len_arr[train_per:]
    return train_inds,test_inds

def train_test_split_day_balance(day_lst, per_train):
    '''
    Computes random train/test split across events,
    taking equal amounts from each day for test data.
    '''
    day_np = np.asarray(day_lst)
    day_uni = np.unique(day_np)
    n_test_day = int(((1-per_train)*len(day_lst))//len(day_uni))
    train_inds, test_inds = [], []
    for day in day_uni:
        day_inds = np.nonzero(day_np==day)[0]
        len_arr = np.arange(len(day_inds))
        np.random.shuffle(len_arr)
        train_inds.extend(day_inds[len_arr[:-n_test_day]])
        test_inds.extend(day_inds[len_arr[-n_test_day:]])
    return np.asarray(train_inds),np.asarray(test_inds)


def _forward_selected_new(data, response,d_val=False):
    """Linear model designed by forward selection. 
    Obtained at https://planspace.org/20150423-forward_selection_with_statsmodels/

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    selected: "optimal" features
               selected by forward selection
               evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
        elif current_score == best_new_score:
            current_score +=1 #break the while loop if score stays the same
    return selected


def num2written(num_in):
    '''
    Convert day digits to words.
    '''
    if num_in=='3':
        return 'three'
    elif num_in=='4':
        return 'four'
    elif num_in=='5':
        return 'five'
    elif num_in=='6':
        return 'six'
    elif num_in=='7':
        return 'seven'
    else:
        return num_in

    
def written2num(num_in):
    '''
    Convert day words to digits.
    '''
    if num_in=='three':
        return '3'
    elif num_in=='four':
        return '4'
    elif num_in=='five':
        return '5'
    elif num_in=='six':
        return '6'
    elif num_in=='seven':
        return '7'
    else:
        return num_in


def single_subj_regression(tfr_lp,max_chan_num,feats2use,n_perms,model_type,add_interactions=False,s=0):
    '''
    Runs single subject regression models, using n_perms random train/test splits.
    Also computes R2 and delta R2 values and outputs these values across permutations, along with coefficients.
    '''
    add_prev_pow = False
    
    #Other params: 
    n_freqs = config.constants_regress['n_freqs'] #number of spectral feature frequency bands
    t_ave = config.constants_regress['t_ave'] #time bins to average over (in sec)
    atlas = config.constants_regress['atlas']
    dipole_dens_thresh = config.constants_regress['dipole_dens_thresh']
    n_estimators = config.constants_regress['n_estimators']
    max_depth = config.constants_regress['max_depth']
    n_feats = len(feats2use)-1 #subtract 1 because 'patient_id' is not a feature
    
    feat_select_prob = np.zeros([max_chan_num,n_feats,n_freqs,n_perms])
    coefs_pats_out = np.empty([max_chan_num,n_feats+1,n_freqs,n_perms]) #n_feats+1 because includes intercept
    coefs_pats_out[:] = np.nan
    del_r2_test_reduced_pats_out = np.empty([max_chan_num,n_feats,n_freqs,n_perms])
    del_r2_test_reduced_pats_out[:] = np.nan
    del_r2_train_reduced_pats_out = del_r2_test_reduced_pats_out.copy()
    r2_train_full_pats_out = np.empty([max_chan_num,n_freqs,n_perms])
    r2_train_full_pats_out[:] = np.nan
    r2_test_full_pats_out = r2_train_full_pats_out.copy()
    for perm in tqdm(range(n_perms)):
        coefs_pats = np.empty([max_chan_num,n_feats+1,n_freqs]) #n_feats+1 because includes intercept
        coefs_pats[:] = np.nan
        del_r2_test_reduced_pats = np.empty([max_chan_num,n_feats,n_freqs])
        del_r2_test_reduced_pats[:] = np.nan
        del_r2_train_reduced_pats = del_r2_test_reduced_pats.copy()
        r2_train_full_pats = np.empty([max_chan_num,n_freqs])
        r2_train_full_pats[:] = np.nan
        r2_test_full_pats = r2_train_full_pats.copy()
        
        #Select subject-specific frequency bands
        f_ave_lo,f_ave_hi = user_def_f_aves(s)

        #Combine metadata and power across days
        power_subj,df_metadata_all = load_power_metadata(s,tfr_lp)
        print('Finished combining metadata and power!')

        df_metadata_all = update_events_dataframe(df_metadata_all,log_cols=['reach_duration'],
                                                  log_cols1=['reach_r','onset_velocity'])

        #Crop power data and average across time bins
        if add_prev_pow:
            power_prev = power_subj.copy()
            power_prev = power_prev.crop(tmin=-0.5,tmax=0)
            power_prev_t_ave = np.median(power_prev.data,axis=-1) #average across time bins
            
        power_subj = power_subj.crop(tmin=t_ave[0],tmax=t_ave[1])
        power_subj_t_ave = np.median(power_subj.data,axis=-1) #average across time bins
        
        #Average over frequency bands
        power_subj_t_ave_cp = power_subj_t_ave.copy()
        power_subj_t_ave = power_subj_t_ave[...,:n_freqs] #only need n_freqs frequencies
        f_inds_lo = np.nonzero(np.logical_and(power_subj.freqs>=f_ave_lo[0],power_subj.freqs<=f_ave_lo[1]))[0]
        f_inds_hi = np.nonzero(np.logical_and(power_subj.freqs>=f_ave_hi[0],power_subj.freqs<=f_ave_hi[1]))[0]
        power_subj_t_ave[...,0] = np.median(power_subj_t_ave_cp[...,f_inds_lo],axis=-1)
        power_subj_t_ave[...,1] = np.median(power_subj_t_ave_cp[...,f_inds_hi],axis=-1)

        #Update metadata reach angle (so between -90 and 90)
        df_metadata_all['reach_a'] = -df_metadata_all['reach_a'] # Correct angle for upper half
        df_metadata_all['reach_a'] = df_metadata_all['reach_a'].map(lambda x: x if x <= 90 else 180-x)
        df_metadata_all['reach_a'] = df_metadata_all['reach_a'].map(lambda x: x if x >= -90 else -180-x)

        #Create one-hot encoding for day information
        one_hot = pd.get_dummies(df_metadata_all['day'])

        #Create one-hot encoding for time information
        df_metadata_all['tod'] = df_metadata_all['tod']/3600
        tod=[None]*3
        tod[0] = [1 if ((val>=0) & (val<=8)) else 0 for val in df_metadata_all['tod']]
        tod[1] = [1 if ((val>=8) & (val<=16)) else 0 for val in df_metadata_all['tod']]
        tod[2] = [1 if ((val>=16) & (val<=24)) else 0 for val in df_metadata_all['tod']]
        tod_one_hot = pd.DataFrame(np.asarray(tod).T,columns=['morning','afternoon','evening'])
        
        #Standardize the features and add one-hot encodings
        df_metadata_std = standardize_feats(df_metadata_all.loc[:,feats2use])
        
        #Add interaction terms
        if add_interactions:
            feat_cols = df_metadata_std.columns[:-1]
            for day_col in one_hot.columns:
                for feat in feat_cols:
                    df_metadata_std[feat+'_'+str(day_col)] = df_metadata_std[feat]*one_hot[day_col]
                
        #Standardize the features and add one-hot encodings
        df_metadata_std = df_metadata_std.join(one_hot)
        df_metadata_std = df_metadata_std.join(tod_one_hot)
        df_metadata_std = df_metadata_std.drop('tod',axis=1)
        day_lst = df_metadata_all['day']
        df_metadata_std = df_metadata_std.drop('day',axis=1)
        
        if add_prev_pow:
            power_prev_t_ave_cp = power_prev_t_ave.copy()
            power_prev_t_ave = power_prev_t_ave[...,:n_freqs] #only need n_freqs frequencies
            f_inds_lo_prev = np.nonzero(np.logical_and(power_prev.freqs>=f_ave_lo[0],power_prev.freqs<=f_ave_lo[1]))[0]
            f_inds_hi_prev = np.nonzero(np.logical_and(power_prev.freqs>=f_ave_hi[0],power_prev.freqs<=f_ave_hi[1]))[0]
            power_prev_t_ave[...,0] = np.median(power_prev_t_ave_cp[...,f_inds_lo_prev],axis=-1)
            power_prev_t_ave[...,1] = np.median(power_prev_t_ave_cp[...,f_inds_hi_prev],axis=-1)
        
        #Find train and validation (last day) indices
#         train_inds,val_inds = train_test_split_power_inds(df_metadata_std.shape[0],per_train=.9)
        train_inds,val_inds = train_test_split_day_balance(day_lst,per_train=.8)
        
        #Perform regressions
        coef_inds_keep = len(set(df_metadata_std.columns.tolist()).intersection(feats2use))-1
        for elec_ind in tqdm(range(power_subj_t_ave.shape[1])):
            for freq_i in range(power_subj_t_ave.shape[-1]):
                df_copy = df_metadata_std.copy()
                df_copy = df_copy.drop('patient_id',axis=1)
                df_copy.columns = df_copy.columns.astype(str)
                
                #Forward selection to select only the most important features
                df_forward = df_copy.copy()
                df_forward['power'] = power_subj_t_ave[:,elec_ind,freq_i]
                new_colnames = []
                for col_val in df_forward.columns:
                    new_colnames.append(num2written(col_val))
                df_forward.columns = new_colnames.copy()
                selected_feats = _forward_selected_new(df_forward, 'power')
                new_selected_feats = []
                for col_val in selected_feats:
                    new_selected_feats.append(written2num(col_val))
                
                #Compute feature selection probability
                for feat_sel in selected_feats:
                    col_ind = np.nonzero(feat_sel==np.asarray(df_forward.columns))[0][0]
                    if col_ind<coef_inds_keep:
                        feat_ind = col_ind
                        feat_select_prob[elec_ind,feat_ind,freq_i,perm] += 1
                    elif col_ind < (coef_inds_keep+one_hot.shape[1]):
                        #(len(df_forward.columns)-4):
                        feat_ind = coef_inds_keep
                        N_days = len(df_forward.columns)-4-coef_inds_keep
                        feat_select_prob[elec_ind,feat_ind,freq_i,perm] += (1/N_days)
                    elif col_ind < (coef_inds_keep+one_hot.shape[1]+tod_one_hot.shape[1]):
                        feat_ind = coef_inds_keep+1
                        feat_select_prob[elec_ind,feat_ind,freq_i,perm] += (1/3)
                  
                bad_feats = np.setdiff1d(np.asarray(df_copy.columns),np.asarray(new_selected_feats))
                for bad_feat in bad_feats:
                    df_copy[bad_feat].values[:] = 0
                
                if add_prev_pow:
                    pow_prev = power_prev_t_ave[:,elec_ind,freq_i]
                    df_copy['prev_pow'] = (pow_prev-np.mean(pow_prev))/np.std(pow_prev)
#                     df_copy['prev_pow'] = power_prev_t_ave[:,elec_ind,freq_i]
                
                X = df_copy.to_numpy()
                y = power_subj_t_ave[:,elec_ind,freq_i]

                #Fit regression model (linear [Huber norm] or random forest)
                if model_type=='rf':
                    clf = RandomForestRegressor(n_estimators = n_estimators,max_depth = max_depth)
                else:
                    clf = HuberRegressor()
                clf.fit(X[train_inds,...],y[train_inds,...]) #fit model
                
                #Compute R2 scores and extract coefficient values
                r2_train_full = clf.score(X[train_inds,...],y[train_inds,...])
                r2_train_full_pats[elec_ind,freq_i] = r2_train_full
                r2_test_full = clf.score(X[val_inds,...],y[val_inds,...])
                r2_test_full_pats[elec_ind,freq_i] = r2_test_full
                if model_type=='rf':
                    coefs_pats[elec_ind,:,freq_i] = 0
                else:
                    day_std = np.std(np.asarray(clf.coef_[coef_inds_keep:-3]))
                    tod_std = np.std(np.asarray(clf.coef_[-3:]))
                    coefs_np = np.asarray([clf.intercept_]+clf.coef_.tolist()[:coef_inds_keep]+[day_std]+[tod_std])
                    coefs_pats[elec_ind,:,freq_i] = coefs_np
                
                #Reduced models for change in R2
                del_r2_test_reduced,del_r2_train_reduced = [],[]
                for j in range(coef_inds_keep+2):
                    if len(np.nonzero(X[:,j])[0])>0:
                        if j<coef_inds_keep:
                            #Shuffle ind var
                            X_shuffled = X.copy()
                            X_col = X_shuffled[:,j]
                            np.random.shuffle(X_col)
                            X_shuffled[:,j] = X_col
                        elif j==coef_inds_keep:
                            #Shuffle day variables
                            day_colnames = one_hot.columns.tolist()
                            X_shuffled = X.copy()
                            X_col = df_copy.copy()
                            X_col = X_col.sample(frac=1).reset_index(drop=True) #shuffle rows
                            for colname in day_colnames:
                                col_ind_X = np.nonzero(np.asarray(df_copy.columns)==str(colname))[0][0]
                                X_shuffled[:,col_ind_X] = X_col[str(colname)].to_numpy()
                        elif j==(coef_inds_keep+1):
                            #Shuffle time of day variables
                            tod_colnames = tod_one_hot.columns.tolist()
                            X_shuffled = X.copy()
                            X_col = df_copy.copy()
                            X_col = X_col.sample(frac=1).reset_index(drop=True) #shuffle rows
                            for colname in tod_colnames:
                                col_ind_X = np.nonzero(np.asarray(df_copy.columns)==colname)[0][0]
                                X_shuffled[:,col_ind_X] = X_col[colname].to_numpy()

                        #Compute model
                        if model_type=='rf':
                            clf = RandomForestRegressor(n_estimators = n_estimators,max_depth = max_depth)
                        else:
                            clf = HuberRegressor()
                        clf.fit(X_shuffled[train_inds,...],y[train_inds,...])
                        r2_test_reduced = clf.score(X_shuffled[val_inds,...],y[val_inds,...])
                        del_r2_test_reduced_pats[elec_ind,j,freq_i] = r2_test_full-r2_test_reduced
                        r2_train_reduced = clf.score(X_shuffled[train_inds,...],y[train_inds,...])
                        del_r2_train_reduced_pats[elec_ind,j,freq_i] = r2_train_full-r2_train_reduced
                    else:
                        del_r2_test_reduced_pats[elec_ind,j,freq_i] = 0
                        del_r2_train_reduced_pats[elec_ind,j,freq_i] = 0
        
        coefs_pats_out[...,perm] = coefs_pats.copy()
        r2_train_full_pats_out[...,perm] = r2_train_full_pats.copy()
        r2_test_full_pats_out[...,perm] = r2_test_full_pats.copy()
        del_r2_test_reduced_pats_out[...,perm] = del_r2_test_reduced_pats.copy()
        del_r2_train_reduced_pats_out[...,perm] = del_r2_train_reduced_pats.copy()
    
    return coefs_pats_out,r2_test_full_pats_out,r2_train_full_pats_out,\
           del_r2_test_reduced_pats_out,del_r2_train_reduced_pats_out,feat_select_prob


def compute_r2_score_colors(r2_dat_plot,del_r2_dat_plot,n_subjs,n_freqs,n_coefs,
                            reg_r2_test_ave,r2_thresh,vmax=None):
    '''
    Takes in R2 scores and delta R2 scores for each input feature and converts them to RGB colors
    based on color map defined in config file.
    '''
    cmap = get_cmap(config.constants_regress['cmap_r2'])
    red_cm = {'red':   ((0,1,1),(1,1,1)),
              'green': ((0,1,1),(1,0,0)),
              'blue':  ((0,1,1),(1,0,0))
             }
    cdict = {**red_cm,"alpha": ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0))}
    cmap_w = LinearSegmentedColormap("curr_cmap",cdict)
    if vmax==None:
        norm = Normalize(vmin=config.constants_regress['vscale_r2'][0], vmax=config.constants_regress['vscale_r2'][1])
    else:
        norm = Normalize(vmin=config.constants_regress['vscale_r2'][0], vmax=vmax)

    colors_all = np.empty([n_subjs,n_freqs,n_coefs],dtype=object)
    colors_all_r2 = np.empty([n_subjs,n_freqs],dtype=object)
    vals_all_r2imp = colors_all.copy()
    for i in range(n_subjs):
        tmp_val_lo = del_r2_dat_plot[i,:,:,0]
        tmp_val_hi = del_r2_dat_plot[i,:,:,1]

        reg_r2_test_lo = r2_dat_plot[i,:,0]
        reg_r2_test_hi = r2_dat_plot[i,:,1]
        for s in range(n_coefs):
            if s==0:
                colors_lo,colors_hi = [],[]
                for test_val in reg_r2_test_lo.tolist():
                    if str(test_val) != 'nan':
                        colors_lo.append(cmap(norm(test_val))[0:3])
                for test_val in reg_r2_test_hi.tolist():
                    if str(test_val) != 'nan':
                        colors_hi.append(cmap(norm(test_val))[0:3])
                colors_all[i,0,s] = colors_lo.copy()
                colors_all[i,1,s] = colors_hi.copy()
                vals_all_r2imp[i,0,s] = reg_r2_test_lo.tolist()
                vals_all_r2imp[i,1,s] = reg_r2_test_hi.tolist()

                tmp_val_lo_r2 = reg_r2_test_ave[i,:,0].tolist() #test_full_all[i][::2]
                tmp_val_lo_r2 = [val for val in tmp_val_lo_r2 if str(val) != 'nan']
                tmp_val_lo_r2 = [val if val>r2_thresh else -.1 for val in tmp_val_lo_r2 ]
                tmp_val_hi_r2 = reg_r2_test_ave[i,:,1].tolist() #test_full_all[i][1::2]
                tmp_val_hi_r2 = [val for val in tmp_val_hi_r2 if str(val) != 'nan']
                tmp_val_hi_r2 = [val if val>r2_thresh else -.1 for val in tmp_val_hi_r2 ]
                colors_lo_w,colors_hi_w = [],[]
                for j in range(len(tmp_val_lo_r2)):
                    colors_lo_w.append(cmap_w(norm(tmp_val_lo_r2[j]))[0:3])
                for j in range(len(tmp_val_hi_r2)):
                    colors_hi_w.append(cmap_w(norm(tmp_val_hi_r2[j]))[0:3])
                colors_all_r2[i,0] = colors_lo_w.copy()
                colors_all_r2[i,1] = colors_hi_w.copy()
            else:
                colors_lo,colors_hi = [],[]
                vals_lo,vals_hi = [],[]
                for j in range(tmp_val_lo.shape[0]):
                    if str(tmp_val_lo[j,s-1]) != 'nan':
                        colors_lo.append(cmap(norm(tmp_val_lo[j,s-1]))[0:3])
                        vals_lo.append(tmp_val_lo[j,s-1])
                colors_all[i,0,s] = colors_lo.copy()
                for j in range(tmp_val_hi.shape[0]):
                    if str(tmp_val_hi[j,s-1]) != 'nan':
                        colors_hi.append(cmap(norm(tmp_val_hi[j,s-1]))[0:3])
                        vals_hi.append(tmp_val_hi[j,s-1])
                colors_all[i,1,s] = colors_hi.copy()
                vals_all_r2imp[i,0,s] = vals_lo.copy()
                vals_all_r2imp[i,1,s] = vals_hi.copy()
    return colors_all,colors_all_r2,vals_all_r2imp,cmap

def compute_reg_coef_colors(coefs_pats_all,n_subjs,n_freqs,n_coefs,plot_sd_coef=False):
    '''
    Takes in regression coefficient values for each input feature (and intercept) and converts them to RGB colors
    based on color map defined in config file.
    '''
    if plot_sd_coef:
        vscale = config.constants_regress['vscale_coef_sd']
        vscale_intercept = config.constants_regress['vscale_coef_sd']
    else:
        vscale = config.constants_regress['vscale_coef']
        vscale_intercept = config.constants_regress['vscale_intercept']
    cmap = get_cmap(config.constants_regress['cmap_coef'])

    colors_all = np.empty([n_subjs,n_freqs,n_coefs],dtype=object)
    vals_all_coefs = colors_all.copy()
    for i in range(n_subjs):
        numer_len = np.sum(1-np.isnan(coefs_pats_all[i,:,0,0]))
        for s in range(n_coefs):
            if s==0:
                norm = Normalize(vmin=vscale_intercept[0], vmax=vscale_intercept[1])
            else:
                norm = Normalize(vmin=vscale[0], vmax=vscale[1])
            colors_lo,colors_hi = [],[]
            for j in range(numer_len):
                tmp_val_lo = coefs_pats_all[i,j,s,0]
                tmp_val_hi = coefs_pats_all[i,j,s,1]
                colors_lo.append(cmap(norm(tmp_val_lo))[0:3])
                colors_hi.append(cmap(norm(tmp_val_hi))[0:3])
            colors_all[i,0,s] = colors_lo.copy()
            colors_all[i,1,s] = colors_hi.copy()

            vals_all_coefs[i,0,s] = coefs_pats_all[i,:,s,0]
            vals_all_coefs[i,1,s] = coefs_pats_all[i,:,s,1]
    return colors_all,vals_all_coefs,cmap

def compute_ind_subj_r2_colors(reg_r2_test_ave,n_subjs,n_freqs,r2_thresh):
    '''
    Computes full model R2 score colors for plotting single-subject results.
    '''
    norm = Normalize(vmin=config.constants_regress['vscale_r2'][0], vmax=config.constants_regress['vscale_r2'][1])
    cmap = get_cmap(config.constants_regress['cmap_r2'])
    
    colors_all_r2_new = np.empty([n_subjs,n_freqs],dtype=object)
    vals_all_r2 = colors_all_r2_new.copy()
    for i in range(n_subjs):
        tmp_val_lo = reg_r2_test_ave[i,:,0].tolist()
        tmp_val_lo = [val for val in tmp_val_lo if str(val) != 'nan']
        tmp_val_hi = reg_r2_test_ave[i,:,1].tolist()
        tmp_val_hi = [val for val in tmp_val_hi if str(val) != 'nan']
        tmp_val_lo = [val if val>r2_thresh else 0 for val in tmp_val_lo ]
        tmp_val_hi = [val if val>r2_thresh else 0 for val in tmp_val_hi ]
        colors_lo,colors_hi = [],[]
        for j in range(len(tmp_val_lo)):
            colors_lo.append(cmap(norm(tmp_val_lo[j]))[0:3])
        for j in range(len(tmp_val_hi)):
            colors_hi.append(cmap(norm(tmp_val_hi[j]))[0:3])
        colors_all_r2_new[i,0] = colors_lo.copy()
        colors_all_r2_new[i,1] = colors_hi.copy()
        vals_all_r2[i,0] = tmp_val_lo.copy()
        vals_all_r2[i,1] = tmp_val_hi.copy()
    return colors_all_r2_new,vals_all_r2