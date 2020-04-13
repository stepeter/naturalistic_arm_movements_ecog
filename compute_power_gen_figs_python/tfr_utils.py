import os,mne,argparse,random,natsort,glob,pdb
os.environ["OMP_NUM_THREADS"] = "1" #avoid multithreading if have Anaconda numpy
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_morlet
from math import ceil
import statsmodels.stats.multitest as smm
from scipy.stats import percentileofscore
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from nilearn import plotting as ni_plt
import pingouin as pg
from tqdm import tqdm
from astropy.stats import mad_std
import seaborn as sns

import config

'''
Utility functions for computing and plotting spectral power.
'''

def compute_tfr(epochsAllMove,epoch_times,freqs = np.arange(6, 123, 3),crop_val=0.5,decim=30):
    #Time-frequency analysis (from https://plot.ly/ipython-notebooks/mne-tutorial/ -> Time-frequency analysis)
    n_cycles = freqs / 4.  # different number of cycle per frequency

    #Compute power for move trials
    print('Computing power...')
    power = tfr_morlet(epochsAllMove, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                       return_itc=False, decim=decim, n_jobs=1,average=False)
    print('Power computation complete!')
    power.crop(epoch_times[0]+crop_val, epoch_times[1]-crop_val) #trim epoch to avoid edge effects
    power.data = 10*np.log10(power.data) #convert to log scale
    power.data[np.isinf(power.data)]=0 #set infinite values to 0
    return power

def tfr_subtract_baseline(power,chan_ind,baseidx,compute_mean=False):
    """
    From MNE format to time x freq x trials with baseline subtraction
    """
    if power.data.ndim==4:
        input_power = np.squeeze(power.data[:,chan_ind,:,:]) #trials x freq x time
        input_power = np.moveaxis(input_power,0,-1) #freq x time x trials
        if compute_mean:
            baseline = np.mean(input_power[:,baseidx,:],axis=1) 
        else:
            baseline = np.median(input_power[:,baseidx,:],axis=1)
        curr_power = input_power-np.tile(np.expand_dims(baseline,1),(1,input_power.shape[1],1)) #subtract baseline
    elif power.data.ndim==3:
        input_power = np.squeeze(power.data[chan_ind,:,:]) #freq x time
        if compute_mean:
            baseline = np.mean(input_power[:,baseidx],axis=1) 
        else:
            baseline = np.median(input_power[:,baseidx],axis=1)
        curr_power = input_power-np.tile(np.expand_dims(baseline,1),(1,input_power.shape[1]))
    return curr_power

def tfr_boot_sig_mask(curr_power,baseidx,n_perms=2000,alpha=0.05,method='fdr_bh',averagePower=True,useMedian=True):
    """
    Bootstraps and significance masks time-frequency power (can perform multiple comparisons correction)
    """
    #Bootstrap and significance mask
    if not(np.isnan(alpha)):
        if useMedian:
            curr_power_ave = np.median(curr_power,axis=2)
        else:
            curr_power_ave = np.mean(curr_power,axis=2) #take mean across trials

        #Create bootstrap distribution (based on EEGLAB's bootstat function)
        num_iters = ceil(n_perms/len(baseidx))
        boot_dist = np.zeros([curr_power.shape[0],len(baseidx),num_iters])

        for n in range(num_iters):
#             print(int((n+1)*len(baseidx)))
            #Shuffle time dimension, holding freq and trials fixed
            curr_power_tmp = curr_power.copy()
            curr_power_tmp = curr_power_tmp[:,baseidx,:]
            for j in range(curr_power_tmp.shape[0]):
                for k in range(curr_power_tmp.shape[2]):
                    list_tmp = curr_power_tmp[j,:,k].tolist()
                    random.shuffle(list_tmp)
                    curr_power_tmp[j,:,k] = np.asarray(list_tmp)
#                     np.random.shuffle(curr_power_tmp[j,:,k])

            #Take median across trials
            boot_dist[:,:,n] = np.median(curr_power_tmp,2)

        #Reformat into n_perms x n_freqs
        boot_dist = boot_dist.reshape((curr_power.shape[0],len(baseidx)*num_iters)).T

        #Compute uncorrected p-values
#         alpha=0.05
        p_raw = np.zeros(list(curr_power_ave.shape))
        for j in range(p_raw.shape[0]): #freq
            for k in range(p_raw.shape[1]): #time
                percentile_temp = percentileofscore(boot_dist[:,j],curr_power_ave[j,k])/100
                if percentile_temp>0.5:
                    p_raw[j,k] = 2*(1-percentile_temp)
                else:
                    p_raw[j,k] = 2*(percentile_temp)

        #Correct p-value with FDR
        if method!='none':
            rej, pval_corr = smm.multipletests(p_raw.flatten(), alpha=alpha, method=method)[:2]
            pval_corr = pval_corr.reshape(curr_power_ave.shape)
        else:
            pval_corr = p_raw.copy()

        #Significance mask the result
        curr_masked_power_ave = np.copy(curr_power_ave)
        curr_masked_power_ave[pval_corr>=alpha]=0 #set non-significant timepoints to 0
    else:
        if averagePower:
            if useMedian:
                curr_power = np.median(curr_power,axis=2) 
            else:
                curr_power = np.mean(curr_power,axis=2) #take mean across trials
        curr_masked_power_ave = np.copy(curr_power) #no significance mask
    return curr_masked_power_ave

def str2bool(v):
    #Allows True/False booleans in argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def load_project_spectral_power(tfr_lp,roi_proj_loadpath,good_rois,n_subjs,atlas='aal',rem_bad_chans=False):
    '''
    Loads in electrode-level spectral power and projects it to ROI's. 
    '''
    first_pass = 0
    metadata_list = []
    for j in range(n_subjs):
        fname_tfr = natsort.natsorted(glob.glob(tfr_lp+'subj_'+str(j+1).zfill(2)+'*_allEpochs_tfr.h5'))
        for i,fname_curr in enumerate(fname_tfr):
            if i==0:
                power_load = mne.time_frequency.read_tfrs(fname_curr)[0]
                bad_chans = power_load.info['bads']
                ch_list = np.asarray(power_load.info['ch_names'])
            else:
                pow_temp = mne.time_frequency.read_tfrs(fname_curr)[0]
                power_load.data = np.concatenate((power_load.data,pow_temp.data),axis=0)
        metadata_list += [str(j)]*int(power_load.data.shape[0])

        #Project to ROI's
        df = pd.read_csv(roi_proj_loadpath+atlas+'_'+str(j+1).zfill(2)+'_elecs2ROI.csv')
        chan_ind_vals = np.nonzero(df.transpose().mean().values!=0)[0][1:] #+ 1
        
        if rem_bad_chans:
            inds2drop = []
            for i,bad_ch in enumerate(bad_chans):
                inds2drop.append(np.nonzero(ch_list==bad_ch)[0])
            inds2drop = np.asarray(inds2drop)
            df.iloc[inds2drop] = 0
            sum_vals = df.sum(axis=0).values
            for s in range(len(sum_vals)):
                df.iloc[:,s] = df.iloc[:,s]/sum_vals[s]

        if first_pass == 0:
            power_ROI = power_load.copy()
            #Remove channels so have some number as good ROIs
            n_ch = len(power_load.info['ch_names'])
            chs_rem = power_load.info['ch_names'][len(good_rois):]
            power_ROI.drop_channels(chs_rem)
            first_pass = 1

            for s, roi_ind in enumerate(good_rois):
                power_tmp = power_load.copy()
                normalized_weights = np.asarray(df.iloc[chan_ind_vals,roi_ind])
                pow_dat_tmp = np.moveaxis(power_tmp.data,0,-1) #move epochs to last dimension
                orig_pow_shape = pow_dat_tmp.shape
                reshaped_pow_dat = np.reshape(pow_dat_tmp,(orig_pow_shape[0],np.prod(orig_pow_shape[1:])))
                del pow_dat_tmp
                power_norm = np.dot(normalized_weights, reshaped_pow_dat)
                power_ROI.data[:,s,:,:] = np.moveaxis(np.reshape(power_norm,orig_pow_shape[1:]),-1,0)
        else:
            pow_dat_all_roi_tmp = np.zeros([power_load.data.shape[0],len(good_rois),power_load.data.shape[2],power_load.data.shape[3]])
            for s, roi_ind in enumerate(good_rois):
                power_tmp = power_load.copy()
                normalized_weights = np.asarray(df.iloc[chan_ind_vals,roi_ind])
                pow_dat_tmp = np.moveaxis(power_tmp.data,0,-1) #move epochs to last dimension
                orig_pow_shape = pow_dat_tmp.shape
                reshaped_pow_dat = np.reshape(pow_dat_tmp,(orig_pow_shape[0],np.prod(orig_pow_shape[1:])))
                del pow_dat_tmp
                power_norm = np.dot(normalized_weights, reshaped_pow_dat)
                pow_dat_all_roi_tmp[:,s,:,:] = np.moveaxis(np.reshape(power_norm,orig_pow_shape[1:]),-1,0)

            #Concatenate along epoch dimension
            power_ROI.data = np.concatenate((power_ROI.data,pow_dat_all_roi_tmp),axis=0)
    
    power_ROI._metadata = pd.DataFrame(metadata_list,columns = ['patient_id'])
    return power_ROI

def _plot_spectrogram_subplots(n_rows,n_cols,subplot_num,ax1,fig,power_ave_masked,curr_ind,
                              epoch_times=[-2.5,2.5],vscale_val=3,freq_lims=[8,120],
                              y_tick_step = 25,axvlines = [0],pad_val=0.5,log_freq_scale=[],
                              cmap='bwr',scale_one_dir=False,fontweight='bold',xtick_vals=None,
                              axvline_w = 3):
    '''
    Helper function to plot spectrograms into specified subplot arrangement, formatting x,y axes labels appropriately
    '''
    if scale_one_dir:
        vscale_val_min = 0
    else:
        vscale_val_min = -vscale_val
    power_ave_masked.plot(curr_ind, baseline=None, colorbar=False, title="", yscale='linear', tmin=epoch_times[0], tmax=epoch_times[1],vmin=vscale_val_min,vmax=vscale_val,cmap=cmap,verbose=False,axes=ax1)
    for vals in axvlines:
        ax1.axvline(vals, linewidth=axvline_w, color="black", linestyle="--")  # event
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=12, fontweight=fontweight)
    plt.setp(ax1.get_yticklabels(), fontsize=12, fontweight=fontweight)
    ax1.set_ylabel('Frequency (Hz)',fontsize=14, fontweight=fontweight)
    ax1.set_xlabel('Time (s)',fontsize=14, fontweight=fontweight)
    ax1.set_ylim(freq_lims[0],freq_lims[1]) #8,120)
    ax1.set_xlim([epoch_times[0]+pad_val,epoch_times[1]-pad_val])
    if len(log_freq_scale) == 0:
        y_tick_list = list(np.arange(0,freq_lims[1]+y_tick_step,y_tick_step))
        y_tick_list[0] = freq_lims[0]
        y_tick_list[-1] = freq_lims[1]
        ax1.set_yticks(y_tick_list)
    else:
        spaced_y_ticks = np.linspace(log_freq_scale[0],log_freq_scale[-1],len(log_freq_scale)+1)[:-1]
        half_step = np.mean(np.diff(spaced_y_ticks))/2
        y_tick_list = list(spaced_y_ticks+half_step)
        ax1.set_yticks(y_tick_list)
        log_freq_scale_str = [str(int(round(val))) for val in log_freq_scale]
        ax1.set_yticklabels(log_freq_scale_str)
    if xtick_vals is None:
        ax1.set_xticks(np.arange(epoch_times[0]+pad_val,epoch_times[1]-pad_val).tolist()) #[-2,-1,0,1,2])
    else:
        ax1.set_xticks(xtick_vals)
    ax1.spines["bottom"].set_linewidth(1.5)
    ax1.spines["left"].set_linewidth(1.5)
    for item in [fig, ax1]:
        item.patch.set_visible(False)
        
    if subplot_num//n_cols == (n_rows-1):
        #Time info in last row
        plt.setp(ax1.get_xticklabels(), fontsize=12, fontweight=fontweight)
        ax1.set_xlabel('')
    else:
        ax1.set_xlabel('')
        
    if subplot_num%n_cols == 0:
        #Frequency info in first column
        ax1.set_ylabel('')
        plt.setp(ax1.get_yticklabels(), fontsize=12, fontweight=fontweight)
    else:
        ax1.set_ylabel('')
        
def add_colorbar(f_in,vmin,vmax,cmap,width=0.025,height=0.16,horiz_pos=0.85,border_width=1.5,
                 tick_len = 0,adjust_subplots_right=0.84,label_name='',tick_fontsize=14,
                 label_fontsize=18,label_pad=15,label_y=0.6,label_rotation=0,fontweight='bold',
                 fontname='Times New Roman'):
    '''
    Adds colorbar to existing plot based on vmin, vmax, and cmap
    '''
    f12636, a14u3u43 = plt.subplots(1,1,figsize=(0,0))
    im = a14u3u43.imshow(np.random.random((10,10)), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.close(f12636)
    f_in.subplots_adjust(right=adjust_subplots_right)
    vert_pos = (1-height)/2
    cbar_ax = f_in.add_axes([horiz_pos, vert_pos, width, height])
    cbar = f_in.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([vmin,0,vmax])
    cbar.ax.set_yticklabels([vmin,0,vmax], fontsize=tick_fontsize,
                            weight=fontweight, fontname=fontname)
    cbar.ax.tick_params(length=tick_len)
    cbar.outline.set_linewidth(border_width)
    cbar.set_label(label_name,rotation=label_rotation,fontsize=label_fontsize,
                   weight=fontweight,labelpad=label_pad, y=label_y, fontname=fontname)
        
        
def plot_specs_group(power_roi_subj_ave,roi_labels,good_rois,averagePower=False,useMedian=False):
    '''
    Plot group-level spectrograms
    '''
    
    #Load config params
    epoch_times_plot = config.constants_plot_tfr['epoch_times_plot']
    vscale_val = config.constants_plot_tfr['vscale_val']
    roi_title_cols = config.constants_plot_tfr['roi_title_cols']
    roi_label_dict = config.constants_plot_tfr['roi_label_dict']
    n_perms=config.constants_plot_tfr['n_perms']
    alpha=config.constants_plot_tfr['alpha']
    stat_method=config.constants_plot_tfr['stat_method']
    baseline_vals = config.constants_plot_tfr['baseline_vals']


    baseidx = np.nonzero(np.logical_and(power_roi_subj_ave.times>=baseline_vals[0],
                                        power_roi_subj_ave.times<=baseline_vals[1]))[0]

    #Subtract median baseline from each epoch separately, before stats
    power_roi_group_ave = power_roi_subj_ave.average()
    if useMedian:
        power_roi_group_ave.data = np.median(power_roi_subj_ave.data,axis=0)
    baseline_val = np.median(power_roi_group_ave.data[...,baseidx],axis=-1)
    baseline_val_trial = np.tile(np.expand_dims(baseline_val,2),(1,1,len(power_roi_subj_ave.times)))
    for i in range(power_roi_subj_ave.data.shape[0]):
        power_roi_subj_ave.data[i,...] = power_roi_subj_ave.data[i,...] - baseline_val_trial


    #Calculate and subtract baseline (for each channel)
    for chan in range(power_roi_subj_ave.data.shape[1]):
        curr_power = np.squeeze(power_roi_subj_ave.data[:,chan,:,:]) #trials x freq x time
        curr_power = np.moveaxis(curr_power,0,-1) #freq x time x trials
        curr_masked_power_ave = tfr_boot_sig_mask(curr_power,baseidx,n_perms,alpha,stat_method,averagePower,useMedian)
        power_roi_group_ave.data[chan,...]=curr_masked_power_ave
        del curr_masked_power_ave,curr_power
    
    #Perform plotting
    n_rows=2
    n_cols=4
    f, ax = plt.subplots(n_rows,n_cols,sharex=True,sharey=True,figsize=(6*1.2,2.7*1.2)) #7.8,5.8 #(8,6)) #(10,6))
    roi_ind_order_orig=list(np.arange(power_roi_subj_ave.data.shape[1]))
    dict_roi = dict(zip(roi_ind_order_orig,[1,0,2,3,4,5,6,7])) #dict(zip([0,1,2,3,4,5,6,7],[1,0,2,5,4,3,6,7]))
    for ii,curr_ind in enumerate(roi_ind_order_orig):
        i = dict_roi[ii]
        _plot_spectrogram_subplots(n_rows,n_cols,i,ax[i//n_cols][i%n_cols],f,power_roi_group_ave,
                                  list(np.array([curr_ind])),epoch_times_plot,vscale_val,freq_lims=[4,120],
                                  y_tick_step=25,axvlines = [0],pad_val=0,xtick_vals=[-1,0,1],cmap='RdBu_r',
                                  axvline_w=2)
        plt.setp(ax[i//n_cols][i%n_cols].get_xticklabels(), fontsize=8, fontweight="normal", fontname="Times New Roman")
        plt.setp(ax[i//n_cols][i%n_cols].get_yticklabels(), fontsize=8, fontweight="normal", fontname="Times New Roman")
        ax[i//n_cols][i%n_cols].set_title(roi_label_dict[roi_labels[good_rois[curr_ind]]],
                                          fontweight='bold',fontsize=9,color=roi_title_cols[ii],pad=2)
        if ((i//n_cols)==1) & ((i%n_cols)==0):
            ax[i//n_cols][i%n_cols].set_xlabel('Time (sec)',fontsize=9, fontweight='normal', fontname="Times New Roman")
            ax[i//n_cols][i%n_cols].set_ylabel('Frequency (Hz)',fontsize=9, fontweight='normal', fontname="Times New Roman")

    count_splot = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if count_splot>=power_roi_subj_ave.data.shape[1]:
                ax[i,j].axis('off')
            count_splot+=1
            
def plot_specs_single_region(power_roi_subj_ave,region_label,roi_labels,good_rois,
                             n_subjs=12,averagePower=False,n_rows=2,n_cols=6):
    '''
    Plot spectrograms for every subject projected to 1 cortical region
    '''
    chosen_ROI_labels = [val for s,val in enumerate(roi_labels) if any(good_rois==s)]

    n_perms=config.constants_plot_tfr['n_perms']
    alpha=config.constants_plot_tfr['alpha']
    stat_method=config.constants_plot_tfr['stat_method']
    roi_label_dict=config.constants_plot_tfr['roi_label_dict']
    epoch_times_plot = config.constants_plot_tfr['epoch_times_plot']
    vscale_val = config.constants_plot_tfr['vscale_val']
    baseline_vals = config.constants_plot_tfr['baseline_vals']

    roi_label_dict_r = {v: k for k, v in roi_label_dict.items()}
    roi_label_plot = roi_label_dict_r[region_label]
    roi_ind = np.nonzero(np.asarray(chosen_ROI_labels)==roi_label_plot)[0][0]
    baseidx = np.nonzero(np.logical_and(power_roi_subj_ave.times>=baseline_vals[0],
                                        power_roi_subj_ave.times<=baseline_vals[1]))[0]
    
    f = plt.figure(figsize=(9.1,2.6))
    gs = gridspec.GridSpec(nrows=n_rows, 
                       ncols=n_cols, 
                       figure=f, 
                       width_ratios= [1]*n_cols,
                       height_ratios= [1]*n_rows,
                       wspace=.2, hspace=.15
                      )
    f.subplots_adjust(right=0.84)
    f.subplots_adjust(bottom=0.15)
    ax = [None]*(n_subjs)
    for s in range(n_subjs):
        ax[s] = f.add_subplot(gs[s//n_cols,s%n_cols])
        power_roi_one_subj = power_roi_subj_ave.average()
        power_roi_one_subj.data = power_roi_subj_ave.data[s,...]

        for chan in range(power_roi_one_subj.data.shape[0]):
            power_roi_one_subj.data[chan,...] = tfr_subtract_baseline(power_roi_one_subj,
                                                                      chan,baseidx,compute_mean=True)

        _plot_spectrogram_subplots(n_rows,n_cols,s,ax[s],f,power_roi_one_subj,
                                  list(np.array([roi_ind])),epoch_times_plot,vscale_val,freq_lims=[4,120],
                                  y_tick_step=25,axvlines = [0],pad_val=0,xtick_vals=[-1,0,1],cmap='RdBu_r',
                                  axvline_w=2) #[4,250],y_tick_step=50)

        y_text,x_text = 106,-1.38
        if s<9:
            ax[s].text(x_text,y_text,'S0'+str(s+1),fontweight='bold',fontsize=9, fontname="Times New Roman")
        else:
            ax[s].text(x_text,y_text,'S'+str(s+1),fontweight='bold',fontsize=9, fontname="Times New Roman")
        rect = patches.Rectangle((x_text-.06,y_text-3),.85,16,linewidth=1,edgecolor='lightgray',facecolor='w')
        ax[s].add_patch(rect)
        if (s%n_cols)>0:
            ax[s].set_yticklabels([])
        if (s//n_cols < (n_rows-1)):
            ax[s].set_xticklabels([])
        plt.setp(ax[s].get_xticklabels(), fontsize=8, fontweight="normal", fontname="Times New Roman")
        plt.setp(ax[s].get_yticklabels(), fontsize=8, fontweight="normal", fontname="Times New Roman")

    ax[n_cols].set_xlabel('Time (sec)',fontsize=9, fontweight='normal', fontname="Times New Roman")
    ax[n_cols].set_ylabel('Frequency (Hz)',fontsize=9, fontweight='normal', fontname="Times New Roman")

    add_colorbar(f,-vscale_val,vscale_val,plt.cm.RdBu_r,label_name='dB',fontweight='normal',
                 label_fontsize=9,tick_fontsize=8,label_pad=-15,label_y=1.25,
                 width=0.025*2/3,height=0.16*3/2,horiz_pos=.86)
    plt.show()
    
def _setup_subplot_view(locs,sides_2_display,figsize):
    """
    Decide whether to plot L or R hemisphere based on x coordinates
    """
    if sides_2_display=='auto':
        average_xpos_sign = np.mean(np.asarray(locs['x']))
        if average_xpos_sign>0:
            sides_2_display='yrz'
        else:
            sides_2_display='ylz'
    
    #Create figure and axes
    if sides_2_display=='ortho':
        N = 1
    else:
        N = len(sides_2_display)
        
    if sides_2_display=='yrz' or sides_2_display=='ylz':
        gridspec.GridSpec(0,3)
        fig,axes=plt.subplots(1,N, figsize=figsize)
    else:
        fig,axes=plt.subplots(1,N, figsize=figsize)
    return N,axes,sides_2_display

def _plot_electrodes(locs,node_size,colors,axes,sides_2_display,N,node_edge_colors,alpha,edge_linewidths):
    """
    Handles plotting
    """
    if N==1:
        ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths},
                               node_size=node_size, node_color=colors,axes=axes,display_mode=sides_2_display)
    elif sides_2_display=='yrz' or sides_2_display=='ylz':
        colspans=[5,6,5] #different sized subplot to make saggital view similar size to other two slices
        current_col=0
        total_colspans=int(np.sum(np.asarray(colspans)))
        for ind,colspan in enumerate(colspans):
            axes[ind]=plt.subplot2grid((1,total_colspans), (0,current_col), colspan=colspan, rowspan=1)
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths},
                               node_size=node_size, node_color=colors,axes=axes[ind],display_mode=sides_2_display[ind])
            current_col+=colspan
    else:
        for i in range(N):
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                                   node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths},
                                   node_size=node_size, node_color=colors,axes=axes[i],display_mode=sides_2_display[i])
    
def plot_ecog_electrodes_mni_in_order(elec_locs,bad_chans,chan_labels='all',num_grid_chans=64,colors_in=None,node_size=50,
                                      figsize=(16,6),sides_2_display='auto',node_edge_colors=None,
                                      alpha=0.5,edge_linewidths=3,ax_in=None,rem_zero_chans=False,
                                      allLH=False,zero_rem_thresh=.99,elec_col_suppl_in=None,
                                      sort_vals_in=None,sort_abs=False,rem_zero_chans_show=False,rem_show_col=[0,0,0]):
    """
    Plots ECoG electrodes from MNI coordinates in order based on a value (only for specified labels)
    
    NOTE: If running in Jupyter, use '%matplotlib inline' instead of '%matplotlib notebook'
    """ 
    for k in range(len(elec_locs)):
        #Load channel locations
        good_chan_inds = np.ones([elec_locs[k].shape[0],1])
        good_chan_inds[bad_chans[k],0] = 0
        

        if (colors_in is not None) and isinstance(colors_in, list):
            colors = colors_in.copy() #one subject
        elif (colors_in is not None) and isinstance(colors_in, np.ndarray):
            colors = colors_in[k] #multiple subjects
        else:
            colors = None
        
        if (elec_col_suppl_in is not None) and isinstance(elec_col_suppl_in, list):
            elec_col_suppl = elec_col_suppl_in.copy() #one subject
        elif (elec_col_suppl_in is not None) and isinstance(elec_col_suppl_in, np.ndarray):
            elec_col_suppl = elec_col_suppl_in[k].copy() #multiple subjects
        else:
            elec_col_suppl = None

        if (sort_vals_in is not None) and isinstance(sort_vals_in, list):
            sort_vals = sort_vals_in.copy() #one subject
        elif (sort_vals_in is not None) and isinstance(sort_vals_in, np.ndarray):
            sort_vals = sort_vals_in[k] #multiple subjects
        else:
            sort_vals = None
            
        #Create dataframe for electrode locations
        if chan_labels== 'all':
            locs = pd.DataFrame(np.concatenate(elec_locs[k],axis=1),columns=['X','Y','Z'])
        elif chan_labels== 'allgood':
            locs = pd.DataFrame(np.concatenate((elec_locs[k],good_chan_inds),axis=1),columns=['X','Y','Z','goodChanInds'])

        if (colors is not None):
            if (locs.shape[0]>len(colors)) & isinstance(colors, list):
                locs = locs.iloc[:len(colors),:]
        locs.rename(columns={'X':'x','Y':'y','Z':'z'}, inplace=True)
        chan_loc_x = elec_locs[k][:,0] #chan_info.loc['X',:].values
        
        #Remove NaN electrode locations (no location info)
        nan_drop_inds = np.nonzero(np.isnan(chan_loc_x))[0]
        locs.dropna(axis=0,inplace=True) #remove NaN locations
        if (colors is not None) & isinstance(colors, list):
            colors_new,sort_vals_new,loc_inds_2_drop = [],[],[]
            for s,val in enumerate(colors):
                if not (s in nan_drop_inds):
                    colors_new.append(val)
                    if (sort_vals is not None):
                        sort_vals_new.append(sort_vals[s])
                else:
                    loc_inds_2_drop.append(s)
            colors = colors_new.copy()
            sort_vals = sort_vals_new.copy()

            if elec_col_suppl is not None:
                loc_inds_2_drop.reverse() #go from high to low values
                for val in loc_inds_2_drop:
                    del elec_col_suppl[val]

        if chan_labels=='allgood':
            goodChanInds = good_chan_inds.copy()
            inds2drop = np.nonzero(locs['goodChanInds'].to_numpy()==0)[0]
            locs.drop(columns=['goodChanInds'],inplace=True)
            locs.drop(index=locs.index[inds2drop],inplace=True)

            if colors is not None:
                colors_new,sort_vals_new,loc_inds_2_drop = [],[],[]
                for s,val in enumerate(colors):
                    if not (s in inds2drop):
    #                     np.all(s!=inds2drop):
                        colors_new.append(val)
                        if (len(sort_vals)>0):
                            sort_vals_new.append(sort_vals[s])
                    else:
                        loc_inds_2_drop.append(s)
                colors = colors_new.copy()
                sort_vals = sort_vals_new.copy()

                if elec_col_suppl is not None:
                    loc_inds_2_drop.reverse() #go from high to low values
                    for val in loc_inds_2_drop:
                        del elec_col_suppl[val]
        
        if rem_zero_chans:
            #Remove channels with zero values (white colors)
            colors_new,sort_vals_new,loc_inds_2_drop = [],[],[]
            for s,val in enumerate(colors):
                if np.mean(val)<zero_rem_thresh:
                    colors_new.append(val)
                    if (len(sort_vals)>0):
                        sort_vals_new.append(sort_vals[s])
                else:
                    loc_inds_2_drop.append(s)
            colors = colors_new.copy()
            sort_vals = sort_vals_new.copy()
            locs.drop(index=locs.index[loc_inds_2_drop],inplace=True)

            if elec_col_suppl is not None:
                loc_inds_2_drop.reverse() #go from high to low values
                for val in loc_inds_2_drop:
                    del elec_col_suppl[val]
        elif rem_zero_chans_show:
            #Channels with zero values are white
            for s,val in enumerate(colors):
                if np.mean(val)>=zero_rem_thresh:
                    elec_col_suppl[s] = rem_show_col.copy()

        #Decide whether to plot L or R hemisphere based on x coordinates
        if len(sides_2_display)>1:
            N,axes,sides_2_display = _setup_subplot_view(locs,sides_2_display,figsize)
        else:
            N = 1
            axes = ax_in
            if allLH:
                average_xpos_sign = np.mean(np.asarray(locs['x']))
                if average_xpos_sign>0:
                    locs['x'] = -locs['x']
                sides_2_display ='l'

        if elec_col_suppl is not None:
            colors = elec_col_suppl.copy()
        
        if k == 0:
            locs2 = locs.copy()
            colors2 = colors.copy()
            sort_vals2 = sort_vals.copy()
        else:
            locs2 = pd.concat([locs2,locs],axis=0,ignore_index=True)
            colors2.extend(colors)
            sort_vals2.extend(sort_vals)
    
    #Re-order by magnitude
    if (len(colors2)>0) and (len(sort_vals2)>0):
        if sort_abs:
            #Use absolute value
            sort_vals2 = [abs(val) for val in sort_vals2]
        sort_inds = np.argsort(np.asarray(sort_vals2))
        colors2_np = np.asarray(colors2)
        colors_out = colors2_np[sort_inds,:].tolist()
        locs_out = locs2.iloc[sort_inds,:]
    else:
        colors_out = colors2.copy()
        locs_out = locs2.copy()
    
    #Plot the result
    _plot_electrodes(locs_out,node_size,colors_out,axes,sides_2_display,N,node_edge_colors,alpha,edge_linewidths)
    
def compute_wrist_radial_distance(contra_metadata_sbj,wrist_contra):
    '''
    Create variable for plotting contralateral wrist displacements. 
    '''
    #Compute baseline-subtracted displacement
    wrist_x = contra_metadata_sbj.loc[:,wrist_contra+'_wrist_x']
    wrist_x = [np.asarray(val) for val in wrist_x]
    wrist_y = contra_metadata_sbj.loc[:,wrist_contra+'_wrist_y']
    wrist_y = [np.asarray(val) for val in wrist_y]
    contra_vals = np.sqrt(np.asarray(wrist_x)**2 +np.asarray(wrist_y)**2)

    #Determine baseline in samples
    min_t_val = config.metadata_pos_min_time
    base_times = np.asarray(config.baseline_vals_movement)
    base_times = (base_times-min_t_val)*config.fs_video #baseline frame limits
    base_times_range = np.arange(base_times[0],base_times[1]+1)
    base_times_range = base_times_range.astype('int')

    #Subtract baseline from position data
    contra_vals_basesub = contra_vals.copy()
    for j in range(contra_vals_basesub.shape[0]):
        curr_magnitude = contra_vals[j,:]
        contra_vals_basesub[j,:] = np.abs(curr_magnitude - np.mean(curr_magnitude[base_times_range]))
    
    #Create time variable for plotting
    df_contra_wrist = pd.DataFrame(contra_vals_basesub).melt()
    df_contra_wrist["variable"] = df_contra_wrist["variable"]/config.fs_video+min_t_val
    return df_contra_wrist

def create_spectral_features(power,freq_bands,baseidx,t_ave_secs,condition,df_mad=None,fband_label=None):
    '''
    Compute spectral features from electrode power, using specified low/high frequency bands.
    '''
    alpha = config.constants_plot_tfr['alpha']
    n_perms = config.constants_plot_tfr['n_perms']
    method = config.constants_plot_tfr['stat_method']
    if power.data.shape[2]>2:
        pow_dat_tmp = power.data.copy()
        f_inds_lo = np.nonzero(np.logical_and(power.freqs>=freq_bands[0][0],\
                                        power.freqs<=freq_bands[0][1]))[0]
        f_inds_hi = np.nonzero(np.logical_and(power.freqs>=freq_bands[1][0],\
                                        power.freqs<=freq_bands[1][1]))[0]
        pow_dat_tmp_lo = np.expand_dims(np.mean(power.data[...,f_inds_lo,:],axis=2),2)
        pow_dat_tmp_hi = np.expand_dims(np.mean(power.data[...,f_inds_hi,:],axis=2),2)
        power.data = np.concatenate((pow_dat_tmp_lo,pow_dat_tmp_hi),axis=2)
        power.freqs = np.array([freq_bands[0][0],freq_bands[1][0]])
        
    #Insert average for first half-second at t=0
    t_ave_inds = np.nonzero(np.logical_and(power.times>=t_ave_secs[0],power.times<=t_ave_secs[1]))[0]
    tind0 = np.argmin(np.abs(power.times)) #save average data in closest time point to 0
    power.data[...,tind0] = np.mean(power.data[...,t_ave_inds],axis=-1)
    
    #Subtract baseline
    power_ave = power.copy().average()
    power_copy = power.copy()
    #Calculate and subtract baseline (for each channel)
    for chan in tqdm(range(power.data.shape[1])):
#         curr_power = np.squeeze(power.data[:,chan,:,:]) #trials x freq x time
#         curr_power = np.moveaxis(curr_power,0,-1) #freq x time x trials
        curr_power = tfr_subtract_baseline(power,chan,baseidx,compute_mean=True)
        curr_masked_power_ave = tfr_boot_sig_mask(curr_power,baseidx,n_perms,alpha,method,averagePower=False)
        curr_power = np.moveaxis(curr_power,-1,0)
        power_copy.data[:,chan,...]=curr_power
        power_ave.data[chan,...]=curr_masked_power_ave
        del curr_masked_power_ave,curr_power
    
    #Compute MAD across events
    for k in range(len(power.freqs)):
        mad_vals = mad_std(power.data[...,k,tind0],axis=0).tolist()
        n = len(mad_vals)
        df_temp = pd.DataFrame(list(zip(mad_vals,[condition]*n,[fband_label[k]]*n)),
                                columns=['MAD','Condition','Fband'])
        if df_mad is None:
            df_mad = df_temp.copy()
        else:
            df_mad = pd.concat((df_mad,df_temp),ignore_index=True)
        
    return power_ave,tind0,df_mad

def compute_voltage_variable(df_voltage_vals):
    '''
    Convert voltage metadata into variable to be plotted
    '''
    metadata_pos_min_time_clenches = config.constants_hand_clench_tfr['metadata_pos_min_time_clenches']
    Fs_cyber = config.constants_hand_clench_tfr['fs_ecog']
    
    voltage_vals = np.asarray([np.asarray(val) for val in df_voltage_vals.loc[:,'glove_voltage']])
    
    #Determine baseline in samples
    base_times = np.asarray(config.baseline_vals_movement)
    base_times = (base_times-metadata_pos_min_time_clenches)*Fs_cyber #baseline frame limits
    base_times_range = np.arange(base_times[0],base_times[1]+1)
    base_times_range = base_times_range.astype('int')

    #Subtract baseline from position data
    voltage_vals_basesub = voltage_vals.copy()
    for j in range(voltage_vals_basesub.shape[0]):
        curr_voltage = voltage_vals[j]
        voltage_vals_basesub[j,:] = np.abs(curr_voltage - np.mean(curr_voltage[base_times_range]))
    
    #Create time variable for plotting
    df_voltage = pd.DataFrame(voltage_vals_basesub).melt()
    df_voltage["variable"] = df_voltage["variable"]/Fs_cyber+metadata_pos_min_time_clenches
    return df_voltage


def plot_spectral_power_variability_stats(df_mad,cond_labels,my_pal,figsize=(1.55,3.4)):
    f, ax = plt.subplots(2,1,figsize=figsize) 
    df_LFB = df_mad.loc[df_mad['Fband']=='LFB',:]
    ax[0] = sns.boxplot(x='Condition', y='MAD',data=df_LFB,order=cond_labels, palette=my_pal,ax=ax[0])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('') #MAD',fontsize=11,fontweight='normal')
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].set_ylim(0,9)
    # ax[0].set_title('LFB',fontsize=22,fontweight='bold',pad=0)
    ax[0].set_yticks([0,3,6,9])
    plt.setp(ax[0].get_yticklabels(), fontsize=9,fontweight='normal')
    plt.setp(ax[0].get_xticklabels(), fontsize=9,fontweight='normal',rotation=-10)
    ax[0].set_xticklabels([])
    ax[0].tick_params(axis='x', which='major', pad=-2)

    # statistical annotation
    x1, x2 = 0, 1   # columns to annotate
    y, h, col = df_LFB['MAD'].max() + 0.5, 0.5, 'k'
    ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    # ax[0].text((x1+x2)*.5, y+h-.3, "*", ha='center', va='bottom', color=col, fontsize=20,fontweight='bold')

    for i,artist in enumerate(ax[0].artists):
        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        artist.set_edgecolor('k')
        for j in range(i*6,i*6+6):
            line = ax[0].lines[j]
            line.set_color('k')
            line.set_mfc('k')
            line.set_mec('k')

#     t_result = pg.pairwise_ttests(dv='MAD',between='Condition', data=df_LFB)
#     print('p-value: '+str(t_result['p-unc'][0]))

    df_HFB = df_mad.loc[df_mad['Fband']=='HFB',:]
    ax[1] = sns.boxplot(x='Condition', y='MAD',data=df_HFB,order=cond_labels, palette=my_pal,ax=ax[1])
    ax[1].set_xlabel('')
    ax[1].set_ylabel('') #MAD',fontsize=11,fontweight='normal')
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].set_ylim(0,9)
    # ax[1].set_title('HFB',fontsize=22,fontweight='bold',pad=0)
    ax[1].set_yticks([0,3,6,9])
    ax[1].set_xticklabels(cond_labels,rotation=15)
    plt.setp(ax[1].get_yticklabels(), fontsize=9,fontweight='normal')
    plt.setp(ax[1].get_xticklabels(), fontsize=9,fontweight='normal') #,rotation=-10)
    ax[1].tick_params(axis='x', which='major') #, pad=-2)

    # statistical annotation
    x1, x2 = 0, 1   # columns to annotate
    y, h, col = df_HFB['MAD'].max() + 0.5, 0.5, 'k'
    ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    # ax[1].text((x1+x2)*.5, y+h-.3, "*", ha='center', va='bottom', color=col, fontsize=20,fontweight='bold')

    for i,artist in enumerate(ax[1].artists):
        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        artist.set_edgecolor('k')
        for j in range(i*6,i*6+6):
            line = ax[1].lines[j]
            line.set_color('k')
            line.set_mfc('k')
            line.set_mec('k')

#     t_result = pg.pairwise_ttests(dv='MAD',between='Condition', data=df_HFB)
#     print('p-value: '+str(t_result['p-unc'][0]))

    plt.show()
    return  f