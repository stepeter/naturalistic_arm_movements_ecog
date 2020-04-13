import os,pdb
os.environ["OMP_NUM_THREADS"] = "1" #avoid multithreading if have Anaconda numpy
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import config
from pandas.plotting import scatter_matrix
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

def update_behavioral_features(df_metadata):
    '''
    Transforms behavioral/environmental features
    '''
    #Set reach angle to range from -90 to 90
    df_metadata['reach_a'] = -df_metadata['reach_a'] # Correct angle for upper half
    df_metadata['reach_a'] = df_metadata['reach_a'].map(lambda x: x if x <= 90 else 180-x)
    df_metadata['reach_a'] = df_metadata['reach_a'].map(lambda x: x if x >= -90 else -180-x)
    
    #Convert time of day to seconds
    df_metadata['tod'] = pd.to_timedelta(df_metadata['time'], unit='ms').dt.total_seconds()
    df_metadata['I_over_C_ratio'].fillna(0.5, inplace=True) #set NaN's to 0.5 (away from ratio extremes)
    
    #Re-compute bimanual feature based on other_lead_lag variable
    tmp_oll = df_metadata['other_lead_lag'].values
    bimanual = (1-np.isnan(tmp_oll)).tolist()
    df_metadata['bimanual'] = bimanual
    return df_metadata

def feat_plot_unit_conversions(df_metadata):
    '''
    Converts features to more interpretable units prior to plotting.
    '''
    df_metadata_plot = df_metadata.copy()
    df_metadata_plot['tod'] = df_metadata['tod']/3600 #convert to hours
    df_metadata_plot['reach_duration'] = df_metadata['reach_duration']/config.fs_video
    df_metadata_plot['other_reach_overlap'] = df_metadata['other_reach_overlap']/config.fs_video
    df_metadata_plot['onset_velocity'] = df_metadata['onset_velocity']*config.fs_video
    
    return df_metadata_plot

def feat_plot(df_metadata_plot,feature_labels,n_subjs):
    '''
    Plots feature data to match paper figure.
    '''
    titles_dict = {'day':'Day of\nrecording','tod':'Time of\nday','run_1':'Reach\nduration',
                   'reach_r':'Reach\nmagnitude','reach_a':'Reach\nangle',
                   'onset_velocity':'Onset\nspeed','audio_ratio':'Speech\nratio',
                   'C_over_I_ratio':'Bimanual\nratio','other_lag_overlap':'Bimanual\noverlap',
                   'bimanual':'Bimanual\nclass','I_over_C_ratio':'Bimanual\nratio',
                   'other_reach_overlap':'Bimanual\noverlap',
                   'reach_duration':'Reach\nduration'}

    #Determine colors for plotting
    cols1 = sns.color_palette()
    cols2 = sns.hls_palette(8, l=.4, s=.8)
    cols2 = [val for i,val in enumerate(cols2) if ((i==5) or (i==7))]
    cols_all = cols1[:3]+[cols2[-1]]+cols1[3:6]+[cols2[0]]+cols1[6:]
    colors_out = cols_all.copy()

    fig = plt.figure(figsize=(8.8,4.6)) #8.5,4.3))
    gs = gridspec.GridSpec(nrows=n_subjs, 
                           ncols=len(feature_labels), 
                           figure=fig, 
                           width_ratios= [1]*len(feature_labels),
                           height_ratios= [1]*n_subjs,
                           wspace=0.2, hspace=-0.3
                          )
    normalize_dists = True
    y_scale_fact = [1,.1,1,.015,.02,.02,4,8,1.5,1]
    for p,feat in enumerate(feature_labels):
        ax = [None]*(n_subjs + 1)
        ylims_all,xlims_all_min,xlims_all_max = [],[],[]
        yax_unnorm = []
        for s in range(n_subjs):
            patient = 'subj_'+str(s+1).zfill(2)
            ax[s] = fig.add_subplot(gs[s, p])
            if (feat=='day'):
                bar_width = .6
                bins_use = []
                for k in [3,4,5,6,7]:
                    bins_use.append(k-bar_width/2)
                    bins_use.append(k+bar_width/2)
                sns.distplot(df_metadata_plot.loc[df_metadata_plot['patient_id']==patient,feat],
                             bins=np.asarray(bins_use), kde=False,color=colors_out[s],ax=ax[s],norm_hist=normalize_dists,hist_kws=dict(alpha=1))
                ax[s].set_xlabel('')
                ax[s].axhline(0,color=colors_out[s])
            elif (feat=='bimanual'):
                bar_width = .35
                bins_use = []
                for k in range(2):
                    bins_use.append(k-bar_width/2)
                    bins_use.append(k+bar_width/2)
                sns.distplot(df_metadata_plot.loc[df_metadata_plot['patient_id']==patient,feat],
                             bins=np.asarray(bins_use), kde=False,color=colors_out[s],ax=ax[s],norm_hist=normalize_dists,hist_kws=dict(alpha=1))
                ax[s].set_xlabel('')
                ax[s].axhline(0,color=colors_out[s])
            else:
                if normalize_dists:
                    sns.kdeplot(df_metadata_plot.loc[df_metadata_plot['patient_id']==patient,feat],
                                shade=True,color=colors_out[s],ax=ax[s],legend=False,alpha=1)
                    sns.kdeplot(df_metadata_plot.loc[:,feat],ax=ax[s],legend=False,alpha=0)
                else:
                    sns.kdeplot(df_metadata_plot.loc[df_metadata_plot['patient_id']==patient,feat],
                                shade=True,color=colors_out[s],ax=ax[s],legend=False,alpha=1)
                    yax = ax[s].yaxis
                    sns.distplot(df_metadata_plot.loc[df_metadata_plot['patient_id']==patient,feat],
                                 kde=False,color=colors_out[s],ax=ax[s],norm_hist=False,hist_kws=dict(alpha=0))
                    ax[s].set_xlabel('')
                    yax_unnorm.append(ax[s].get_ylim()[1])
                    ax[s].yaxis = yax
            if s==0:
                plt.title(titles_dict[feat],fontsize=10,fontweight='normal',pad=4,fontname='Times New Roman')
            ylims_all.append(ax[s].get_ylim()[1])
            xlims_all_min.append(ax[s].get_xlim()[0])
            xlims_all_max.append(ax[s].get_xlim()[1])
            ax[s].spines['left'].set_visible(False)
            ax[s].set_yticks([])

            if s!=(n_subjs-1):
                ax[s].spines['bottom'].set_visible(False)
                ax[s].set_xticks([])
            else:
                if feat=='day':
                    ax[s].set_xticks([3,5,7])
                    ax[s].spines['bottom'].set_bounds(3,7)
                elif feat=='tod':
                    ax[s].set_xticks([0,12,24])
                    ax[s].spines['bottom'].set_bounds(0,24)
                    ax[s].set_xlabel('(hr)',fontsize=9,fontweight='normal',fontname='Times New Roman',labelpad=0)
                elif feat=='reach_duration':
                    ax[s].set_xticks([0,2,4])
                    ax[s].spines['bottom'].set_bounds(0,4)
                    ax[s].set_xlabel('(sec)',fontsize=9,fontweight='normal',fontname='Times New Roman',labelpad=0)
                elif feat=='reach_a':
                    ax[s].set_xticks([-90,0,90])
                    ax[s].spines['bottom'].set_bounds(-90,90)
                    ax[s].set_xlabel('($^\circ$)',fontsize=9,fontweight='normal',fontname='Times New Roman',labelpad=0)
                elif feat=='audio_ratio':
                    ax[s].set_xticks([0,0.7])
                    ax[s].spines['bottom'].set_bounds(0,0.7)
                elif feat=='reach_r':
                    ax[s].set_xlabel('(px)',fontsize=9,fontweight='normal',fontname='Times New Roman',labelpad=0)
                elif feat=='onset_velocity':
                    ax[s].set_xlabel('(px/sec)',fontsize=9,fontweight='normal',fontname='Times New Roman',labelpad=0)
                elif feat=='other_reach_overlap':
                    ax[s].set_xticks([0,4])
                    ax[s].spines['bottom'].set_bounds(0,4)
                    ax[s].set_xlabel('(sec)',fontsize=9,fontweight='normal',fontname='Times New Roman',labelpad=0)
                elif feat=='bimanual':
                    ax[s].set_xticks([0,1])
                    ax[s].spines['bottom'].set_bounds(0,1)
                    ax[s].set_xticklabels(['Uni','Bi'])
                plt.setp(ax[s].get_xticklabels(), fontsize=8,fontweight='normal', fontname="Times New Roman")
            ax[s].spines['right'].set_visible(False)
            ax[s].spines['top'].set_visible(False)
            ax[s].patch.set_alpha(0)
        if p==0:
            if not normalize_dists:
                use_ylim = np.max(ylims_all)+50 #expand y axis for recording day
            else:
                use_ylim = np.max(ylims_all)
        else:
            use_ylim = np.max(ylims_all)
        use_xlim_max = np.mean(xlims_all_max)+np.std(xlims_all_max)
        use_xlim_min = np.mean(xlims_all_min)-np.std(xlims_all_min)

        if normalize_dists:
            for s in range(n_subjs):
                ax[s].set_ylim([0,use_ylim])
                if feat=='reach_duration':
                    ax[s].set_xlim([0,use_xlim_max])
                elif feat=='reach_a':
                    ax[s].set_xlim([-100,100])
                elif feat=='onset_velocity':
                    ax[s].set_xlim([0,150])
        else:
            if (p==0) or (p==len(feature_labels[:-1])):
                for s in range(n_subjs):
                    ax[s].set_ylim([0,use_ylim])
                    if feat=='reach_duration':
                        ax[s].set_xlim([0,use_xlim_max])
                    elif feat=='reach_a':
                        ax[s].set_xlim([-100,100])
                    elif feat=='onset_velocity':
                        ax[s].set_xlim([0,150])
            else:
                for s in range(n_subjs):
                    use_ylim_unnorm = np.max(yax_unnorm)/yax_unnorm[s]*y_scale_fact[p]
                    ax[s].set_ylim([0,use_ylim_unnorm])
                    if feat=='reach_duration':
                        ax[s].set_xlim([0,use_xlim_max])
                    elif feat=='reach_a':
                        ax[s].set_xlim([-100,100])
                    elif feat=='onset_velocity':
                        ax[s].set_xlim([0,150])
                        
def plot_feature_scattermatrix(df_metadata_plot):
    '''
    Plots scatter matrix of behavioral/environmental features to match supplemental figure.
    '''
    df_scm = df_metadata_plot.copy()
    df_scm = df_scm.iloc[:,:-1]

    #Change column names
    titles_dict = {'day':'Recording\nday','tod':'Time of\nday','run_1':'Reach\nduration',
                   'reach_r':'Reach\nmagnitude','reach_a':'Reach\nangle',
                   'onset_velocity':'Onset\nspeed','audio_ratio':'Audio\nratio',
                   'C_over_I_ratio':'Bimanual\nratio','other_lag_overlap':'Bimanual\noverlap',
                   'bimanual':'Bimanual\nclass','I_over_C_ratio':'Bimanual\nratio',
                   'other_reach_overlap':'Bimanual\noverlap',
                   'reach_duration':'Reach\nduration'
                  }
    df_cols = df_scm.columns
    df_cols2 = [titles_dict[val] for val in df_cols]
    df_scm.columns = df_cols2

    fig,axes = plt.subplots(1,1,figsize=(8,8))

    axes = scatter_matrix(df_scm, alpha=0.5,ax=axes,s=5)
    corr = df_scm.corr().values

    ticks = [[3,5,7],[0,12,24],[0,2,4],[0,500],[-90,0,90],[0,100],[0,0.7],[0,1],[0,4],[0,1]]
    bounds = [[2.7,7.3],[0,24],[0,4],[0,600],[-90,90],[0,150],[0,0.7],[0,1],[0,4],[-.2,1.2]]

    fig.subplots_adjust(wspace=0.25)
    fig.subplots_adjust(hspace=0.25)

    for i in range(df_scm.shape[1]):
        for j in range(df_scm.shape[1]):
            if i==j:
                axes[i, i].set_xticks(ticks[i])
                axes[i, i].set_xlim(bounds[i])
                axes[i, j].set_xlabel('')
                axes[i, j].set_ylabel('')
                if j==(df_scm.shape[1]-1):
                    axes[i, j].set_xticklabels(['Uni','Bi'])
                if j==0:
                    ymax = axes[i, j].get_ylim()
                    y_max_r = ymax[1]-ymax[0]
                    axes[i, j].set_yticks([ymax[0]+(y_max_r/5*.3),y_max_r/2,y_max_r-(y_max_r/5*.3)])
                    axes[i, j].set_yticklabels(['3','5','7'])
            else:
                axes[i, j].set_ylim(bounds[i])
                axes[i, j].set_xlim(bounds[j])
                if (j==0) or (i==(df_scm.shape[1]-1)):
                    axes[i, j].set_xlabel('')
                    axes[i, j].set_ylabel('')
                    axes[i, j].set_xticks(ticks[j])
                    axes[i, j].set_yticks(ticks[i])

                    if i==(df_scm.shape[1]-1):
                        axes[i, j].set_yticklabels(['Uni','Bi'])
                    if j==(df_scm.shape[1]-1):
                        axes[i, j].set_xticklabels(['Uni','Bi'])
            plt.setp(axes[i,j].get_xticklabels(), fontsize=8,fontweight='normal',
                     fontname="Times New Roman",rotation=0)
            plt.setp(axes[i,j].get_yticklabels(), fontsize=8,fontweight='normal',
                     fontname="Times New Roman",rotation=0)

    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        rect = patches.Rectangle(((bounds[j][1]-bounds[j][0])*.37+bounds[j][0],(bounds[i][1]-bounds[i][0])*.65+bounds[i][0]),
                                 (bounds[j][1]-bounds[j][0])*.6,(bounds[i][1]-bounds[i][0])*.32,
                                 linewidth=1,edgecolor='lightgray',facecolor='w',alpha=.75)
        axes[i, j].add_patch(rect)
        axes[i, j].annotate("%.2f" %(round(corr[i,j],2)+0.), (0.7, 0.8), xycoords='axes fraction', ha='center', va='center',
                            fontname='Times New Roman',fontsize=10)
        
def plot_behavior_traces(df_in,condition_label='',ylabel='',ylim_max=60,
                         plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']):
    
    fig = plt.figure(figsize=(1.89,1.76))
    ax = fig.add_axes([0.25,0.23,0.65,0.67])
    for i in range(len(df_in)):
        sns.lineplot(x='variable',y='value',data=df_in[i],ax=ax,
                     linewidth=1.5,ci='sd',estimator=np.median,color=plot_colors[i])

    ax.set_ylim([0,ylim_max])
    ax.set_xlim([-0.5,1.5])
    sns.despine()
    ax.axvline(0, color="black", linestyle="--",linewidth=1.5)
    ax.set_xticks([-0.5,0,0.5,1,1.5])
    ax.set_xticklabels(['-.5','0','.5','1','1.5'])
    plt.setp(ax.get_xticklabels(), fontsize=9, fontweight="normal", fontname="Arial")
    plt.setp(ax.get_yticklabels(), fontsize=9, fontweight="normal", fontname="Arial")
    ax.set_ylabel(ylabel,fontsize=11,fontweight='normal', fontname="Arial") 
    ax.set_xlabel('Time (sec)',fontsize=11,fontweight='normal', fontname="Arial") 
    plt.title(condition_label)
    plt.show()