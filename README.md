# Behavioral and neural variability of naturalistic arm movements

This repository contains scripts used to analyze the behavioral and neural variability of 12 human subjects as they performed spontaneous arm movements across multiple days. Neural data was monitored intracranially using electrocorticography (ECoG); wrist trajectories extracted from simultaneously recorded video was used to generate behavioral data. This repository is meant to be used with our published dataset (https://figshare.com/projects/Behavioral_and_neural_variability_of_naturalistic_arm_movements/78666) to replicate our analysis and generate most of the figures from our manuscript (https://doi.org/10.1101/2020.04.17.047357). Such analyses include ECoG spectral power computation and projection to pre-defined regions of interest, comparison of naturalistic reach data to cued hand clench data, and regression of spectral power on behavioral and environmental features.

## Getting started

This code was primarily run using Python 3.6.8, with one optional part requiring Matlab 2018b. Note that Matlab and Python code are separated into separate folders. Analysis is heavily reliant on MNE (https://mne.tools/stable/index.html) for data analysis and saving. Some specific package versions used:

-  Python: Matplotlib 3.0.2, MNE 0.19.0, Nilearn 0.5.0, Numpy 1.15.4, Pandas 0.25.3

-  Matlab: EEGLAB 13.5.4b, Measure Projection Toolbox 1.661

## Analysis steps

All Python scripts can be found in *compute_power_gen_figs_python*. This folder includes Jupyter notebooks to generate plots, scripts to perform longer analyses such as computing spectral power and fitting regression models, and utility scripts. There is also a config file that contains many parameters that may be shared by multiple scripts or do not need to be changed.

### Step 1: Computing spectral power (*compute_spectral_power.py*)

This script will compute spectral power via Morlet wavelet transform for ECoG segments aligned to movement onset events (FIF files, which are of type  mne.Epochs). Requires inputs for the load and save directories. This process takes several hours.

```
python compute_spectral_power.py -eclp <load_path> -sp <save_path>
```

### Step 2: Visualize behavioral and environmental features used for regression analysis (*Behavioral_features_Fig2.ipynb*)

Here, we extract behavioral and environmental metadata based on the time when each reach began (day of recording, time of day), how the contralateral wrist moved during the reach (reach duration, magnitude, angle, and onset speed), whether people were speaking during movement initiation (speech ratio), and how much both wrists moved during each movement (bimanual ratio, overlap, and class). These 10 features are later used to model changes in neural spectral power via multiple linear regression. This notebook, however, plots the normalized distributions of these 10 features for each subject. Be sure to set *tfr_lp* to be the directory with the spectral power data (TFR files). These files include the behavioral feature data in their metadata attribute.

### (Optional) Step 3: Create projection matrices (*project_ecog_to_rois.m*)

To visualize the spectral power of multiple subjects, we project the power across electrodes into pre-defined regions of interest using the Measure Projection Toolbox (https://sccn.ucsd.edu/wiki/MPT). The projection matrices are included with the dataset, but can be generated using the Matlab code in *project_data_rois_matlab*. First, open *set_environment_paths.m* and update the path names. **Note that other versions of EEGLAB may cause errors.**

Next, run *project_ecog_to_rois.m*, setting *save_proj_matrix* and *save_plots* to 1 to save the results. The plot will show all regions of interest on the left hemisphere, with the 8 regions of interests we used shown in color.

### Step 4: Visualize projected spectral power (*Projected_spectral_power_Figs3_4.ipynb*)

This notebook will plot projected power averaged across subjects for all 8 regions, along with projected power at 1 region showing each individual subject. Be sure to specify the directories where the spectral power results (TFR files) are saved using *tfr_lp* and where the projection matrices (CSV files) are saved using *roi_proj_loadpath*.


### Step 5: Plot inter-event variability in low/high-frequency band spectral power (*Plot_banded_power_Fig5.ipynb*)

This notebook generates Fig. 5 from the paper, which shows the event-by-event variability in spectral power for each subject, separated across recording days. Be sure to set the pathnames for *tfr_lp* and *roi_proj_loadpath*.


### Step 6: Fit regression models (*fit_regression_models.py*)

We now want to see how well the behavioral metadata from the previous step explains changes in the neural activity during movement initiation. This script performs separate multiple regression models for each ECoG recording electrode, performing several random train/test splits to minimize bias. For each fit model, the R2 score is computed on the withheld test data to measure how well the model generalizes to new data.

Additionally, a delta R2 score is computed for each feature, which involves randomizing that feature’s training data, fitting a new model and computing a new R2 score, and subtracting this new R2 score from the R2 score computed using non-randomized training data. Higher delta R2 scores indicate more explanatory features.

To run this script, just specify the load path (with TFR files) and save path for regression output. The type of regression model can also be specified as *linear* or *rf* (random forest). Note that this script will need to run for several hours, especially if the number of permutations is high.

```
python fit_regression_models.py -tflp <TFR_load_path> -sp <save_path>
```

### Step 7: Visualize regression results (*Regression_results_Fig6.ipynb*)

This notebook visualizes the R2 scores and coefficients obtained from the previous step. Specify the directories of the regression output (*reg_lp*) and spectral power files (*tfr_lp*). In addition, *plot_sd_coef* allows for plotting the standard deviation (True) or average (False) coefficient value across permutations.
