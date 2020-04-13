%Start eeglab
run('set_environment_paths.m')
if ~exist('ALLCOM','var')
    curr_dir = pwd;
    eeglab_dir = getenv('eeglab_dir');
    run([eeglab_dir 'eeglab.m']);
    cd(curr_dir);
    if exist([eeglab_dir 'plugins/groupSIFT_03/'],'dir')==7
        rmpath([eeglab_dir 'plugins/groupSIFT_03/']);
    end
end
addpath(genpath(getenv('mne_matlab_dir')))
close all;
proj_folder = getenv('data_load_dir');
save_path = getenv('save_dir');

save_proj_matrix = 1; keyword_roi = 'Parietal_Inf_R';
save_plots=1; plotAllRegionsTogether = 1;
hemisphere = 'L'; %'L', 'R'
%white background: set(gcf,'color','w'); set(gcf,'color','k');
atlas2use = 'aal'; %'loni','aal','brodmann'
subj_ids = {'01'}; %Add in subject numbers to load
for ppp = 1:length(subj_ids)
    rois_used = {'Precentral_L','Frontal_Mid_L',...
        'Postcentral_L','Parietal_Inf_L',...
        'SupraMarginal_L','Temporal_Sup_L','Temporal_Mid_L','Temporal_Inf_L'};
    customColors_orig = [1 0 0; 0 0 1; 0 1 0; 1 0 1; 0 1 1; .8 .8 0; 1 .5 0;...
        .5 0 1; 0 .5 1; 0 1 .5; 1 0 .5; .6,.3,.2];
    subj_id = subj_ids{ppp};

    %Load in electrode locations
    files = dir([proj_folder 'subj_' subj_ids{1} '*_epo.fif']);
    [epochs] = fiff_read_epochs([proj_folder files(1).name]);

    nchan = epochs.info.nchan;
    elec_locs = zeros(nchan,3);
    for i=1:nchan
        elec_locs(i,:) = epochs.info.chs(i).loc(1:3);
    end
    elec_locs(elec_locs(:,1)>0,1) = -elec_locs(elec_locs(:,1)>0,1); %flip all electrodes to left hemisphere

    % Add electrode location information
    uniqueDipole = pr.dipole;
    uniqueDipole.location = elec_locs;
    uniqueDipole.residualVariance = repmat(0.05,size(elec_locs,1),1); %needs to be <.15 to be retained

    % Load headGrid cubes
    headGrid = pr.headGrid;

    % Set Gaussian smoothing kernel
        % FWHM = 2.355*sigma See https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        % Note that Gaussian is NOT truncated until reaching 10*SD apart from the center
    userInputKernelSize = 20; %FWHM
    standardDeviationOfEstimatedDipoleLocation = userInputKernelSize/2.355; % this calculates sigma in Gaussian equation
    projectionParameter = pr.projectionParameter(standardDeviationOfEstimatedDipoleLocation);
    projectionParameter.numberOfStandardDeviationsToTruncatedGaussaian = 10*standardDeviationOfEstimatedDipoleLocation;
    [~,totalDipoleDenisty,gaussianWeightMatrix]= pr.meanProjection.getProjectionMatrix(uniqueDipole, headGrid, projectionParameter);
    %projection matrix (output 1) normalizes in each region, which is not
    %necessary for ROI analysis (yet)

    % Define anatomical ROIs
    if strcmp(atlas2use,'brodmann')
        roiLabels = num2cell(1:52);
    elseif strcmp(atlas2use,'none')
        roiLabels = num2cell(1:size(gaussianWeightMatrix,2));
        xCents = headGrid.xCube(headGrid.insideBrainCube(:)==1);
        yCents = headGrid.yCube(headGrid.insideBrainCube(:)==1);
        zCents = headGrid.zCube(headGrid.insideBrainCube(:)==1);
    else
        roiLabels = pr.regionOfInterestFromAnatomy.getAllAnatomicalLabels('atlas',atlas2use,'onlyEEGSources',false);
    end

    % Obtain dipole density and centroids in each ROI
    numberOfRegionsOfInterest = length(roiLabels);
    dipoleProbabilityInRegion = zeros(uniqueDipole.numberOfDipoles, numberOfRegionsOfInterest);

    roiCentroids = zeros(numberOfRegionsOfInterest,3);
    for i=1:numberOfRegionsOfInterest
        disp([num2str((i/numberOfRegionsOfInterest)*100) '%'])
        if strcmp(atlas2use,'none')
            dipoleProbabilityInRegion(:,i) = gaussianWeightMatrix(:,i);
            roiCentroids(i,:) = [xCents(i) yCents(i) zCents(i)];
        else
            regionOfInterest(i) = pr.regionOfInterestFromAnatomy(pr.headGrid, roiLabels{i},'atlas', atlas2use); %, 'partialMatch', 1); %roiLabels{i});
            %To use Brodmann Areas, just use insideBrainGridLocationBrodmannAreaCount(:,#) in brodmannAreaAtlas.m, where # is BA # 
            dipoleProbabilityInRegion(:,i) = gaussianWeightMatrix * regionOfInterest(i).membershipProbabilityCube(headGrid.insideBrainCube);
            %gaussianWeightMatrix normalizes IC across voxels

            %Compute centroids of ROIs
            xCube = regionOfInterest(i).headGrid.xCube;
            yCube = regionOfInterest(i).headGrid.yCube;
            zCube = regionOfInterest(i).headGrid.zCube;
            membershipCube = regionOfInterest(i).membershipCube;
            tmp_x = xCube(membershipCube);
            tmp_y = yCube(membershipCube);
            tmp_z = zCube(membershipCube);
            if strcmp(hemisphere,'L')
                xCentroid = mean(tmp_x(tmp_x<0)); %xCube(membershipCube));
                yCentroid = mean(tmp_y(tmp_x<0)); %yCube(membershipCube));
                zCentroid = mean(tmp_z(tmp_x<0)); %zCube(membershipCube));
            else
                xCentroid = mean(tmp_x(tmp_x>0)); %xCube(membershipCube));
                yCentroid = mean(tmp_y(tmp_x>0)); %yCube(membershipCube));
                zCentroid = mean(tmp_z(tmp_x>0)); %zCube(membershipCube));
            end
            roiCentroids(i,:) = [xCentroid yCentroid zCentroid];
        end
    end
    roiCentroids(isnan(roiCentroids))=0;
    
    dipoleProbabilityInRegion(isnan(dipoleProbabilityInRegion)) = 0;
    
    %Normalize across ROI's (necessary for scaling)
    normdipoleProbabilityInRegion=[];
    for j=1:size(dipoleProbabilityInRegion,2)
        normdipoleProbabilityInRegion(:,j)=dipoleProbabilityInRegion(:,j)/sum(dipoleProbabilityInRegion(:,j));
    end
    normdipoleProbabilityInRegion(isnan(normdipoleProbabilityInRegion))=0;

    %Outputs
    dipoleDensityROI=sum(dipoleProbabilityInRegion); %Un-normalized measure
    normdipoleProbabilitiesROI=normdipoleProbabilityInRegion;
    addpath(getenv('csvwrite_with_headers_path'));

    %Export outputs to csv (dipoleDensity is first row, rest are normalized
    %values)
    if strcmp(atlas2use,'brodmann') || strcmp(atlas2use,'none')
        roiLabels_tmp = cellfun(@num2str,roiLabels,'un',0); %Convert labels to strings
    end
    if save_proj_matrix == 1
        if strcmp(atlas2use,'brodmann') || strcmp(atlas2use,'none')
            csvwrite_with_headers([save_path atlas2use '_' subj_id '_elecs2ROI.csv'],[dipoleDensityROI; normdipoleProbabilitiesROI],roiLabels_tmp);
        else
            csvwrite_with_headers([save_path atlas2use '_' subj_id '_elecs2ROI.csv'],[dipoleDensityROI; normdipoleProbabilitiesROI],roiLabels);
        end
        csvwrite_with_headers([save_path atlas2use '_' subj_id '_ROIcentroids_' hemisphere 'side.csv'],roiCentroids,{'x','y','z'});
    end
end
%%
%% Plot ROI's
if ~strcmp(atlas2use,'none')
    if strcmp(atlas2use,'brodmann')
        allROI_spacing = 0.6;
    elseif strcmp(atlas2use,'loni')
        allROI_spacing = 0.5;
    elseif strcmp(atlas2use,'aal')
        allROI_spacing = 0.5;
    end

    if ~exist('customColors')
        customColors = rand(length(regionOfInterest),3);
        
        if ~isempty(rois_used)
%             customColors_orig = customColors;
            customColors = 0.5*ones(length(regionOfInterest),3);
            for i=1:length(rois_used)
                if strcmp(atlas2use,'brodmann')
                    roi_ind_curr = find(strcmpi(roiLabels_tmp,rois_used{i}));
                else
                    roi_ind_curr = find(strcmpi(roiLabels,rois_used{i}));
                end
                customColors(roi_ind_curr,:) = customColors_orig(i,:);
            end
        end
    end
    if plotAllRegionsTogether==1
        figure;
        set(gcf, 'numberTitle', 'off', 'name', ['Anatomical ROIs used: ' atlas2use])
        plot_dipplot_with_cortex;
        for n = 1:length(regionOfInterest)
            pr.plot_head_surface(regionOfInterest(n).headGrid,...
                                 regionOfInterest(n).membershipCube,...
                                 'showProjectedOnMrs', 0,...
                                 'mainLightType', 'left',...
                                 'surfaceColor', customColors(n,:),...
                                 'isosurfaceDistance', headGrid.spacing * allROI_spacing,...
                                 'surfaceOptions', {'facealpha', 1});
        end
        delete(findall(gcf, 'Tag', 'img'))
        view(-90,0); camlight('headlight');%0,90); %
        I = findobj(gcf,'Type','Light');
        for i=1:length(I)
            I(i).Visible='off';
        end
        camlight('left'); material dull;
%         camlight(-90,0); camlight(-90,0); %camlight(-90,0)
        I = findobj(gcf,'Type','Light');
        I(1).Position(2) = 200; I(1).Color = [.9,.9,.9];
        if save_plots==1
            saveas(gcf,[save_path 'allROI_locations.fig']);
            saveas(gcf,[save_path 'allROI_locations.jpg'])
        end
    end

    if plotAllRegionsTogether==0
        if strcmp(atlas2use,'brodmann')
            roi_ind = find(strcmpi(roiLabels_tmp,keyword_roi));
        else
            roi_ind = find(strcmpi(roiLabels,keyword_roi));
        end
        for n=roi_ind %1:length(regionOfInterest)
            figure;
            if strcmp(atlas2use,'brodmann')
                set(gcf, 'numberTitle', 'off', 'name', num2str(regionOfInterest(n).keyword))
            else
                set(gcf, 'numberTitle', 'off', 'name', regionOfInterest(n).keyword)
            end
            plot_dipplot_with_cortex;
            pr.plot_head_surface(regionOfInterest(n).headGrid,...
                regionOfInterest(n).membershipCube,...
                'showProjectedOnMrs', 0,... 1,...
                'surfaceColor', customColors(n,:),...
                'mainLightType', 'left',...
                'surfaceOptions', {'facealpha', 1});
            delete(findall(gcf, 'Tag', 'img'))
            I = findobj(gcf,'Type','Light');
            for i=1:length(I)
                I(i).Visible='off';
            end
            camlight('left'); material dull;
    %         camlight(-90,0); camlight(-90,0); %camlight(-90,0);
            I = findobj(gcf,'Type','Light');
            I(1).Position(2) = 200; I(1).Color = [.9,.9,.9];
            view(-90,0);
            if save_plots==1
                saveas(gcf,[save_path regionOfInterest(n).keyword '_ROI_location.fig']);
                saveas(gcf,[save_path regionOfInterest(n).keyword '_ROI_location.jpg'])
            end
        end
    end
    set(gcf,'color','w');
%     addpath('/Users/stepeter/Documents/BruntonLab/eeglab13_5_4b/plugins/SIFT1.41/utils/export_fig/')
%     view(0,90)
%     export_fig('/Users/stepeter/Documents/BruntonLab/Manuscripts/figs/axial.png')
%     view(0,0)
%     export_fig('/Users/stepeter/Documents/BruntonLab/Manuscripts/figs/coronal.png')
%     view(-90,0)
%     export_fig('/Users/stepeter/Documents/BruntonLab/Manuscripts/figs/sagittal.png')
    disp('ROI''s plotted!');
end