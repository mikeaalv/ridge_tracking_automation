% contruct the training data from ridge tracking result
% for RNN training
% input: ridge point bool vector, local maximum bool vector, intensity vector
% output: relative index for the next tracking point
addpath(genpath('/Users/yuewu/Documents/GitHub/ridge_tracking_automation/'));
close all;
clear all;
comp='/Users/yuewu/';%the computer user location
datadir=[comp 'Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/spectral.related/ridge.net/result_reprod/'];
workdir=[comp 'Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/spectral.related/rid_track_auto/data/tracking_raw/'];
cd(workdir);
load([datadir 'data/unshared/sampleData.mat']);
load([datadir 'result_data/tracing.newmeth.experiment.manual.completerid.mat']);
samples=[1,2,3,7,8,9];
% feature parameter
wind_half_feature=20;%window size
%each row one time point and each column one feature. different samples will be separated by nan
input_mat=[];
output_mat=[];
for sample_i=1:length(samples)
  sample=samples(sample_i);
  input_mat_temp=[];
  output_mat_temp=[];
  %
  specdata=sampleData(sample);
  mat=specdata.Xcollapsed_1h1d;
  ppm=specdata.ppm_1h1d;
  % regions
  ridge_data=Sample_complete_rid(sample_i).ridges;
  ridge_data=ridge_data(2:length(ridge_data));
  nridges=length(ridge_data);
  regions=[];
  for ridi=1:nridges
    loc_region=ridge_data(ridi).parameters.region;
    regions=[regions; loc_region];
  end
  regions=unique(regions,'rows');
  nregion=size(regions,1);
  for regioni=1:nregion
    regioni
    input_mat_reg=[];
    output_mat_reg=[];
    region_loc=regions(regioni,:);
    range_loc=sort(matchPPMs(region_loc,ppm));
    ind_loc=range_loc(1):range_loc(2);
    submat=mat(:,ind_loc);
    subppm=ppm(ind_loc);
    regionsize=length(subppm);
    % classify ridge point matrix
    ridmat=rid_point_pred(submat);
    % classify local maximum point matrix
    maxmat=zeros(size(submat));
    for i = 1:size(submat,1)
      tempmat=submat(i,:);
      maxmat(i,:)=islocalmax(tempmat);
    end
    for ridi=1:nridges
      rid_region=ridge_data(ridi).parameters.region;
      if ~(rid_region(1)==region_loc(1) & rid_region(2)==region_loc(2))
        continue;
      end
      rid_info=ridge_data(ridi).result;
      ppm_rid=rid_info.ppm;
      row_ind=rid_info.rowind;
      ntime=length(ppm_rid);
      ridind_glob=[];
      for timei=1:ntime
        ppm_ele=ppm_rid(timei);
        ridind=matchPPMs(ppm_ele,ppm);
        % output
        if timei==1
          ind_rela=0;%the first index always default 0
        else
          ind_rela=ridind-ridind_glob(length(ridind_glob));
        end
        ridind_glob=[ridind_glob ridind];
        % input
        %% calculte index correspond to the global matrix
        window_glob=[ridind-wind_half_feature ridind+wind_half_feature];
        glob_ind=window_glob(1):window_glob(2);
        %
        inten_vec=mat(row_ind(timei),glob_ind);
        % locally normlize the intensity
        inten_vec=(inten_vec-mean(inten_vec))/std(inten_vec);
        % formulate ridge and local maximum vector based on local matrix
        % use zeros to supplement boundaries
        ridind_local=matchPPMs(ppm_ele,subppm);
        window_local=[ridind_local-wind_half_feature ridind_local+wind_half_feature];
        window_local_reform=window_local;
        % two boundaries can both be met
        locbound=[0 0];%the zero filling region for both sides
        if window_local_reform(1)<1
          window_local_reform(1)=1;
          locbound(1)=1-window_local(1);
        end
        if window_local_reform(2)>regionsize
          window_local_reform(2)=regionsize;
          locbound(2)=window_local(2)-regionsize;
        end
        loc_ind=window_local_reform(1):window_local_reform(2);
        rid_vec=[zeros([1,locbound(1)]) ridmat(row_ind(timei),loc_ind) zeros([1,locbound(2)])];
        locmax_vec=[zeros([1,locbound(1)]) maxmat(row_ind(timei),loc_ind) zeros([1,locbound(2)])];
        %
        featurevec=[inten_vec rid_vec locmax_vec];
        input_mat_reg=[input_mat_reg; featurevec];
        output_mat_reg=[output_mat_reg; ind_rela];
      end
      input_mat_reg=[input_mat_reg; nan([1,size(input_mat_reg,2)])];
      output_mat_reg=[output_mat_reg; nan([1,1])];
    end
    input_mat_temp=[input_mat_temp; input_mat_reg];
    output_mat_temp=[output_mat_temp; output_mat_reg];
  end
  input_mat=[input_mat; input_mat_temp];
  output_mat=[output_mat; output_mat_temp];
end
datatab=array2table([input_mat output_mat]);
save('training_data.mat','input_mat','output_mat','datatab');
writetable(datatab,['training_data.txt'],'Delimiter','\t');
