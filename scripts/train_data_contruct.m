% contruct the training data from ridge tracking result
% matrix-> images
% labeling tables
close all;
clear all;
comp='/Users/yuewu/';%the computer user location
datadir=[comp 'Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/spectral.related/ridge.net/result_reprod/'];
workdir=[comp 'Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/spectral.related/rid_track_auto/data/raw/'];
cd(workdir);
load([datadir 'data/unshared/sampleData.mat']);
load([datadir 'result_data/tracing.newmeth.experiment.manual.completerid.mat']);
filename={};
region_attributes={};
shapename='polygon';
fileext='jpg';
for samp_i=1:length(Sample_complete_rid)
  sample_rep=Sample_complete_rid(samp_i);
  nridges=size(sample_rep.ridges,2)-1;
  data=sampleData(samp_i);
  mat=data.Xcollapsed_1h1d;
  ppm=data.ppm_1h1d;
  tempregion=nan();%the targetted region
  for i=1:nridges
    locpara=sample_rep.ridges(i+1).parameters;
    locstr=sample_rep.ridges(i+1).result;
    % local region
    newppm_range=locpara.region;
    region_ind=sort(matchPPMs(newppm_range,ppm));
    % ridge index
    rowind=locstr.rowind;
    ppmloc=locstr.ppm;
    colind_global=matchPPMs(ppmloc,ppm);
    colind_local=colind_global-region_ind(1)+1;
    % segmetation index
    rangeloc_colind=[1 region_ind(2)];
    colind_seg=[colind_local-1 colind_local+1];
    colind_seg(colind_seg<=rangeloc_colind(1))=rangeloc_colind(1);
    colind_seg(colind_seg>=rangeloc_colind(2))=rangeloc_colind(2);
    %
    all_points_x=[colind_seg(:,2)' flip(colind_seg(:,1))'];
    all_points_y=[rowind' flip(rowind)'];
    x_str=strjoin(cellstr(num2str(all_points_x')),',');
    y_str=strjoin(cellstr(num2str(all_points_y')),',');
    region_attributes_ele=['"{""name"":""' shapename '"",""all_points_x"":[' x_str '],""all_points_y"":[' y_str ']}"'];
    %
    imagename=['image_',num2str(samp_i),'_',num2str(newppm_range(1)),'_',num2str(newppm_range(2)),'.',fileext];
    filename=[filename imagename];
    region_attributes=[region_attributes region_attributes_ele];
    if isnan(tempregion) | (~(tempregion(1)==newppm_range(1)&tempregion(2)==newppm_range(2)))
      tempregion=newppm_range;
      locamat=mat(:,region_ind(1):region_ind(2));
      locamat=locamat-min(locamat(:));
      locamat=locamat/max(locamat(:))*255;
      % fig=figure();
      % imshow(uint8(locamat));
      % fig=figure();
      % surf(locamat);
      imwrite(uint8(locamat),[imagename],fileext);
      % close(fig);
    end
  end
end
region_name=repmat('ridge',[length(filename),1]);
labeletab=table(filename',region_attributes',region_name,'VariableNames',{'filename' 'region_attributes' 'region_name'});
writetable(labeletab,['labels.txt'],'Delimiter','\t');
