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
samples=[1,2,3,7,8,9];
for samp_i=1:length(Sample_complete_rid)
  sample_rep=Sample_complete_rid(samp_i);
  nridges=size(sample_rep.ridges,2)-1;
  data=sampleData(samples(samp_i));
  mat=data.Xcollapsed_1h1d;
  ppm=data.ppm_1h1d;
  tempregion=nan();%the targetted region
  % all regions
  regions=[];
  for ridi=1:length(sample_rep)
    loc_region=sort(sample_rep(ridi).parameters.region);
    regions=[regions; loc_region];
  end
  regions=unique(regions,'rows');
  regions=sortrows(regions,[1,2]);
  % combine overlapped regions
  regions_comb=[];
  for regi=1:size(regions,1)
    newreg=regions(regi,:);
    if length(regions_comb)==0
      regions_comb=[regions_comb; newreg];
    else
      lastreg=regions_comb(end,:);
      if lastreg(2)<=newreg(1)
        regions_comb=[regions_comb; newreg];
      else
        regions_comb(end,:)=[lastreg(1) max(lastreg(2),newreg(2))];
      end
    end
  end
  % store image for each region
  filenames_range={};
  for regioni=1:size(regions_comb,1)
    newppm_range=regions_comb(regioni,:);
    region_ind=sort(matchPPMs(newppm_range,ppm));
    locamat=mat(:,region_ind(1):region_ind(2));
    locamat=locamat-min(locamat(:));
    locamat=locamat/max(locamat(:))*255;
    % fig=figure();
    % imshow(uint8(locamat));
    % fig=figure();
    % surf(locamat);
    imagename=['image_',num2str(samples(samp_i)),'_',num2str(newppm_range(1)),'_',num2str(newppm_range(2)),'.',fileext];
    imwrite(uint8(locamat),[imagename],fileext);
    filenames_range=[filenames_range imagename];
    % close(fig);
  end
  for i=1:nridges
    locpara=sample_rep.ridges(i+1).parameters;
    locstr=sample_rep.ridges(i+1).result;
    % local region
    newppm_range_loc=sort(locpara.region);
    regmatchind=find(newppm_range_loc(1)<=regions_comb(:,1)&newppm_range_loc(2)<=regions_comb(:,2));
    newppm_range=regions_comb(regmatchind,:);
    region_ind=sort(matchPPMs(newppm_range,ppm));
    % ridge index
    rowind=locstr.rowind;
    [rowind,reord_ind]=sort(rowind);
    ppmloc=locstr.ppm;
    ppmloc=ppmloc(reord_ind);
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
    filename=[filename filenames_range(regmatchind)];
    region_attributes=[region_attributes region_attributes_ele];
  end
end
region_name=repmat('ridge',[length(filename),1]);
labeletab=table(filename',region_attributes',region_name,'VariableNames',{'filename' 'region_attributes' 'region_name'});
writetable(labeletab,['labels.txt'],'Delimiter','\t');

% plot checking of the result
% surface image with peak ridge
tab_seg=readtable('labels.txt','Delimiter','\t');
rndinds=randsample(1:size(tab_seg,1),10);
for rndind=rndinds
  record=tab_seg(rndind,:);
  immat=imread(record{:,'filename'});
  js_str=jsondecode(record{:,'region_attributes'}{1});
  xvec=js_str.all_points_x;
  yvec=js_str.all_points_y;
  linind=sub2ind(size(immat),yvec,xvec);
  fig=figure(), hold on
      surf(immat,'FaceColor','Interp');
      ylabel('y')
      zlabel('z')
      title(['example'])
      xlabel('x')
      scatter3(xvec,yvec,immat(linind),'r','linewidth',3);
  saveas(fig,['../test/' num2str(rndind) '.fig']);
  close(fig);
