function [ridpoint Hmat Kmat]=rid_point_pred(submat)
% function: Calcuate the ridge point based on the intensity matrix
%
% Argument:
%     submat: numeric matrix. the intensity matrix. size(submat)=[time ppm]. must be provided.
% Return:
%     ridpoint: the ridge point matrix
%     Hmat: the H matrix
%     Kmat: the K matrix
% Examples:
%     refer to the example in rid_point_predTest.ms
% Test:
% results = runtests('rid_point_predTest.m')
%
% Yue Wu 08/19/2021
% Tested with MATLAB R2018b

if ~exist('submat','var')
  error('please provide input intensity matrix');
end
% default parameters
windsize=7;
threhold_H=0;
threhold_K=1;
%
halfwidplus=(windsize+1)/2;
halfwidminus=(windsize-1)/2;
sizes=size(submat);
rown=sizes(1);
coln=sizes(2);
matuse=[repmat(submat(1,:),(windsize-1)/2,1); submat; repmat(submat(end,:),(windsize-1)/2,1)];
%% construct the othogonal basis
x=1:windsize;
x=x-halfwidminus-1;
miu=[];
miu(1)=windsize;%miu0
miu(2)=0;
miu(3)=sum(x.^2);%miu2
phi=[];
phi(1,:)=repmat(1,1,windsize);
phi(2,:)=x;
phi(3,:)=x.^2-repmat(miu(3)/miu(1),1,windsize);
b=[];
for i=1:3
  b(i,:)=phi(i,:)./sum(phi(i,:).^2);
end
%% H-K matrix calculation
ridpoint=zeros(size(submat));
Hmat=zeros(size(submat));
Kmat=zeros(size(submat));
for i=1:rown
  for j=halfwidplus:(coln-halfwidminus)
    tempmat=matuse(i:(i+windsize-1),(j-halfwidminus):(j+halfwidminus));
    a=zeros(3,3);
    for pi=1:3
      for pj=1:3
        summat=(b(pi,:)'*b(pj,:))'.*tempmat;
        a(pi,pj)=sum(summat(:));
      end
    end
    %f fu fv fuu fvv fuv fvu
    % yfuncvec=[tempmat(halfwidplus,halfwidplus) a(2,1) a(1,2) a(3,1) a(1,3) a(2,2) a(2,2)];
    yfuncvec=[tempmat(halfwidplus,halfwidplus) a(2,1) a(1,2) 2*a(3,1) 2*a(1,3) a(2,2) a(2,2)];
    %% vec: x xu xv xuu xvv xuv xvu the first row not used.
    surfvecmat=[0 0 yfuncvec(1); 1 0 yfuncvec(2); 0 1 yfuncvec(3); 0 0 yfuncvec(4); 0 0 yfuncvec(5); 0 0 yfuncvec(6); 0 0 yfuncvec(7)]; %% the first row is never used in calculation
    nv=cross(surfvecmat(2,:),surfvecmat(3,:));
    nval=nv/sqrt(sum(nv.^2));
    G=[surfvecmat(2,:)*surfvecmat(2,:)'  surfvecmat(2,:)*surfvecmat(3,:)'; surfvecmat(2,:)*surfvecmat(3,:)'  surfvecmat(3,:)*surfvecmat(3,:)'];
    B=[surfvecmat(4,:)*nval' surfvecmat(6,:)*nval'; surfvecmat(6,:)*nval' surfvecmat(5,:)*nval'];
    detG=det(G);
    invG=[G(2,2) -G(1,2); -G(2,1) G(1,1)]/detG;
    H=trace(invG*B)/2;
    K=det(B)/detG;
    Ridflag=(H<threhold_H)&(abs(K)<threhold_K);
    ridpoint(i,j)=Ridflag;
    Hmat(i,j)=H;
    Kmat(i,j)=K;
  end
end
