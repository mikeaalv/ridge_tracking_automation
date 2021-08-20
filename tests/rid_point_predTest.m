function tests=rid_point_predTest
  tests=functiontests(localfunctions);
end

function testsimplecases(testCase)
  para=[1 1 -4;%H=-3 K=-17 saddle ridge
      1 1 -1;%H=0 K=-5 minimal surface
      2 6 2;%H=4 K=-20 saddle valley
      -1 2 -1;% H=-2 K=0 ridge surface
      0 0 0;% H=0 K=0 flat surface
      3 6 3;% H=6 K=0 valley surface
      -1 1 -1;% H=-2 K=3 peak surface
      1 1 1;% H=2 K=3 pit surface
      ];%%%A,B,C no condition on K>0 and H=0
  resexp=[-3 -17; 0 -5; 4 -20; -2 0; 0 0; 6 0; -2 3; 2 3];
  [X,Y]=meshgrid(-50:1:50,-50:1:50);
  thresh=0.01;
  ind=51;%%where the H and K is calculated
  flagH=true;
  flagK=true;
  for i=1:size(para,1)
    Apara=para(i,1);
    Bpara=para(i,2);
    Cpara=para(i,3);
    Z=Apara*X.^2 + Bpara*X.*Y + Cpara*Y.^2;
    [ridpoint Hmat Kmat]=rid_point_pred(Z);
    flagH=flagH&(Hmat(ind,ind)-resexp(i,1)<thresh);
    flagK=flagK&(Kmat(ind,ind)-resexp(i,2)<thresh);
  end
  verifyTrue(testCase,flagH&flagK);
end
