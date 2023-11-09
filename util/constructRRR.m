function [R]=constructRRR(fea,multiorder)

%% LPP Paras
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 1;

[~,v]=size(fea);
[n,~]=size(fea{1});

A=cell(1,v);
R=cell(1,v);
for num=1:v
    A{num}=constructW(fea{num},options);
    sum=zeros(n);
    for num2=1:multiorder
        sum=sum+A{num}^num2;
    end
    sum=sum-diag(diag(sum));
    R{num}=mapminmax(sum,0,1);
end

end