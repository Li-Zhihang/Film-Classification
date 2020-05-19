clear
clc

dname = '.\index_output\';
partnum = 8;

flist = dir(dname);
scale_data = zeros(length(flist) - 2, 56);
labels = zeros(1, length(flist) - 2);
for k = 3: length(flist)
    fname = [dname, flist(k).name];
    load(fname)
    flat_trans = reshape(transmat, 1, 49);
    scale_data(k - 2, :) = [h_data, flat_trans];
end

L = [2*ones(1,partnum),2*ones(1,partnum),5*ones(1, partnum),4*ones(1, partnum),ones(1, partnum),5*ones(1, partnum),6*ones(1, partnum),ones(1, partnum),3*ones(1, partnum),2*ones(1, partnum),ones(1, partnum),3*ones(1, partnum),4*ones(1, partnum),3*ones(1, partnum),5*ones(1, partnum)];
Y = tsne(scale_data,'Algorithm','barneshut','NumPCAComponents',30,'NumDimensions',3,'Algorithm','exact','Exaggeration',10);
figure
% gscatter(Y(:,1),Y(:,2),L)
scatter3(Y(:,1),Y(:,2),Y(:,3),15,L,'filled')