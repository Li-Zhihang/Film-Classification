clear
clc

partnum = 6;
pdata = [];

dname = '.\lp_info\';
dlist = dir(dname);
for k = 3:length(dlist)
    flist = dir([dname, dlist(k).name]);
    lpdata = read_lpmat([dname, dlist(k).name, '\'], partnum);
    pdata = [pdata; lpdata];
end
L = [2*ones(1,partnum),2*ones(1,partnum),1*ones(1,partnum),5*ones(1,partnum),5*ones(1,partnum),1*ones(1,partnum),3*ones(1,partnum),2*ones(1,partnum),1*ones(1,partnum),4*ones(1,partnum),5*ones(1,partnum)];
Y = tsne(pdata,'Algorithm','barneshut','NumPCAComponents',15,'NumDimensions',2);
figure
gscatter(Y(:,1),Y(:,2),L)
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,L,'filled')