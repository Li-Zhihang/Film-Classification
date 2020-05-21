clear
clc

partnum = 5;
pdata1 = [];
pdata2 = [];

dname = '.\tss_info\';
dlist = dir(dname);
for k = 3:length(dlist)
    [pd1, pd2] = read_tssmat([dname, dlist(k).name, '\'], partnum);
    pdata1 = [pdata1; pd1];
    pdata2 = [pdata2; pd2];
end

L = [2*ones(1,partnum),2*ones(1,partnum),1*ones(1,partnum),5*ones(1,partnum),5*ones(1,partnum),1*ones(1,partnum),3*ones(1,partnum),2*ones(1,partnum),1*ones(1,partnum),4*ones(1,partnum),5*ones(1,partnum)];
Y = tsne(pdata,'Algorithm','barneshut','NumPCAComponents',30,'NumDimensions',2);
figure
gscatter(Y(:,1),Y(:,2),L)
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,L,'filled')

Y2 = tsne(pdata2,'Algorithm','barneshut','NumPCAComponents',20,'NumDimensions',2);
figure
gscatter(Y2(:,1),Y2(:,2),L)
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,L,'filled')
