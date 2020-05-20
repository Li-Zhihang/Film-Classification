clear
clc

partnum = 6;
pdata = [];
pdata2 = [];

dname = '.\tss_info\';
dlist = dir(dname);
for k = 3:length(dlist)
    flist = dir([dname, dlist(k).name]);
    tinfo = [];
    for p = 3:length(flist)
        fname = [dname, dlist(k).name, '\', flist(p).name];
        load(fname)
        tinfo = [tinfo; info];
    end

    partlen = floor(size(tinfo, 1) / partnum);
    for q = 1:partnum
        pinfo = tinfo((q-1)*partlen + 1:q*partlen, :);
   
        e = 0:1:8;
        to_m1t = histogram(pinfo(:, 1), e, 'Normalization', 'probability');
        to_m1t = to_m1t.Values;
        to_m2t = histogram(pinfo(:, 2), e, 'Normalization', 'probability');
        to_m2t = to_m2t.Values;
   
        e = -2:1:6;
        sc_m1t = histogram(pinfo(:, 7), e, 'Normalization', 'probability');
        sc_m1t = sc_m1t.Values;
        sc_m2t = histogram(pinfo(:, 8), e, 'Normalization', 'probability');
        sc_m2t = sc_m2t.Values;
        
        e = 0:1:3;
        sat_mt = histogram(pinfo(:, 5), e, 'Normalization', 'probability');
        sat_mt = sat_mt.Values;
        
        e = 0:0.1:1;
        sc_m1v = histogram(pinfo(:, 9), e, 'Normalization', 'probability');
        sc_m1v = sc_m1v.Values;
        sc_m2v = histogram(pinfo(:, 10), e, 'Normalization', 'probability');
        sc_m2v = sc_m2v.Values;
        to_m1v = histogram(pinfo(:, 3), e, 'Normalization', 'probability');
        to_m1v = to_m1v.Values;
        to_m2v = histogram(pinfo(:, 4), e, 'Normalization', 'probability');
        to_m2v = to_m2v.Values;
        sa_mv = histogram(pinfo(:, 6), e, 'Normalization', 'probability');
        sa_mv = sa_mv.Values;
        
        close all
        pdata = [pdata; to_m1t, to_m2t, to_m1v, to_m2v, sat_mt, sa_mv];
        pdata2 = [pdata2; sc_m1t, sc_m2t, sc_m1v, sc_m2v];
    end
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
