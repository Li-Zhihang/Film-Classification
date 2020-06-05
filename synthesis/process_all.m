clear
clc

partnum = 5;
tsdata = [];
scdata = [];
lpdata = [];
cdata = [];
sldata = [];

dname = '.\tss_info\';
dlist = dir(dname);
for k = 3:length(dlist)
    [pd1, pd2] = read_tssmat([dname, dlist(k).name, '\'], partnum);
    tsdata = [tsdata; pd1];
    scdata = [scdata; pd2];
end

dname = '.\lp_info\';
dlist = dir(dname);
for k = 3:length(dlist)
    flist = dir([dname, dlist(k).name]);
    lpd = read_lpmat([dname, dlist(k).name, '\'], partnum);
    lpdata = [lpdata; lpd];
end

dname = '.\color_info\';
dlist = dir(dname);
for k = 3:length(dlist)
    flist = dir([dname, dlist(k).name]);
    cd = read_colormat([dname, dlist(k).name, '\'], partnum);
    cdata = [cdata; cd];
end

dname = '.\indexfile\';
flist = dir(dname);
for k = 3:length(flist)
    sld = read_index([dname, flist(k).name], partnum);
    sldata = [sldata; sld];
end

L = [4*ones(1,partnum),2*ones(1,partnum),4*ones(1,partnum),2*ones(1,partnum),1*ones(1,partnum),5*ones(1,partnum),5*ones(1,partnum),6*ones(1,partnum),1*ones(1,partnum),4*ones(1,partnum),5*ones(1,partnum),2*ones(1,partnum),3*ones(1,partnum),6*ones(1,partnum),1*ones(1,partnum),3*ones(1,partnum),2*ones(1,partnum),1*ones(1,partnum),3*ones(1,partnum),4*ones(1,partnum),3*ones(1,partnum),5*ones(1,partnum)];
shot_data = [tsdata, scdata, lpdata, sldata, cdata];

save('film_data.mat', '-v7.3', 'tsdata', 'scdata', 'lpdata', 'sldata', 'cdata', 'L')