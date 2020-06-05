clear
clc

dname = '.\outputs\corpoceleste\';
outdir = '.\color_info\corpoceleste\';
mkdir(outdir)
flist = dir(dname);
for k = 3: length(flist)
    get_shot_color(dname, flist(k).name, outdir)
%     get_shot_tss(dname, flist(k).name, outdir)
%     get_shot_lp(dname, flist(k).name, outdir)
end


