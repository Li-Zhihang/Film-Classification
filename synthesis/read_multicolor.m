clear
clc

dname = '.\outputs\moonrise\';
outdir = '.\color_info\';
mkdir(outdir)
flist = dir(dname);
for k = 3: length(flist)
    process_color(dname, flist(k).name, outdir)
end


