clear
clc

dname = '.\color_info\bladerunner\';
flist = dir(dname);
cinfo = [];
for k = 3: length(flist)
    fname = [dname, flist(k).name];
    load(fname)
    cinfo = [cinfo; info];
end

figure
edges = {0:0.01:1-0.01 0:0.01:1-0.01};
hist3([cinfo(:, 1), cinfo(:, 3)], 'Edges', edges, 'CdataMode','auto');
view(2)


figure
edges = 0:0.02:0.3-0.02;
subplot(3, 2, 1)
histogram(cinfo(:, 2), edges, 'Normalization', 'probability')
title('std-H')
xlim([0., 0.3])
ylim([0., 0.8])
subplot(3, 2, 2)
histogram(cinfo(:, 4), edges, 'Normalization', 'probability')
title('std-S')
xlim([0., 0.3])
ylim([0., 0.8])

subplot(3, 2, 3)
histogram(abs(cinfo(:, 5) - cinfo(:, 1)), edges, 'Normalization', 'probability')
title('center-peak diff-H')
xlim([0., 0.3])
ylim([0., 0.8])
subplot(3, 2, 4)
histogram(abs(cinfo(:, 6) - cinfo(:, 3)), edges, 'Normalization', 'probability')
title('center-peak diff-S')
xlim([0., 0.3])
ylim([0., 0.8])

edges = 0:0.05:1-0.05;
subplot(3, 2, 5)
histogram(cinfo(:, 7), edges, 'Normalization', 'probability')
title('peak in cluster')
xlim([0., 1.])
ylim([0., 0.8])
subplot(3, 2, 6)
histogram(cinfo(:, 8), edges, 'Normalization', 'probability')
title('cluster in whole')
xlim([0., 1.])
ylim([0., 0.8])