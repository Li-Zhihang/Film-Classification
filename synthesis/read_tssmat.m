clear
clc

dname = '.\tss_info\tennenbaums\';
flist = dir(dname);
tinfo = [];
for k = 3: length(flist)
    fname = [dname, flist(k).name];
    load(fname)
    tinfo = [tinfo; info];
end

%% plot
tone_max_t = tinfo(:, 1);
tone_max_v = tinfo(:, 3);
sat_t = tinfo(:, 5);
sat_v = tinfo(:, 6);
scale_max_t = tinfo(:, 7);
scale_max_v = tinfo(:, 9);

figure
subplot(3, 2, 1)
e = -2:1:6;
histogram(scale_max_t, e, 'Normalization', 'probability')
title('scale type')
xlim([-2, 7])
ylim([0, 1])

subplot(3, 2, 2)
e = 0:0.1:1;
histogram(scale_max_v, e, 'Normalization', 'probability')
title('scale score')
xlim([0, 1])
ylim([0, 1])

subplot(3, 2, 3)
e = 0:1:8;
histogram(tone_max_t, e, 'Normalization', 'probability')
title('tone type')
xlim([0, 9])
ylim([0, 1])

subplot(3, 2, 4)
e = 0:0.1:1;
histogram(tone_max_v, e, 'Normalization', 'probability')
title('tone score')
xlim([0, 1])
ylim([0, 1])

subplot(3, 2, 5)
e = 0:1:3;
histogram(sat_t, e, 'Normalization', 'probability');
title('sat type')
xlim([0, 4])
ylim([0, 1])

subplot(3, 2, 6)
e = 0:0.1:1;
histogram(sat_v, e, 'Normalization', 'probability');
title('sat score')
xlim([0, 1])
ylim([0, 1])