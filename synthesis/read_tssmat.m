function [pdata1, pdata2] = read_tssmat(dname, partnum)
flist = dir(dname);
pdata1 = [];
pdata2 = [];
tinf = [];
for p = 3:length(flist)
    fname = [dname, flist(p).name];
    load(fname);
    tinf = [tinf; tinfo];
end

partlen = floor(size(tinf, 1) / partnum);
for q = 1:partnum
    pinfo = tinf((q-1)*partlen + 1:q*partlen, :);

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
    pdata1 = [pdata1; to_m1t, to_m2t, to_m1v, to_m2v, sat_mt, sa_mv];
    pdata2 = [pdata2; sc_m1t, sc_m2t, sc_m1v, sc_m2v];
%     tone_d = [mean(pinfo(:, 1)),std(pinfo(:, 1))];
%     sat_d = [mean(pinfo(:, 5)),std(pinfo(:, 5))];
%     scal_d = [mean(pinfo(:, 7)),std(pinfo(:, 7)),skewness(pinfo(:, 7)),kurtosis(pinfo(:, 7))];
%     pdata1 = [pdata1; tone_d, sat_d];
%     pdata2 = [pdata2; scal_d];
end