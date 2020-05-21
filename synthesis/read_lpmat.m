function lpdata = read_lpmat(dname, partnum)

flist = dir(dname);
lpinfo = [];
lpdata = [];

for p = 3:length(flist)
    fname = [dname, flist(p).name];
    load(fname);
    lpinfo = [lpinfo; pose_typ, pose_val];
end

partlen = floor(size(lpinfo, 1) / partnum);
for q = 1:partnum
    pinfo = lpinfo((q-1)*partlen + 1:q*partlen, :);

    edges = -1:1:6;
    pose_m1t = histogram(pinfo(:, 1), edges, 'Normalization', 'probability');
    pose_m1t = pose_m1t.Values;
    pose_m2t = histogram(pinfo(:, 2), edges, 'Normalization', 'probability');
    pose_m2t = pose_m2t.Values;

    edges = 0:0.1:1;
    pose_m1s = histogram(pinfo(:, 3), edges, 'Normalization', 'probability');
    pose_m1s = pose_m1s.Values;
    pose_m2s = histogram(pinfo(:, 4), edges, 'Normalization', 'probability');
    pose_m2s = pose_m2s.Values;

    close all
    lpdata = [lpdata; pose_m1t, pose_m1s, pose_m2t, pose_m2s];
end
end

