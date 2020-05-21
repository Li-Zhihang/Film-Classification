function cdata = read_colormat(dname, partnum)

flist = dir(dname);

partlen = floor((length(flist) - 2) / partnum);
cdata = [];
for p = 1: partnum
    cin = [];
    for k = 3 + (p-1) * partlen: 2 + p * partlen
        fname = [dname, flist(k).name];
        load(fname)
        cin = [cin; cinfo];
    end
    
    m_h = mean(cin(:,1));
    std_h = std(cin(:,1));
    m_s = mean(cin(:,3));
    std_s = std(cin(:,3));
    ske_h = skewness(cin(:,1));
    ske_s = skewness(cin(:,3));
    kur_h = kurtosis(cin(:,1));
    kur_s = kurtosis(cin(:,3));
    
    cdata = [cdata; m_h, std_h, m_s, std_s, ske_h, ske_s, kur_h, kur_s];
end
end

