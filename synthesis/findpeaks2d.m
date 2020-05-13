function [pks_val, pks_index] = findpeaks2d(sig)
% find dimensions to set up loop
xdim = size(sig, 1);
ydim = size(sig, 2);

% loop through x dimension to find peaks of each row
xpeaks = zeros(xdim, ydim);
for k = 1:xdim
    [~, locs] = findpeaks(sig(k, :));
    xpeaks(k, locs) = 1;
end

ypeaks = zeros(xdim, ydim);
for k = 1:ydim
    [~, locs] = findpeaks(sig(:, k));
    ypeaks(locs, k) = 1;
end

peaks_mask = xpeaks & ypeaks;
pks = sig(peaks_mask);
pks_index = [];
for p = 1:ydim
    for q = 1:xdim
        if peaks_mask(q, p) == 1
            pks_index = [pks_index; [q, p]];
        end
    end
end

% change to descending order
[pks_val, order] = sort(pks','descend');
pks_index = pks_index(order', :);

end