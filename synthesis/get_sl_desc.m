function sldesc = get_sl_desc(intervals)
%% do the counting
edges = [0, 2, 4.5, 7, 10, 22.5, 40];  % edges in sec
edges = [edges, Inf];
% figure
h = histogram(intervals, edges, 'Normalization', 'probability');
h_data = h.Values;
% xlim([0, 80])
%% count transaction
inter_block = 7 * ones(length(intervals), 1);
transmat = zeros(7);
a1 = intervals < edges(2);
a2 = intervals < edges(3);
a3 = intervals < edges(4);
a4 = intervals < edges(5);
a5 = intervals < edges(6);
a6 = intervals < edges(7);
inter_block(a6) = 6;
inter_block(a5) = 5;
inter_block(a4) = 4;
inter_block(a3) = 3;
inter_block(a2) = 2;
inter_block(a1) = 1;
for k = 1:length(intervals) - 1
    transmat(inter_block(k), inter_block(k + 1)) = transmat(inter_block(k), inter_block(k + 1)) + 1;
end
% figure
% edges = {1:7 1:7};
% hist3([inter_block(1:end-1), inter_block(2:end)], 'Edges', edges, 'CdataMode','auto')
% colorbar
% view(2)
transmat = reshape(transmat, 1, 49);
sldesc = [transmat, mean(intervals), std(intervals), skewness(intervals), kurtosis(intervals)];
close all