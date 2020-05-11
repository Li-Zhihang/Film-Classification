clear
clc
%% parameters
scriptname = ".\indexfile\brokenantenna.txt";

%% read script
scriptf = fopen(scriptname, 'r');
videofile = fgetl(scriptf);
fps = str2double(fgetl(scriptf));
location = fscanf(scriptf, '%d\n');
fclose(scriptf);

%% get shot length
start_list = location(1:end-1);
stop_list = location(2:end);
intervals = stop_list - start_list;
intervals = intervals / fps;  % convert to sec
intervals = intervals(intervals>0.2);  % eliminate small shot (might be gradual change)

%% do the counting
% edges = [0, 2, 5, 10, 30, 60, 120, 300, 600];  % edges in sec
% edges = [edges, Inf];
% 
% histogram(intervals, edges)

%% count transaction
inter_block = 9 * ones(length(intervals), 1);
transmat = zeros(9);
a1 = intervals < 2;
a2 = intervals < 5;
a3 = intervals < 10;
a4 = intervals < 20;
a5 = intervals < 60;
a6 = intervals < 120;
a7 = intervals < 300;
a8 = intervals < 600;
inter_block(a8) = 8;
inter_block(a7) = 7;
inter_block(a6) = 6;
inter_block(a5) = 5;
inter_block(a4) = 4;
inter_block(a3) = 3;
inter_block(a2) = 2;
inter_block(a1) = 1;
for k = 1:length(intervals) - 1
    transmat(inter_block(k), inter_block(k + 1)) = transmat(inter_block(k), inter_block(k + 1)) + 1;
end
figure
edges = {1:9 1:9};
hist3([inter_block(1:end-1), inter_block(2:end)], 'Edges', edges, 'CdataMode','auto')
colorbar
view(2)
