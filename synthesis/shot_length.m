clear
clc
close all
%% parameters
dname = '.\indexfile\';
fname = 'Arrival';
savedir = '.\index_output\';
scriptname = [dname, fname, '.txt'];

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

partnum = 8;
partlen = floor(length(intervals) / partnum);
for p = 1:partnum
    getScaleDesc(intervals(1+(p-1)*partlen:p*partlen), [savedir, fname, '_',num2str(p),'.mat'])
end
