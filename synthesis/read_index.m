function sldata = read_index(fname, partnum)
%% read script
scriptf = fopen(fname, 'r');
fgetl(scriptf);
fps = str2double(fgetl(scriptf));
location = fscanf(scriptf, '%d\n');
fclose(scriptf);
%% get shot length
start_list = location(1:end-1);
stop_list = location(2:end);
intervals = stop_list - start_list;
intervals = intervals / fps;  % convert to sec
intervals = intervals(intervals>0.8);  % eliminate small shot (might be gradual change)

sldata = [];
partlen = floor(length(intervals) / partnum);
for p = 1:partnum
    desc = get_sl_desc(intervals(1+(p-1)*partlen:p*partlen));
    sldata = [sldata; desc];
end
end