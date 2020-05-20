clear
clc
close all
%% parameters
dname = '.\indexfile\';
partnum = 5;

sl_desc = [];
flist = dir(dname);
for k = 3: length(flist)
    fname = flist(k).name;
    scriptname = [dname, fname];
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
    intervals = intervals(intervals>0.8);  % eliminate small shot (might be gradual change)

    
    partlen = floor(length(intervals) / partnum);
    for p = 1:partnum
        desc = get_sl_desc(intervals(1+(p-1)*partlen:p*partlen));
        sl_desc = [sl_desc; desc];
    end
end

L = [4*ones(1, partnum),2*ones(1,partnum),4*ones(1, partnum),2*ones(1,partnum),5*ones(1, partnum),5*ones(1, partnum),ones(1, partnum),4*ones(1, partnum),ones(1, partnum),5*ones(1, partnum),2*ones(1,partnum),6*ones(1, partnum),ones(1, partnum),3*ones(1, partnum),3*ones(1, partnum),2*ones(1, partnum),ones(1, partnum),3*ones(1, partnum),4*ones(1, partnum),3*ones(1, partnum),5*ones(1, partnum)];
Y = tsne(sl_desc,'NumPCAComponents',30,'NumDimensions',2,'Algorithm','exact');
figure
gscatter(Y(:,1),Y(:,2),L)
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,L,'filled')
