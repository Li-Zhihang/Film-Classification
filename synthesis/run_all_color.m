clear
clc

color_lighter = [];
color_darkest = [];

dname = '.\outputs\bladerunner\';

flist = dir(dname);
for k = 3: length(flist)
    foutput = fopen([dname, flist(k).name]);
    while ~feof(foutput)
        c = fscanf(foutput, '%d %d %d\n', [3, 5])';
        if isempty(c)
            break
        end
        t = str2double(fgetl(foutput));
        s = str2double(fgetl(foutput));
    %     sy = fscanf(foutput, '%f %f %f %f\n', [4, 1])';
        sc = str2double(fgetl(foutput));
        hf = str2double(fgetl(foutput));
        pi = fscanf(foutput, '%d %d\n', [2, 1])';
        ps = fscanf(foutput, '%f %f\n', [2, 1])';

        c_sum = sum(c, 2);
        [b, index] = sort(c_sum');
        if b(5) < 10
            continue
        end
        c = c(index, :);
        color_lighter = [color_lighter; c(2, :); c(3, :); c(4, :); c(5, :)];
        color_darkest = [color_darkest; c(1, :)];
    end
    fclose(foutput);
end
color_lighter = reshape(color_lighter, size(color_lighter, 1), 1, size(color_lighter, 2));
hsv_lighter = rgb2hsv(color_lighter);
hsv_lighter = squeeze(hsv_lighter);

figure
edges = {0:0.01:1-0.01 0:0.01:1-0.01};
hist3(hsv_lighter(:, 1:2), 'Edges', edges, 'CdataMode','auto');
xlabel('H')
ylabel('S')
view(2)
