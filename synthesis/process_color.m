function process_color(direct, fname, outdir)
foutput = fopen([direct, fname]);
% tone = [];
% sat = [];
color_lighter = [];
color_darkest = [];
frame_num = 0;
while ~feof(foutput)
    c = fscanf(foutput, '%d %d %d\n', [3, 5])';

    t = str2double(fgetl(foutput));
    s = str2double(fgetl(foutput));
%     sy = fscanf(foutput, '%f %f %f %f\n', [4, 1])';
%     sc = str2double(fgetl(foutput));
%     hf = str2double(fgetl(foutput));
%     pi = fscanf(foutput, '%d %d\n', [2, 1])';
%     ps = fscanf(foutput, '%f %f\n', [2, 1])';

    c_sum = sum(c, 2);
    [b, index] = sort(c_sum');
    if b(5) < 10
        continue
    end
    c = c(index, :);
    color_lighter = [color_lighter; c(2, :); c(3, :); c(4, :); c(5, :)];
    color_darkest = [color_darkest; c(1, :)];

%     tone = [tone; t];
%     sat = [sat; s];

%     sym = [sym; sy];
%     scale = [scale; sc];
%     hasFace = [hasFace; hf];
%     posIndex = [posIndex; pi];
%     posScore = [posScore; ps];
    frame_num = frame_num + 1;
end
fclose(foutput);
%% draw color hist

color_lighter = reshape(color_lighter, size(color_lighter, 1), 1, size(color_lighter, 2));
hsv_lighter = rgb2hsv(color_lighter);
hsv_lighter = squeeze(hsv_lighter);

% figure
% edges = {0:0.01:1-0.01 0:0.01:1-0.01};
% hist3(hsv_lighter(:, 1:2), 'Edges', edges, 'CdataMode','auto');
% xlabel('H')
% ylabel('S')
% view(2)

% figure
% plot3(hsv_lighter(:, 1), hsv_lighter(:, 2), hsv_lighter(:, 3), '.')
% xlim([0., 1.])
% ylim([0., 1.])
% zlim([0, 256])

edges = {0:0.01:1-0.01 0:0.01:1-0.01};
h = hist3(hsv_lighter(:, 1:2), 'Edges', edges, 'CdataMode','auto');
[pks_val, pks_index] = findpeaks2d(h);

if ~isempty(pks_val)
    pks_num = size(pks_index, 1);

    center_num = 1;
    center_index = pks_index(1, :);
    center_value = pks_val(1);
    for k = 2: pks_num  % if the local maximum is too close to the prev lm, abandon it
        dist = mydist(pks_index(k, :), center_index(1:center_num, :));
        if pks_val(k) >= min(20, frame_num / 25) && min(dist) > 15
            center_index = [center_index; pks_index(k, :)];
            center_value = [center_value; pks_val(k)];
            center_num = center_num + 1;
        end
    end
    center_index = center_index ./ 100;

    % disp(center_index)
    % disp(center_value)

    % filter all points less than frame_num / 100
    cluster_mask = h > min(5, frame_num / 80);
    cluster_mat = cluster_mask .* h;
    cluster_num = sum(cluster_mat, 'all');
    cluster_coors = zeros(cluster_num, 2);
    idx_count = 0;
    for r = 1: size(h, 1)
        for c = 1: size(h, 2)
            t = cluster_mat(r, c);
            if t ~= 0
                cluster_coors(idx_count + 1: idx_count + t, :) = repmat([(r-0.5)/100, (c-0.5)/100], t, 1);
                idx_count = idx_count + t;
            end
        end
    end

    % random permute
    cluster_coors = cluster_coors(randperm(idx_count), :);

    % figure
    % edges = {0:0.01:1-0.01 0:0.01:1-0.01};
    % hist3(cluster_coors, 'Edges', edges, 'CdataMode','auto');
    % xlabel('H')
    % ylabel('S')
    % view(2)
    
    info = [];
    cluster_idx = kmeans(cluster_coors, center_num, 'Distance', 'cityblock', 'EmptyAction', 'error', 'Start', center_index);
    for k = 1: min(center_num, 4)  % maximum 4 center
        type_coors = [];
        for p = 1: idx_count
            if cluster_idx(p) == k
                type_coors = [type_coors; cluster_coors(p, :)];
            end
        end

%         figure
%         edges = {0:0.01:1-0.01 0:0.01:1-0.01};
%         hist3(type_coors, 'Edges', edges, 'CdataMode','auto');
%         xlabel('H')
%         ylabel('S')
%         title(string(k))
%         view(2)

        mean1 = mean(type_coors(:, 1));
        std1 = std(type_coors(:, 1));
        mean2 = mean(type_coors(:, 2));
        std2 = std(type_coors(:, 2));
        info = [info;[mean1, std1, mean2, std2, center_index(k, 1), center_index(k, 2), center_value(k) / length(type_coors), length(type_coors) / length(hsv_lighter)]];
        
    end
    save([outdir, fname, '.mat'], 'info')
end

end

function dist = mydist(coor, points)
points_num = size(points, 1);
dist = zeros(points_num, 1);
for k = 1: points_num
    t = abs(coor - points(k, :));
    dist(k) = 1.4 * t(1) + 0.6 * t(2);
end
end