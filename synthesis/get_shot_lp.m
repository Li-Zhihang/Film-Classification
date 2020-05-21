function get_shot_lp(dname, fname, outdir)

hasFace = [];
posIndex = [];
posScore = [];

foutput = fopen([dname, fname]);
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
    [b, ~] = sort(c_sum);
    if b(5) < 10
        continue
    end

    hasFace = [hasFace; hf];
    posIndex = [posIndex; pi(1)];
    posScore = [posScore; ps(1)];
end
fclose(foutput);

datalen = length(hasFace);
if datalen > 0

%% merge hasface
for k = 1: datalen
    if hasFace(k) == 0
        posIndex(k) = -1;
    end
end
%% plot sat scale tone
% figure
% edges = -1:1:6;
% histogram(posIndex, edges, 'Normalization', 'probability');
% figure
% plot(posIndex)

%% summary
figure
edges = -1:1:6;
h_pose = histogram(posIndex, edges, 'Normalization', 'probability');
pose_h = h_pose.Values;

close all

[sa, si] = sort(pose_h, 'descend');
pose_typ = [si(1) - 2, si(2) - 2];
pose_val = [sa(1), sa(2)];

save([outdir, fname, '.mat'], 'pose_typ', 'pose_val')
else
    disp(fname)
end
end
