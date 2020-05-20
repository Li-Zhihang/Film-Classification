function get_shot_tss(dname, fname, outdir)

tone = [];
sat = [];
scale = [];
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

    tone = [tone; t];
    sat = [sat; s];

    scale = [scale; sc];
    hasFace = [hasFace; hf];
    posIndex = [posIndex; pi(0)];
    posScore = [posScore; ps(0)];
end
fclose(foutput);

datalen = length(scale);
if datalen > 0
%% smooth
if datalen > 3
for k = 2: datalen - 1
    if scale(k-1) == scale(k+1) && scale(k) ~= scale(k-1)
        scale(k) = scale(k-1);
    end
    if tone(k-1) == tone(k+1) && tone(k) ~= tone(k-1)
        tone(k) = tone(k-1);
    end
    if sat(k-1) == sat(k+1) && sat(k) ~= sat(k-1)
        sat(k) = sat(k-1);
    end
end
end

%% plot sat scale tone
% figure
% subplot(3, 1, 1)
% plot(sat)
% title('sat')
% subplot(3, 1, 2)
% plot(tone)
% title('tone')
% subplot(3, 1, 3)
% plot(scale)
% title('scale')
% 
% figure
% subplot(2, 1, 1)
% edges = 0:1:8;
% histogram(tone, edges, 'Normalization', 'probability')
% title('tone')
% xlim([0, 9])
% ylim([0., 1.])
% 
% subplot(2, 1, 2)
% edges = 0:1:3;
% histogram(sat, edges, 'Normalization', 'probability')
% title('sat')
% xlim([0, 4])
% ylim([0., 1.])
% 
% figure
% edges = -2:1:6;
% histogram(scale, edges, 'Normalization', 'probability')
% title('scale')
% xlim([-2, 7])
% ylim([0., 1.])
%% summary
edges = -2:1:6;
h_scale = histogram(scale, edges, 'Normalization', 'probability');
scale_h = h_scale.Values;

edges = 0:1:8;
h_tone = histogram(tone, edges, 'Normalization', 'probability');
tone_h = h_tone.Values;

edges = 0:1:3;
h_sat = histogram(sat, edges, 'Normalization', 'probability');
sat_h = h_sat.Values;

close all

[sa, si] = sort(tone_h, 'descend');
tone_typ = [si(1) - 1, si(2) - 1];
tone_val = [sa(1), sa(2)];
[sa, si] = sort(sat_h, 'descend');
sat_typ = si(1) - 1;
sat_val = sa(1);
[sa, si] = sort(scale_h, 'descend');
scale_typ = [si(1) - 3, si(2) - 3];
scale_val = [sa(1), sa(2)];


info = [tone_typ, tone_val, sat_typ, sat_val, scale_typ, scale_val];
save([outdir, fname, '.mat'], 'info')
else
    disp(fname)
end
end
