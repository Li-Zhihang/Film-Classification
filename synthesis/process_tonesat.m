clear
clc

dname = '.\outputs\kamchatka\';

tone = [];
sat = [];
frame_num = 0;

flist = dir(dname);
for k = 3:length(flist)
    foutput = fopen([dname, flist(k).name]);
    while ~feof(foutput)
        c = fscanf(foutput, '%d %d %d\n', [3, 5])';

        t = str2double(fgetl(foutput));
        s = str2double(fgetl(foutput));

        c_sum = sum(c, 2);
        [b, index] = sort(c_sum');
        if b(5) < 10
            continue
        end

        tone = [tone; t];
        sat = [sat; s];

        frame_num = frame_num + 1;
    end
    fclose(foutput);
end

%% sat scale tone
figure
subplot(2, 1, 1)
plot(sat)
title('sat')
subplot(2, 1, 2)
plot(tone)
title('tone')

figure
subplot(2, 1, 1)
edges = 0:1:7;
histogram(tone, edges, 'Normalization', 'probability')
title('tone')
xlim([0, 8])
ylim([0., 1.])

subplot(2, 1, 2)
edges = 0:1:2;
histogram(sat, edges, 'Normalization', 'probability')
title('sat')
xlim([0, 3])
ylim([0., 1.])