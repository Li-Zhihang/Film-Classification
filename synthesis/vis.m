clear
clc

load('film_data.mat')
% shot_data = [tsdata, scdata, lpdata, sldata, cdata];
shot_data = cdata;

Y = tsne(shot_data,'Algorithm','barneshut','NumPCAComponents',8,'NumDimensions',3);
figure
gscatter(Y(:,1),Y(:,2),L)
for k = 1:6
    mask = (L == k);
    Y_selected = Y(mask,:);
    h = scatter3(Y_selected(:,1),Y_selected(:,2),Y_selected(:,3),15, k * ones(length(Y_selected), 1),'filled');
    hold on
end
legend([{'Anderson'},{'Villeneuve'},{'Scorsese'},{'Burton'},{'WangJiaWei'},{'Rohrwacher'}])

% klist = [2, 4, 5];
% for k = klist
%     mask = (L == k);
%     Y_selected = Y(mask,:);
%     h = scatter3(Y_selected(:,1),Y_selected(:,2),Y_selected(:,3),15, k * ones(length(Y_selected), 1),'filled');
%     hold on
% end
% legend([{'Anderson'},{'Scorsese'},{'Rohrwacher'}])