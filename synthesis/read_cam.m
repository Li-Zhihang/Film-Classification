clear
clc

parentName = '.\cam\';
folderList = dir(parentName);

camList = [];
L = [];
for k = 3: length(folderList)
    flist = dir([parentName, folderList(k).name]);
    label_count = length(flist) - 2;
    for p = 3:length(flist)
        fname = [parentName, folderList(k).name, '\', flist(p).name, '\', flist(p).name, '.stat'];
        fstat = fopen(fname);
        a = fscanf(fstat, '%f ', [92, 1])';
        if any(isnan(a))
            label_count = label_count - 1;
        else
            camList = [camList; a];
        end
        fclose(fstat);
    end
    L = [L,(k-2) * ones(1,label_count)];
end

L1_idx = (L==2);
L(L1_idx) = 1;
L3_idx = (L==4);
L(L3_idx) = 3;
L = (L+1) / 2;
Y = tsne(camList,'Algorithm','barneshut','NumPCAComponents',40,'NumDimensions',2);
figure
gscatter(Y(:,1),Y(:,2),L)
% scatter3(Y(:,1),Y(:,2),Y(:,3),15,L,'filled')

