% Shapelet transform and power normalized (square root) the output
% Chin-Chia Michael Yeh 05/28/2016
%
% dataTran = spletTran(data, splet)
% Output:
%     dataTran: shapelet trasformation output (vector)
% Input:
%     data: the multi dimensional time series (matrix)
%     splet: shaplet set, each row is a shapelet (matrix)
%

function dataTran = spletTran(data, splet)
%% perform the transform
spletNum = size(splet, 1);
dataDim = size(data, 1);
dataTran = cell(1, dataDim);
for i = 1:dataDim
    dataTran{i} = zeros(1, spletNum);
    for j = 1:spletNum
        dist = distanceProfile(data(i, :)', splet(j, :)');
        dataTran{i}(j) = nthroot(min(dist), 2);
    end
end
dataTran = cell2mat(dataTran);