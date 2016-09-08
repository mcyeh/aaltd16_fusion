% Randomly select shapelet from the dataset
% Chin-Chia Michael Yeh 06/16/2016
%
% splet = randSpletSele(data, spletLen, spletNum)
% Output:
%     splet: shaplet set, each row is a shapelet (matrix)
% Input:
%     data: training set, each row is a training data (cell)
%     spletLen: shaplet length (scalar)
%     spletNum: number of shapelet (scalar)
%

function splet = randSpletSele(data, spletLen, spletNum)
%% extract stat
data = cell2mat(data);
dataNum = size(data, 1);
dataLen = size(data, 2);

%% select shapelet
splet = zeros(spletNum, spletLen);
randIdx = randi([1, dataNum], spletNum, 1);
for i = 1:spletNum
    isHozLine = true;
    while isHozLine
        % generate random length and postiion
        randPos = randi([1, dataLen - spletLen + 1]);

        % extract shapelet
        splet(i, :) = data(randIdx(i), randPos:randPos+spletLen-1);

        % check if the selected shapelet is valid
        if std(splet(i, :), 1) > eps
            isHozLine = false;
        end
    end
end
