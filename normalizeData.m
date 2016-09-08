% Perform power normalization (cube root), consecutive frame concatenation, and
% unit-norm normalize to the input data
% Chin-Chia Michael Yeh 05/28/2016
%
% data = normalizeData(data)
% Output:
%     data: the normalized multi dimensional time series (matrix)
% Input:
%     data: the multi dimensional time series (matrix)
%

function data = normalizeData(data)
dataTemp = nthroot(data, 3);
canNum = 16;
data = zeros(canNum*size(dataTemp, 1), size(dataTemp, 2)-canNum+1);
for i = 1:canNum
    data((i-1)*size(dataTemp, 1)+1:i*size(dataTemp, 1), :) = ...
        dataTemp(:, i:end-canNum+i);
end

for i = 1:size(data, 2)
    data(:, i) = data(:, i) / norm(data(:, i));
end