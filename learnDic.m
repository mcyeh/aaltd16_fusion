% Learn dictionary from the dataset
% Chin-Chia Michael Yeh 05/30/2016
%
% dic = learnDic(data, dicNum)
% Output:
%     dic: dictionary, each column is a dictionary element (matrix)
% Input:
%     data: training set, each row is a training data (cell)
%     dicNum: number of dictionary element (scalar)
%

function dic = learnDic(data, dicNum)
%% convert to matrix
data = cell2mat(data');

%% extract stat
dataDim = size(data, 1);
dataNum = size(data, 2);

%% learning parameter
param.K = dicNum;
param.lambda = 1/sqrt(dataDim);
param.iter = round(dataNum*10 / 512);

%% dictionary learning
dic = mexTrainDL(data, param);