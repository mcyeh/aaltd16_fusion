% Compute the sparse coding of the input data, pool the sparse coding result
% with mean, and power normalized (cube root) the pooled result
% Chin-Chia Michael Yeh 05/28/2016
%
% feat = sparseCodeTran(data, dic)
% Output:
%     feat: pooled and normalized sparse coding output (vector)
% Input:
%     data: the multi dimensional time series (matrix)
%     dic: dictionary, each column is a dictionary element (matrix)
%

function feat = sparseCodeTran(data, dic)
%% sparse coding
param.lambda = 1/sqrt(size(data, 1));
code = mexLasso(data, dic, param);
code = full(code);

%% pooling
code = mean(code, 2);
code = abs(code);
feat = nthroot(code, 3);
feat = feat';