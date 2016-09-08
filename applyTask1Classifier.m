% Apply the classifier to task 1's test data
% Chin-Chia Michael Yeh 05/28/2016
%
% applyTask1Classifier(dataPath, classPath, dicNum, spletLen, spletNum)
% Input:
%     dataPath: path to task 1's test data (string)
%     classPath: path to the output directory (string)
%     dicNum: number of dictionary element for sparse coding (scalar)
%     spletLen: shaplet length (scalar)
%     spletNum: number of shapelet (scalar)
%

function applyTask1Classifier(dataPath, classPath, dicNum, spletLen, spletNum)
%% check if the prediction is already done
fnameOut = sprintf('dn%d__sl%d__sn%d__task1.txt', dicNum, spletLen, spletNum);
fnameOut = fullfile(classPath, fnameOut);
if exist(fnameOut, 'file')
    return
end

%% load classifier
fnameClass = sprintf('dn%d__sl%d__sn%d.mat', dicNum, spletLen, spletNum);
fnameClass = fullfile(classPath, fnameClass);
load(fnameClass);

%% load data
fprintf('Loading data ... ');
tTemp = tic();
info = hdf5info(dataPath);
dataOrg = hdf5read(info.GroupHierarchy.Datasets(1));
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% organize data
fprintf('Organizing data ... ');
tTemp = tic();
dataDim = size(dataOrg, 1);
dataLen = size(dataOrg, 2);
dataNum = size(dataOrg, 3);
dataOrg2 = cell(dataDim, 1);
for i = 1:dataDim
    dataOrg2{i} = squeeze(dataOrg(i, :, :))';
end
dataRS = cell(dataNum, 1);
dataSC = cell(dataNum, 1);
lab = double(ones(dataNum, 1));
for i = 1:dataNum
    dataRS{i} = zeros(dataDim, dataLen);
    for k = 1:dataDim
        dataRS{i}(k, :) = dataOrg2{k}(i, :);
    end
    dataSC{i} = normalizeData(dataRS{i});
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% shapelet trainsform
fprintf('Shapelet transform and sparse coding ... ');
tTemp = tic();
featRS = cell(dataNum, 1);
featSC = cell(dataNum, 1);
for i = 1:dataNum
    featRS{i} = spletTran(dataRS{i}, splet);
    featSC{i} = sparseCodeTran(dataSC{i}, dic);
end
featRS = cell2mat(featRS);
featSC = cell2mat(featSC);
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% normalization
fprintf('Normalize feature ... ');
tTemp = tic();
featRS = featRS - repmat(featMean, dataNum, 1);
featRS = featRS ./ repmat(featStd, dataNum, 1);
featRS = full(featRS);
featSC = full(featSC);
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% apply svm model
fprintf('Apply SVM model ...\n');
tTemp = tic();
[~, ~, probRSTemp] = svmpredict2(lab, featRS, svmModelRS, '-b 1');
probRS = zeros(size(probRSTemp));
probRS(:, svmModelRS.Label) = probRSTemp;
[~, ~, probSCTemp] = svmpredict2(lab, featSC, svmModelSC, '-b 1');
probSC = zeros(size(probSCTemp));
probSC(:, svmModelSC.Label) = probSCTemp;
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% fuse result
fprintf('Fusion ... ');
tTemp = tic();
probFus = probSC*bestAccSC + probRS*bestAccRS;
labPr = zeros(dataNum, 1);
for j = 1:dataNum
    [~, labPr(j)] = max(probFus(j, :));
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% output predicted label
fileID = fopen(fnameOut,'w');
fprintf(fileID, '%d\n',labPr);
fclose(fileID);