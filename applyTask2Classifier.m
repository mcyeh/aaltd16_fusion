% Apply the classifier to task 2's test data
% Chin-Chia Michael Yeh 05/28/2016
%
% applyTask1Classifier(dataPath, classPath, dicNum, spletLen, spletNum)
% Input:
%     dataPath: path to task 2's test data (string)
%     classPath: path to the output directory (string)
%     dicNum: number of dictionary element for sparse coding (scalar)
%     spletLen: shaplet length (scalar)
%     spletNum: number of shapelet (scalar)
%

function applyTask2Classifier(dataPath, classPath, dicNum, spletLen, spletNum)
%% check if the prediction is already done
fnameOut = sprintf('dn%d__sl%d__sn%d__task2.txt', dicNum, spletLen, spletNum);
fnameOut = fullfile(classPath, fnameOut);
if exist(fnameOut, 'file')
    return
end

%% initialization for parfor
fprintf('Initialize for parfor ... \n');
tTemp = tic();
workerNum = 4;
if isempty(which('parpool'))
    if matlabpool('size') <= 0 %#ok<*DPOOL>
        matlabpool(workerNum);
    elseif matlabpool('size')~= workerNum
        matlabpool('close');
        matlabpool(workerNum);
    end
else
    parProfile = gcp('nocreate');
    if isempty(gcp('nocreate'))
        parpool(workerNum);
    elseif parProfile.NumWorkers ~= workerNum
        delete(gcp('nocreate'));
        parpool(workerNum);
    end
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% load classifier
fnameClass = sprintf('dn%d__sl%d__sn%d.mat', dicNum, spletLen, spletNum);
fnameClass = fullfile(classPath, fnameClass);
dataRSLen = [];
dataSCLen = [];
splet = [];
dic = [];
svmModelRS = [];
svmModelSC = [];
featMean = [];
featStd = [];
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
for i = 1:dataNum
    dataRS{i} = zeros(dataDim, dataLen);
    for k = 1:dataDim
        dataRS{i}(k, :) = dataOrg2{k}(i, :);
    end
    dataRS{i} = [dataRS{i}, repmat(dataRS{i}(:, end), 1, 50)];
    dataSC{i} = normalizeData(dataRS{i});
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% compute prediction curve
fprintf('Generate prediction curve (random shapelet) ...\n');
tTemp = tic();
probPrRS = cell(dataNum, 1);
parfor i = 1:dataNum
    [~, probPrRS{i}] = applySlidingWindowRS(dataRS{i}, dataRSLen, ...
        splet, featMean, featStd, svmModelRS);
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% compute prediction curve
fprintf('Generate prediction curve (sparse coding) ...\n');
tTemp = tic();
probPrSC = cell(dataNum, 1);
parfor i = 1:dataNum
    [~, probPrSC{i}] = applySlidingWindowSC(dataSC{i}, dataSCLen, ...
        dic, svmModelSC);
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% post process the curve
probPr = cell(size(probPrRS));
labPr = cell(size(probPrRS));
for i = 1:length(probPr)
   for j = 1:size(probPrRS{i}, 1)
       probPrRS{i}(j, :) = smooth(probPrRS{i}(j, :), 51);
   end
   probPr{i} = probPrRS{i} * bestAccRS + probPrSC{i} * bestAccSC;
   labPr{i} = zeros(1, size(probPr{i}, 2));
   for j = 1:size(probPr{i}, 2)
       probPr{i}(:, j) = probPr{i}(:, j) / sum(probPr{i}(:, j));
       [~, labPr{i}(j)] = max(probPr{i}(:, j));
   end
end

%% find start and end
fprintf('Find label, start points, and end points ... ');
tTemp = tic();
stLoc = zeros(dataNum, 6);
edLoc = zeros(dataNum, 6);
lab = zeros(dataNum, 6);
for i = 1:dataNum
    for j = 1:6
        stLocLeft = max(1, (j-1)*51-24);
        stLocRight = min(306, (j-1)*51+26);
        lab(i,j) = mode(labPr{i}(stLocLeft:stLocRight));
        [~, stLoc(i,j)] = max(probPr{i}(lab(i,j), stLocLeft:stLocRight));
        stLoc(i,j) = stLoc(i,j) + stLocLeft - 1;
    end
    for j = 1:6
        edLocLeft = stLoc(i,j);
        if j == 6
            edLocRight = 306;
        else
            edLocRight = stLoc(i,j+1)-1;
        end
        [~, edLoc(i,j)] = min(probPr{i}(lab(i,j), edLocLeft:edLocRight));
        edLoc(i,j) = edLoc(i,j) + edLocLeft - 1;
        if j < 6 && stLoc(i,j+1)-edLoc(i,j)>20
            stLoc(i,j+1) = edLoc(i,j) + 1;
        end
    end
    edLoc(i,6) = 306;
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% output predicted label
fileID = fopen(fnameOut,'w');
for i = 1:dataNum
    for j = 1:6
        fprintf(fileID, '%d:%d-%d ', lab(i,j), stLoc(i,j)-1, edLoc(i,j)-1);
    end
    fprintf(fileID, '\n');
end
fclose(fileID);

function [labPr, probPr] = applySlidingWindowRS(data, slLen, splet, ...
    featMean, featStd, svmModel)
%% initialization
dataLen = size(data, 2);
slNum = dataLen - slLen + 1;

%% extract subsequences
feat = cell(slNum, 1);
for j = 1:slNum
    feat{j} = spletTran(data(:, j:j+slLen-1), splet);
end
feat = cell2mat(feat);
feat = feat - repmat(featMean, slNum, 1);
feat = feat ./ repmat(featStd, slNum, 1);
feat = full(feat);

%% apply svm model
lab = double(ones(slNum, 1));
[labPr, ~, probPr] = svmpredict2(lab, feat, svmModel, '-b 1');

%% reorder svm model
labPr = labPr';
probPrRe = zeros(size(probPr))';
for i = 1:length(svmModel.Label)
    probPrRe(svmModel.Label(i), :) = probPr(:, i);
end
probPr = probPrRe;

function [labPr, probPr] = applySlidingWindowSC(data, slLen, dic, svmModel)
%% initialization
dataLen = size(data, 2);
slNum = dataLen - slLen + 1;

%% extract subsequences
feat = cell(slNum, 1);
for j = 1:slNum
    feat{j} = sparseCodeTran(data(:, j:j+slLen-1), dic);
end
feat = cell2mat(feat);
feat = full(feat);

%% apply svm model
lab = double(ones(slNum, 1));
[labPr, ~, probPr] = svmpredict2(lab, feat, svmModel, '-b 1');

%% reorder svm model
labPr = labPr';
probPrRe = zeros(size(probPr))';
for i = 1:length(svmModel.Label)
    probPrRe(svmModel.Label(i), :) = probPr(:, i);
end
probPr = probPrRe;