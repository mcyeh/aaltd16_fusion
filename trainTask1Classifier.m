% Train the classifier using the training data
% Chin-Chia Michael Yeh 05/28/2016
%
% trainTask1Classifier(dataPath, classPath, dicNum, spletLen, spletNum)
% Input:
%     dataPath: path to the training data (string)
%     classPath: path to the output directory (string)
%     dicNum: number of dictionary element for sparse coding (scalar)
%     spletLen: shaplet length (scalar)
%     spletNum: number of shapelet (scalar)
%

function trainTask1Classifier(dataPath, classPath, dicNum, spletLen, spletNum)
%% check if the classifier is already train
fname = sprintf('dn%d__sl%d__sn%d.mat', dicNum, spletLen, spletNum);
fname = fullfile(classPath, fname);
if exist(fname, 'file')
   return
end

%% load data
fprintf('Loading data ... ');
tTemp = tic();
info = hdf5info(dataPath);
lab = hdf5read(info.GroupHierarchy.Datasets(1));
dataOrg = hdf5read(info.GroupHierarchy.Datasets(2));
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
lab = double(lab);
for i = 1:dataNum
    dataRS{i} = zeros(dataDim, dataLen);
    for k = 1:dataDim
        dataRS{i}(k, :) = dataOrg2{k}(i, :);
    end
    dataSC{i} = normalizeData(dataRS{i});
end
dataRSLen = size(dataRS{1}, 2);
dataSCLen = size(dataSC{1}, 2);
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% extract shapelet
fprintf('Dictionary training ... ');
tTemp = tic();
splet = randSpletSele(dataRS, spletLen, spletNum);
dic = learnDic(dataSC, dicNum);
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
featMean = mean(featRS, 1);
featStd = std(featRS, 1, 1);
featRS = featRS - repmat(featMean, dataNum, 1);
featRS = featRS ./ repmat(featStd, dataNum, 1);
featRS = full(featRS);
featSC = full(featSC);
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% cv for svm parameter
fprintf('Cross validation for SVM parameter (random shapelet) ...\n');
tTemp = tic();
svmCs = 2 .^ (-5:2:5);
bestAccRS = 0;
bestCRS = 1;
for i = 1:length(svmCs)
    svmC = svmCs(i);
    svmOption = ['-q -v 5 -t 0 -c ', num2str(svmC)];
    accTemp = svmtrain2(lab, featRS, svmOption);
    if accTemp >= bestAccRS
        bestCRS = svmC;
        bestAccRS = accTemp;
    end
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% cv for svm parameter
fprintf('Cross validation for SVM parameter (sparse coding) ...\n');
tTemp = tic();
svmCs = 2 .^ (-5:2:5);
bestAccSC = 0;
bestCSC = 1;
for i = 1:length(svmCs)
    svmC = svmCs(i);
    svmOption = ['-q -v 5 -t 0 -c ', num2str(svmC)];
    accTemp = svmtrain2(lab, featSC, svmOption);
    if accTemp >= bestAccSC
        bestCSC = svmC;
        bestAccSC = accTemp;
    end
end
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% train svm model
fprintf('Train SVM model (random shapelet) ... ');
tTemp = tic();
svmOption = ['-q -t 0 -b 1 -c ', num2str(bestCRS)];
svmModelRS = svmtrain2(lab, featRS, svmOption);
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% train svm model
fprintf('Train SVM model (sparse coding) ... ');
tTemp = tic();
svmOption = ['-q -t 0 -b 1 -c ', num2str(bestCSC)];
svmModelSC = svmtrain2(lab, featSC, svmOption);
tTemp = toc(tTemp);
fprintf('%5.3f s\n', tTemp);

%% save model
save(fname, 'dataRSLen', 'dataSCLen', 'splet', 'dic', 'featMean', 'featStd', ...
    'svmModelRS', 'svmModelSC', 'bestAccRS', 'bestAccSC');