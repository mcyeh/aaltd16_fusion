%%
clear
clc

%% setting
dicNum = 1024;
spletLen = 25;
spletNum = 150;
classPath = '.\output';

%% train calssifier
dataPath = '.\dataset\train.h5';
trainTask1Classifier(dataPath, classPath, dicNum, spletLen, spletNum);

%% apply classifier on task 1
dataPath = '.\dataset\test_task1.h5';
applyTask1Classifier(dataPath, classPath, dicNum, spletLen, spletNum);

%% apply classifier on task 2
dataPath = '.\dataset\test_task2.h5';
applyTask2Classifier(dataPath, classPath, dicNum, spletLen, spletNum);