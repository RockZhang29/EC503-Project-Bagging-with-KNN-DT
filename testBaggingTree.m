clear all;
close all;
clc;

fprintf('Loading the diabetes dataset\n');
raw = load('datasets/diabetes.mat');
addpath 'Z:\BU\2022Spring\EC503 Intro to Learning from Data\Project\bagging-boosting-random-forests-master\bagging\prtools4.2.5\prtools'
ratioTrainingSet = 0.75; % 百分之多少的数据进行训练,可输入
[Xapp, Yapp, Xtest, Ytest] = split(raw.X, raw.Y, ratioTrainingSet);
fprintf('Keeping %f %% of the data in the training set\n', ratioTrainingSet * 100);

nbBags = 100;
nbFolds = 200;
err_arr=[];
for i = 1:25
    fprintf('Running bagging with a tree classifier\n');
    fprintf('%d bags with %d folds per bag\n', nbBags, nbFolds);
    tic;classifiers = baggingTrain(Xapp, Yapp, 'tree', nbBags, nbFolds);toc
    tic;baggingPredictions = baggingPredict(classifiers, Xtest, Ytest);toc
    err = baggingError(baggingPredictions, Ytest);
    fprintf('Error made: %f %%\n', err * 100);
    err_arr(end+1)=(err*100);
end
mn=mean(err_arr);
fprintf('The mean test error is %d %',mn);
