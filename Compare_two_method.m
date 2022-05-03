% Compare
clear all;
close all;
clc;

fprintf('Loading the diabetes dataset\n');
raw = load('datasets/a9a.mat');
Xapp=raw.Xte;
Yapp=raw.yte;
Xtest=raw.Xtr;
Ytest=raw.ytr;
%raw = load('datasets/diabetes.mat');
%addpath 'Z:\BU\2022Spring\EC503 Intro to Learning from Data\Project\bagging-boosting-random-forests-master\bagging\prtools4.2.5\prtools'
%ratioTrainingSet = 0.8; % 百分之多少的数据进行训练,可输入
%[Xapp, Yapp, Xtest, Ytest] = split(raw.X, raw.Y, ratioTrainingSet);
%fprintf('Keeping %f %% of the data in the training set\n', ratioTrainingSet * 100);

nbBags = 100;
nbFolds = 200;
err_tree=[];
err_knn=[];
t_tree=0;
t_knn=0;
for i = 1:2
    fprintf('Running bagging with a tree classifier\n');
    fprintf('%d bags with %d folds per bag\n', nbBags, nbFolds);
    tic;
    classifiers = baggingTrain(Xapp, Yapp, 'tree', nbBags, nbFolds);
    t1=toc;
    t_tree=t_tree+t1;
    tic;
    baggingPredictions = baggingPredict(classifiers, Xtest, Ytest);
    t2=toc;
    t_tree=t_tree+t2;
    err = baggingError(baggingPredictions, Ytest);
    %fprintf('Error made: %f %%\n', err * 100);
    err_tree(end+1)=(err*100);
end
mn_tree=mean(err_tree);

for i = 1:2
    fprintf('Running bagging with a tree classifier\n');
    fprintf('%d bags with %d folds per bag\n', nbBags, nbFolds);
    tic;
    classifiers = baggingTrain(Xapp, Yapp, 'knn', nbBags, nbFolds);
    t1=toc;
    t_knn=t_knn+t1;
    tic;
    baggingPredictions = baggingPredict(classifiers, Xtest, Ytest);
    t2=toc;
    t_knn=t_knn+t2;
    err = baggingError(baggingPredictions, Ytest);
    %fprintf('Error made: %f %%\n', err * 100);
    err_knn(end+1)=(err*100);
end
mn_knn=mean(err_knn);

k=1:10;
plot(k,err_tree,'r*-');
hold on;
plot(k,err_knn,'b.-');
title('Validation error between Knn and Tree');
legend('Tree','Knn');
