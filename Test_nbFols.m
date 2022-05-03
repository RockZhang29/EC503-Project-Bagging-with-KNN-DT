clear all;
close all;
clc;

fprintf('Loading the diabetes dataset\n');
raw = load('datasets/diabetes.mat');
addpath 'Z:\BU\2022Spring\EC503 Intro to Learning from Data\Project\bagging-boosting-random-forests-master\bagging\prtools4.2.5\prtools'
ratioTrainingSet = 0.8; % 百分之多少的数据进行训练,可输入
[Xapp, Yapp, Xtest, Ytest] = split(raw.X, raw.Y, ratioTrainingSet);
fprintf('Keeping %f %% of the data in the training set\n', ratioTrainingSet * 100);

err_arr=[];
cpu_t_train=[];
cpu_t_pre=[];
nbBags = 100;
% number of bags: 将数据分成nbBags份
% elements per bag: nbFolds
for nbFolds = 50:50:1500
    err_index=[];
    cpu_index_train=[];
    cpu_index_pre=[];
    
    for j=1:2 % iterations calculate mean
    
    %nbFolds = length(Xapp);

    fprintf('Running bagging with a tree classifier\n');
    fprintf('%d bags with %d folds per bag\n', nbBags, nbFolds);
    tic;
    classifiers = baggingTrain(Xapp, Yapp, 'tree', nbBags, nbFolds);
    %classifiers = baggingTrain(Xapp, Yapp, 'knn', nbBags, nbFolds);
    cpu_index_train(end+1)=toc;
    tic;
    baggingPredictions = baggingPredict(classifiers, Xtest, Ytest);
    cpu_index_pre(end+1)=toc;
    err = baggingError(baggingPredictions, Ytest);
    err_index(end+1)=(err*100);
    %fprintf('Error made: %f %%\n', err * 100);
    end
    mean_err=mean(err_index);
    mean_cpu_train=mean(cpu_index_train);
    mean_cpu_pre=mean(cpu_index_pre);
    err_arr=[err_arr mean_err];
    cpu_t_train(end+1)=mean_cpu_train;
    cpu_t_pre(end+1)=mean_cpu_pre;
end

figure(1);
i=50:50:1500;
plot(i,err_arr,'*-')
title('Test error verse elements per bags')
xlabel('number of elements pre bags');
ylabel('Error precentage');

figure(2);
plot(i,cpu_t_train,'r*-');
hold on;
plot(i,cpu_t_pre,'b.-');
title('CPU times for train & prediction');
xlabel('Number of Bags');
ylabel('Times');
legend('CPU training time','CPU prediction time');