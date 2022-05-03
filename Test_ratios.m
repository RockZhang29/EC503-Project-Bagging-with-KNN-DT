clear all;
close all;
clc;

fprintf('Loading the diabetes dataset\n');
raw = load('datasets/diabetes.mat');
addpath 'Z:\BU\2022Spring\EC503 Intro to Learning from Data\Project\bagging-boosting-random-forests-master\bagging\prtools4.2.5\prtools'

err_arr=[];
cpu_t_train=[];
cpu_t_pre=[];

for i=0.6:0.05:0.95
    err_index=[];
    cpu_index_train=[];
    cpu_index_pre=[];
    for j=1:10 % iterations calculate mean
        ratioTrainingSet = i; % 百分之多少的数据进行训练,可输入
        [Xapp, Yapp, Xtest, Ytest] = split(raw.X, raw.Y, ratioTrainingSet);
        fprintf('Keeping %f %% of the data in the training set\n', ratioTrainingSet * 100);

        nbBags = 100;  % number of bags: 提取nbBags份数据来做分析
        nbFolds = length(Xapp);

        fprintf('Running bagging with a tree classifier\n');
        fprintf('%d bags with %d folds per bag\n', nbBags, nbFolds);
        tic;
        %classifiers = baggingTrain(Xapp, Yapp, 'tree', nbBags, nbFolds);
        classifiers = baggingTrain(Xapp, Yapp, 'knn', nbBags, nbFolds);
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
    err_arr(end+1)=mean_err;
    cpu_t_train(end+1)=mean_cpu_train;
    cpu_t_pre(end+1)=mean_cpu_pre;
end
figure(1);
i=0.6:0.05:0.95;
plot(i,err_arr,'*-')
title('Validation error verse ratio')
xlabel('Ratio');
ylabel('Error precentage');

figure(2);
plot(i,cpu_t_train,'r*-');
hold on;
plot(i,cpu_t_pre,'b.-');
title('CPU times for train & prediction');
xlabel('Ratio');
ylabel('Times');
legend('CPU training time','CPU prediction time');