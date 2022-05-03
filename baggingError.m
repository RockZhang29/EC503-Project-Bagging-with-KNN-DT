% function err = baggingError(baggingPredictions, Ytest)
% 	Count the number of errors made by the bagging method
% 	Input:
% 	- baggingPredictions: predictions made by the bagging method
% 	- Ytest: real labels from the test set
%	Output:
%	- err: the error made, between 0 and 1
function err = baggingError(baggingPredictions, Ytest)
	% Count the number of errors made
	err = sum(baggingPredictions ~= Ytest) / length(baggingPredictions); % 预测的值不等于Ytest里的值的时候相加，然后除以预测的个数
end
