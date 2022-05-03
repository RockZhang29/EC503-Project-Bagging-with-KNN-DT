% out = skewnessBootstrap(data, nbBags, nbFolds)
% 	Compute the mean skewness over a dataset using the Boostrap method
% 	Input:
% 	- data: the dataset
% 	- nbBags: the number of bags to create
% 	- nbFolds: the number of elements per bag
% 	Ouput:
% 	- out: the mean skewness for the dataset
function out = skewnessBootstrap(data, nbBags, nbFolds)
	skews = [];

	for i=1:nbBags
		[bag, oob] = drawBootstrap(length(data), nbFolds);
		skews = [skews;skewness(data(bag))]; %将数据集中所有数据的偏态进行计算，并存储
	end

	out = mean(skews); % 得到整个数据集的平均偏态值
end
