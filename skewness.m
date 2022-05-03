% out = skewness(data)
% Compute the skewness over a dataset
function out = skewness(data)
	out = mean((data - mean(data) / std(data)).^3); % 偏态是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征
end
