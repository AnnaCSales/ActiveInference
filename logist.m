
function [y] = logist(x, k, max, min, mean)  %for calculating the decay factor df
max=max-min;
y=min+(max./ (1+exp(-k*(-(x-mean)))));
end