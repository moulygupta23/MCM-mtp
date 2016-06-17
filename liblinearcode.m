function [ ] = liblinearcode(datafiletrain,datatest ,Issparse)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

addpath('liblinear-2.1/matlab');
%datafile=('dataset/iono.mat');
%{
data=load(datafile);
x=[data.x1;data.x2;data.x3;data.x4];
y=[data.y1;data.y2;data.y3;data.y4];

xt=data.x5;
yt=data.y5;
%}
if Issparse
    datafiletrain
    [y, x] = libsvmread(datafiletrain);
    disp('data load');

    [yt, xt] = libsvmread(datatest);
    disp('data load');
else
    datafiletrain
    data = dlmread(datafiletrain);
    x = data(:,2:size(data,2));
    y = data(:,1);
    disp('x');

    data = dlmread(datatest);
    xt = data(:,2:size(data,2));
    yt = data(:,1);
    disp('xt');
end

tic
model = train(y, sparse(x), '-s 3 -c 1');
toc
%dlmwrite('webspam.model',model);
disp('data load');
% test the training data
[predict_label, accuracy, dec_values] = predict(yt, sparse(xt), model);
disp('data load');

end

