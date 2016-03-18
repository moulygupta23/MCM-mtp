function [ ] = liblinearcode(datafiletrain,datatest )
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
tic
[y, x] = libsvmread(datafiletrain);
disp('data load');

[yt, xt] = libsvmread(datatest);c
disp('data load');
model = train(y, sparse(x), '-c 1');
disp('data load');
% test the training data
[predict_label, accuracy, dec_values] = predict(yt, sparse(xt), model);
disp('data load');
toc
end

