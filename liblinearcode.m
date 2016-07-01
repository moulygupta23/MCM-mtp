function [ accuracy] = liblinearcode%(datafiletrain,datatest ,Issparse)
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
if true
    
    [y, x] = libsvmread('../Data_ML/webspam.train');
    disp('data load');

    [yt, xt] = libsvmread('../Data_ML/webspam.test');
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
str=sprintf('-s 3 -c 1');
model = train(y, sparse(x), str);
toc
%dlmwrite('webspam.model',model);
disp('data load');
% test the training data
[predict_label, accuracy, dec_values] = predict(yt, sparse(xt), model);
pred=[predict_label,yt];
tp=0;
tn=0;
fn=0;
fp=0;
min(yt)
max(yt)
for i=1:length(yt)
    if predict_label(i)==1&&yt(i)==1
        tp = tp+1;
    elseif predict_label(i)==1&&yt(i)==-1
        fp = fp+1;
    elseif predict_label(i)==-1&&yt(i)==1
        fn= fn+1;
    else
        tn=tn+1;
    end
end
tp
fp
fn
tn
tn+tp+fn+fp
length(yt)
disp('data load');

end

