function [acc] = libSVMCode(datafiletrain,datatest ,Issparse)
addpath('libsvm-3.20/matlab');
%---------- koiN.mat 74%
%---------- ionoN.mat 86.06%
%---------- heartN.mat 84.07%
%---------- fdN.mat 87%
%---------- bupaN.mat 68.4%
%---------- pimaN.mat 76.1688%
% data=load('../Data_ML/');
%cd libsvm-3.20/matlab/
%make
%disp('make done');
%ac= ones(5,2)*10;
c=1;

if Issparse
    datafiletrain = '../Data_ML/a9a.train';
    datafiletest = '../Data_ML/a9a.test';
    [y, x] = libsvmread(datafiletrain);
    disp('train data loaded into memory');

    [yt, xt] = libsvmread(datafiletest);
    disp('test data loaded into memory');

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
% x = [data.x1;data.x2;data.x3;data.x4];
% y = [data.y1;data.y2;data.y3;data.y4];
% %ac(1,1)=gridSearchLibSVM(x,y);
str=sprintf('-s 0 -t 0 -c %d',1);
% xt = data.x5;
% yt = data.y5;
tic
model = svmtrain(y, x ,str);
toc
[predicted_label, accuracy, dv] = svmpredict(yt, xt, model);
accuracy
ac(1,2)=accuracy(1);
%  ---------------
%{
x = [data.x1;data.x2;data.x3;data.x5];
y = [data.y1;data.y2;data.y3;data.y5];
%ac(2,1)=gridSearchLibSVM(x,y);
str=sprintf('-s 0 -t 0 -c %d',ac(2,1));
xt = data.x4;
yt = data.y4;

model = svmtrain(y, x ,str);
[predicted_label, accuracy, dv] = svmpredict(yt, xt, model);
ac(2,2)=accuracy(1);

%------------------------
x = [data.x1;data.x2;data.x5;data.x4];
y = [data.y1;data.y2;data.y5;data.y4];
%ac(3,1)=gridSearchLibSVM(x,y);
str=sprintf('-s 0 -t 0 -c %d',ac(3,1));
xt = data.x3;
yt = data.y3;
model = svmtrain(y, x ,str);
[predicted_label, accuracy, dv] = svmpredict(yt, xt, model);
ac(3,2)=accuracy(1);
%-----------------------------
x = [data.x1;data.x5;data.x3;data.x4];
y = [data.y1;data.y5;data.y3;data.y4];
%ac(4,1)=gridSearchLibSVM(x,y);
str=sprintf('-s 0 -t 0 -c %d',ac(4,1));
xt = data.x2;
yt = data.y2;
model = svmtrain(y, x ,str);
[predicted_label, accuracy, dv] = svmpredict(yt, xt, model);
ac(4,2)=accuracy(1);
%-------------------------------------
x = [data.x5;data.x2;data.x3;data.x4];
y = [data.y5;data.y2;data.y3;data.y4];
%ac(5,1)=gridSearchLibSVM(x,y);
str=sprintf('-s 0 -t 0 -c %d',ac(5,1));
xt = data.x1;
yt = data.y1;
model = svmtrain(y, x ,str);
[predicted_label, accuracy, dv] = svmpredict(yt, xt, model);
ac(5,2)=accuracy(1);

dlmwrite(outfile,ac);
acc=mean(ac(:,2))
%cd ../../
%}
end