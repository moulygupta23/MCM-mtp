function [max_accuracy] = linearClassifier(option,train,test)
%MCM linear classifier with soft margin at option=2 and hard margin at
%option=1 this is margin option
%option=3 if soft margin gausian kernel 

%-------------------------------hard margin------------------------%
if option==1
    tic
    %data = importdata(dataset);
    data=load(dataset);
    a=zeros(5,1);

    x = [data.x1;data.x2;data.x3;data.x4];
    y = [data.y1;data.y2;data.y3;data.y4];
    xt=data.x5;
    yt=data.y5;
    xm = mean(x);
    xs = std(x);
    xs(xs==0)=1;

    x=(x-repmat(xm,size(x,1),1))./repmat(xs,size(x,1),1);
    xt=(xt-repmat(xm,size(xt,1),1))./repmat(xs,size(xt,1),1);

    a(1)=hardMCM(x,y,xt,yt)
%{\\
    x = [data.x1;data.x2;data.x3;data.x5];
    y = [data.y1;data.y2;data.y3;data.y5];
    xt=data.x4;
    yt=data.y4;
    xm = mean(x);
    xs = std(x);
    xs(xs==0)=1;

    x=(x-repmat(xm,size(x,1),1))./repmat(xs,size(x,1),1);
    xt=(xt-repmat(xm,size(xt,1),1))./repmat(xs,size(xt,1),1);
    a(2)=hardMCM(x,y,xt,yt);

    x = [data.x1;data.x2;data.x5;data.x4];
    y = [data.y1;data.y2;data.y5;data.y4];
    xt=data.x3;
    yt=data.y3;
    xm = mean(x);
    xs = std(x);
    xs(xs==0)=1;

    x=(x-repmat(xm,size(x,1),1))./repmat(xs,size(x,1),1);
    xt=(xt-repmat(xm,size(xt,1),1))./repmat(xs,size(xt,1),1);
    a(3)=hardMCM(x,y,xt,yt);

    x = [data.x1;data.x5;data.x3;data.x4];
    y = [data.y1;data.y5;data.y3;data.y4];
    xt=data.x2;
    yt=data.y2;
    xm = mean(x);
    xs = std(x);
    xs(xs==0)=1;

    x=(x-repmat(xm,size(x,1),1))./repmat(xs,size(x,1),1);
    xt=(xt-repmat(xm,size(xt,1),1))./repmat(xs,size(xt,1),1);
    a(4)=hardMCM(x,y,xt,yt);

    x = [data.x5;data.x2;data.x3;data.x4];
    y = [data.y5;data.y2;data.y3;data.y4];
    xt=data.x1;
    yt=data.y1;
    xm = mean(x);
    xs = std(x);
    xs(xs==0)=1;

    x=(x-repmat(xm,size(x,1),1))./repmat(xs,size(x,1),1);
    xt=(xt-repmat(xm,size(xt,1),1))./repmat(xs,size(xt,1),1);
    a(5)=hardMCM(x,y,xt,yt);

    max_accuracy=mean(a)
    toc
    %%}
    %--------------------end of hard margin-----------------------%
elseif option ==2
    %------------------------soft margin---------------------------%
    %------the follwing x and y are used as testing and training both
%    x= importdata('ix.data');%data(:,1:size(data,2)-1);
%    y= cell2mat(importdata('iy.data'));%data(:,size(data,2));
%     %--- this is done because in file class is in g and b
%    class = ones(size(y));
%    ind=find(y(:)=='g');
%     
%     for i=1:size(ind)
%         class(ind(i)) = -1;
%     end

    % comment above if want to use training and test data
    
    
    %data=load(dataset);
    choice1=0;%input('grid search? ');
    %choice1 0 means don't run and 1 means run
    choice=0;%input('compare cvx along side? ');
    %choice 0 means don't run and 1 means run
    a=zeros(5,2);
%     disp('fold1');
%     x= [data.x2;data.x3;data.x5;data.x4];
%     y = [data.y2;data.y3;data.y5;data.y4];
% addpath('liblinear-2.1/matlab');
%     datafiletrain = '/home/mouly/Documents/mtp/MCM-mtp/sparsedata1.train';
%     datafiletest = '/home/mouly/Documents/mtp/MCM-mtp/sparsedata1.test';

% datafiletrain = 'D:/Mouly/Data_ML/news20b_sparse_1.train';
% datafiletest = 'D:/Mouly/Data_ML/news20b_sparse_1.test';
% [y, x] = libsvmread(datafiletrain);
% disp('train data loaded into memory');
% 
% [yt, xt] = libsvmread(datafiletest);
% disp('test data loaded into memory');
data=dlmread(train);
x=data(:,2:size(data,2));
y=data(:,1);

data=dlmread(test);
xt=data(:,2:size(data,2));
yt=data(:,1);



    if choice1== 1
        c=gridSearchMCM(x,y);
    else
        %file=input('enter file with c value');
        %cdata=importdata(file);
        %c=cdata(1,2);
        c=1;
    end
    tic
    a(1,1)=mcm_classifier(x,y,xt,yt,c,option-1,choice);
    toc
    %a(1,2)=c;
    %{
    disp('fold2');
    x= [data.x1;data.x3;data.x5;data.x4];
    y = [data.y1;data.y3;data.y5;data.y4];
    if choice1== 1
        c=gridSearchMCM(x,y);
    else
       c=cdata(2,2);
    end
    a(2,1)=mcm_classifier(x,y,data.x2,data.y2,c,option-1,choice);
    a(2,2)=c;
    
    disp('fold3');
    x= [data.x2;data.x1;data.x5;data.x4];
    y = [data.y2;data.y1;data.y5;data.y4];
    if choice1== 1
        c=gridSearchMCM(x,y);
    else
        c=cdata(3,2);
    end
    a(3,1)=mcm_classifier(x,y,data.x3,data.y3,c,option-1,choice);
    a(3,2)=c;
    
    disp('fold4');
    x= [data.x2;data.x3;data.x5;data.x1];
    y = [data.y2;data.y3;data.y5;data.y1];
    if choice1== 1
        c=gridSearchMCM(x,y);
    else
        c=cdata(4,2);
    end
    a(4,1)=mcm_classifier(x,y,data.x4,data.y4,c,option-1,choice);
    a(4,2)=c;
    
    disp('fold5');
    x= [data.x2;data.x3;data.x1;data.x4];
    y = [data.y2;data.y3;data.y1;data.y4];
    if choice1== 1
        c=gridSearchMCM(x,y);
    else
        c=cdata(5,2);
    end
    a(5,1)=mcm_classifier(x,y,data.x5,data.y5,c,option-1,choice);
    a(5,2)=c;
    
    max_accuracy=mean(a(:,1))
    %load gong.mat;
    %sound(y);
    %if choice1==1
        %qw=input('enter file name');
        dlmwrite(outfile,a);
    %end
    %}
else
    
    data=load(dataset);
   
    a=zeros(5,1);
    
    c=linspace(1,1000,30);
    %k=1;
    gamma=linspace(1e-8,1e-5,50);
    max_accuracy=0;
    for k=1:1%length(c)
   % k=1;
    %j=1;
    %c(k)=100;
    %gamma(j)= 1e-8;
    for j = 1:1%length(gamma)
        x= [data.x2;data.x3;data.x5;data.x4];
        y = [data.y2;data.y3;data.y5;data.y4];
        a(1)=mcmWithKernel(x,y,data.x1,data.y1,c(k),gamma(j));

        x= [data.x1;data.x3;data.x5;data.x4];
        y = [data.y1;data.y3;data.y5;data.y4];
        a(2)=mcmWithKernel(x,y,data.x2,data.y2,c(k),gamma(j));

        x= [data.x2;data.x1;data.x5;data.x4];
        y = [data.y2;data.y1;data.y5;data.y4];
        a(3)=mcmWithKernel(x,y,data.x3,data.y3,c(k),gamma(j));

        x= [data.x2;data.x3;data.x5;data.x1];
        y = [data.y2;data.y3;data.y5;data.y1];
        a(4)=mcmWithKernel(x,y,data.x4,data.y4,c(k),gamma(j));

        x= [data.x2;data.x3;data.x1;data.x4];
        y = [data.y2;data.y3;data.y1;data.y4];
        a(5)=mcmWithKernel(x,y,data.x5,data.y5,c(k),gamma(j));

        accuracy=mean(a);
        if(accuracy >= max_accuracy)
            max_accuracy=accuracy;
            optC=c(k);
            og=gamma(j);
        end
        k
        j
    end
    end
    max_accuracy
    optC
    og
    size(data.x1)
end

end

function [accuracy] = hardMCM(x,y,xt,yt)

[m,n] = size(x);
%f= cat (2, 1, zeros(1,n+1));
f=cat (2, 1, zeros(1,m+1));%[h,lembda,b]
k=x*x';
r=repmat(y,1,m).*k;

cons1 = cat(2,ones(m,1)*-1,  r    ,  y);
cons2 = cat(2,zeros(m,1),-r,-y);
u1 = zeros(m,1);
u2 = ones(m,1)*-1;
D = cat(1,cons1,cons2);
U = cat(1,u1,u2);
options=optimset('Largescale', 'off', 'Simplex', 'on');
%  [h   ;lambda        ;b   ]
lb=[1.0  ;-inf*ones(m,1);-inf];
ub=+inf*ones(m+2,1);
x0=[];
Aeq=[];
Beq=[];
g=linprog(f,D,U,Aeq,Beq,lb,ub,x0,options);
lambda=g(2:m+1);
lambda(lambda~=0);
b=g(m+2)
h=g(1);
w=lambda'*x
nnz(lambda)
k= x*xt';

disp('kernel recalculated');
pred = sign(lambda'*k +b)';
%{
f= cat (2, 1, zeros(1,n+1));

%-----h constraint--%
cons1 = cat(2,ones(m,n+2)*-1);
u1 = zeros(m,1);

for i=1:m
    for j=2:n+2
        if j <= n+1
            cons1(i,j) = x(i,j-1)*y(i);
        else
            cons1(i,j) = y(i);
        end
    end
end

%--- >=1 constraint--%
cons2 = cat(2,zeros(m,n+2));
u2 = ones(m,1)*-1;

for i=1:m
    for j=2:n+2
        if j <= n+1
            cons2(i,j) = -x(i,j-1)*y(i);
        else
            cons2(i,j) = -y(i);
        end
    end
end

%----- all constraint matrix A---%
D = cat(1,cons1,cons2);
%-----matrix B------%
U = cat(u2 = ones(m,1)*-1;1,u1,u2);
options=optimset('Largescale', 'off', 'Simplex', 'on');

g=linprog(f,D,U,[],[],[0;-inf*ones(n+1,1)],+inf*ones(n+2,1),[],options);

w = g(2:n+1);
b = g(n+2);
%q = g(n+3:size(g,1)); 

%-------------------when test and train are same
%     xt=x;
%     yt=y;
%--------------- when and train are different load the test set here
%and comment above

pred = sign(xt*w +b);
%}
correct = sum((pred-yt)== 0);

accuracy = correct/length(pred)
end

