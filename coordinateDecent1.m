function [nsv,accuracy] = coordinateDecent1(x,y,xt,yt,n,lambda)
%this function reads file with the help of libsvmread function
tic
lambda=7.7;
C=1e0;
c1=10e0;
c2=10e0;
isSparse=true;

if ~isSparse
    files='data1.mat';
    data=load(files);
    xt=data.x5;
    yt=data.y5;

    x=[data.x1;data.x2;data.x3;data.x4];
    y=[data.y1;data.y2;data.y3;data.y4];

    xm = mean(x);
    xs = std(x);
    xs(xs==0)=1;

    x=(x-repmat(xm,size(x,1),1))./repmat(xs,size(x,1),1);
    xt=(xt-repmat(xm,size(xt,1),1))./repmat(xs,size(xt,1),1);
%else
    addpath('liblinear-2.1/matlab');
%     datafiletrain = '/home/mouly/Documents/mtp/MCM-mtp/sparsedata1.train';
%     datafiletest = '/home/mouly/Documents/mtp/MCM-mtp/sparsedata1.test';

    datafiletrain = '../Data_ML/a9a.train';
    datafiletest = '../Data_ML/a9a.test';
    [y, x] = libsvmread(datafiletrain);
    disp('train data loaded into memory');

    [yt, xt] = libsvmread(datafiletest);
    disp('test data loaded into memory');

    %n=max(n,nt);
end

m=size(x,1);

x=[x,ones(m,1)];
xt=[xt,ones(size(xt,1),1)];

n=size(x,2);
nt=size(xt,2);
%[              1;  2;                 3;       4;           5;          6;
%[initailaization;DCD;for loop iteration;qii copy;alpha update;beta update;
%       7;         8;             9]
%w update;prediction;gredient update]
timemat=zeros(9,1);
start=cputime;
converge=false;
a1=ones(m,1)/m;
a1old=a1;

MAXITR=400;

b1=zeros(m,1);
b1old=b1;

w=(y.*(b1-a1))'*x;
j=0;
Gb=y.*(x*w')-1;
Ga=lambda-Gb-1;
Q=zeros(1,m);
for i=1:m
    Q(i)=sum(x(i,:).^2);
end
disp('initialization complete');
timemat(1)=cputime-start;
start=cputime;

while ~converge
    changedvariable=0;
    for i=1:m
        %tic
        start2=cputime;
        %qii=sum(x(i,:).^2);
        qii=Q(i);
        timemat(4)=max(timemat(4),cputime-start2);
        pga=true;
        pgb=true;
         if abs(Ga(i)) < 1e-4
            pga=false;
         end
         if a1(i) <= 1e-5 && Ga(i)==max(Ga(i),0)
            pga=false;
        elseif a1(i) >= c2-1e-5 && Ga(i)==min(Ga(i),0)
            pga=false;
        end
        if pga
            start1=cputime;
            t=a1(i);
            a1old(i,:)=t;
            a1(i)=min(max(a1(i)-Ga(i)/qii,0),c2);
            if abs(a1(i)-a1old(i))>=1e-4
                changedvariable=changedvariable+1;
            end
            timemat(5)=max(timemat(5),cputime-start1);
        end  
        if abs(Gb(i)) < 1e-4
            pgb=false;
        end
        if b1(i) <=1e-5 && Gb(i)==max(Gb(i),0)
            pgb=false;
        elseif b1(i) >= c1-1e-5 && Gb(i)==min(Gb(i),0)
            pgb=false;
        end

        if pgb
            start1=cputime;
            t=b1(i);
            b1old(i)=t;
            b1(i)=min(max(b1(i)-Gb(i)/qii,0),c1);
            if abs(b1(i)-b1old(i))>=1e-4
                changedvariable=changedvariable+1;
            end
            timemat(6)=max(timemat(6),cputime-start1);
        end 

        if pga || pgb
            start1=cputime;
            delw=(((b1(i)-b1old(i)-a1(i)+a1old(i))*y(i))*x(i,:));
            w=w+delw;
            timemat(7)=max(timemat(7),cputime-start1);
            start1=cputime;
            Gb=Gb+y.*(x*delw');
            Ga=lambda-Gb-1;
            timemat(9)=max(timemat(9),cputime-start1);    
        end
        %toc
        timemat(3)=max(timemat(3),cputime-start2);
    end
    %alphas=[a1old,a1];
    %betas=[b1old,b1];

    j=j+1
    if j==MAXITR || changedvariable == 0 
        converge=true;
        changedvariable
        j
    end
    if cputime-start >= 18000
        converge=true;
        changedvariable
        j
        disp('timeout :P');
    end
end
timemat(2)=cputime-start;
toc

start1=cputime;
if n < nt
    xt=xt(:,1:n);
    w1=w;
elseif nt <= n
    w1=w(1:nt);
end

sumOfAlpha=sum(a1)
pred1=sign(xt*w1');
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt)
timemat(8)=cputime-start1;
nsv=nnz(b1-a1)
margin=1/norm(w1)
figure
%end



