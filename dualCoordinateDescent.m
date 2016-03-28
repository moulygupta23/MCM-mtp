function [] = dualCoordinateDescent%(x,y,xt,yt)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

c1=1e0;
c2=1e0;

files='dataset/iono.mat';
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
[m,n]=size(x);

xt=[xt,ones(size(xt,1),1)];
x=[x,ones(m,1)];

converge=false;

a1=ones(m,1)/m;
a1old=a1;

MAXITR=200;

b1=zeros(m,1);%c1*rand(m,1);
b1old=b1;%zeros(m,1);
w=(y.*(b1-a1))'*x;

tic
j=0;
while ~converge
    sum(a1);
    for i=1:m-1
        Gb=y(i)*x(i,:)*w'-1;
        Ga=-Gb-1;
        a1old(i,:)=a1(i);
        qii=x(i,:)*x(i,:)';
        a1(i)=min(max(a1(i)-Ga/qii,0),c2);
        diff=a1(i)-a1old(i);
        k=i+1;
        a1old(k)=a1(k);
        a1(k)=min(max(a1(k)-diff,0),c2);
        b1old(i)=b1(i);
        b1(i)=min(max(b1(i)-Gb/qii,0),c1);
        b1old(k)=b1(k);
        qkk=x(k,:)*x(k,:)';
        Gbk=y(k)*x(k,:)*w'-1;
        b1(k)=min(max(b1(k)-Gbk/qkk,0),c1);
        w=(y.*(b1-a1))'*x;%w+((((b1(i)-a1(i))-(b1old(i)-a1old(i)))*y(i))*x(i,:)); 
        i=i+1;
    end
    
    sum(a1);
    j=j+1;
    changedvariable=nnz([abs(a1old-a1);abs(b1old-b1)]>=1e-5);
    if j==MAXITR || changedvariable == 0
        converge=true;
        j
        changedvariable
    end
end
disp('coordinate descent');
size(xt)
size(w)
pred1=sign(xt*w');
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt)
nsv=nnz(b1-a1)

toc

end