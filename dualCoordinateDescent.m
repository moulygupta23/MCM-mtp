function [] = coordinateDecent%(x,y,xt,yt)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

C=1e0;
c1=1e0;
c2=1e0;

files='data/data1.mat';
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
o= ones(1,m);

x=[x,ones(m,1)];

lambda=4;

xt=[xt,ones(size(xt,1),1)];

x=[x,ones(m,1)];

%r=repmat(y,1,n+1).*x;
%Q=r*r';
converge=false;
%lambda=1.011; %for delta
%lambda=1.014; % for beta
%lambda=1.0001;
lambda=24.999;

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
    for i=1:m
        Gb=y(i)*x(i,:)*w'-1;
        Ga=-Gb-1;
        a1old(i,:)=a1(i);
        qii=x(i,:)*x(i,:)';
        a1(i)=min(max(a1(i)-Ga/qii,0),c2);
        diff=sum(a1)-a1(i)+a1old(i)-sum(a1old);
        a1=distributediff(a1,i,diff,c2);
%         a1(1:i-1)=min(max(a1(1:i-1)-(diff/(m-1)),0),c2);
%         a1(i+1:m)=min(max(a1(i+1:m)-(diff/(m-1)),0),c2);
        b1old(i)=b1(i);
        b1(i)=min(max(b1(i)-Gb/qii,0),c1);
        w=(y.*(b1-a1))'*x;%w+((((b1(i)-a1(i))-(b1old(i)-a1old(i)))*y(i))*x(i,:)); 
    end
    
    sum(a1);
    j=j+1;
    changedvariable=nnz([abs(a1old-a1);abs(b1old-b1)]>=1e-5);
    %Gdnew=Q*d1-o';
    %objnew=(0.5*((b1-a1)'*Q*(b1-a1))-o*(b1)+lambda*(o*a1-1));
    %d=max(Gdnew-Gd)
    %d=max(Gdnew)-min(Gdnew);
    %d=abs(obj-objnew);
    j;
    if j==MAXITR || changedvariable == 0
        converge=true;
        j
        changedvariable
    end
end
disp('coordinate descent');
toc

end
function [a1]=distributediff(a1,i,diff,c2)

m=length(a1);
diff2=2;
count=m;
while diff2~=0
    diff2=0;
    for j=1:m
        if i~=j
            tnew=a1(j)-(diff/(count-1));
            if a1(j)~=0 && a1(j) ~=c2
                if tnew < 0
                    diff2= diff - tnew;
                    a1(j) = 0;
                    count=count-1;
                elseif tnew > c2
                    diff2 = diff + (tnew-c2);
                    a1(j) = c2;
                    count=count-1;
                else
                    a1(j)=tnew;
                end
            end
        end
        
    end
    diff=diff2;    
end

end
