function [sumOfAlpha] = coordinateDecent1%(x,y,xt,yt,n,lambda)
%this function reads file with the help of libsvmread function
lambda=23.6;
C=1e0;
c1=1e0;
c2=1e0;
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
else
    addpath('liblinear-2.1/matlab');
%     datafiletrain = '/home/mouly/Documents/mtp/MCM-mtp/sparsedata1.train';
%     datafiletest = '/home/mouly/Documents/mtp/MCM-mtp/sparsedata1.test';

    datafiletrain = '/home/mouly/Documents/mtp/Data_ML/mnist38_norm_svm_full_1.train';
    datafiletest = '/home/mouly/Documents/mtp/Data_ML/mnist38_norm_svm_full_1.test';
    [y, x] = libsvmread(datafiletrain);
    disp('train data loaded into memory');

    [yt, xt] = libsvmread(datafiletest);
    disp('test data loaded into memory');

    n=size(x,2);
    nt=size(xt,2);
    %n=max(n,nt);
end

m=size(x,1);
if n < nt
    x=[x,zeros(m,nt-n)];
elseif nt < n
    xt=[xt,zeros(m,n-nt)];
end
n=max(n,nt);
x=[x,ones(m,1)];
xt=[xt,ones(size(xt,1),1)];

converge=false;
a1=ones(m,1)/m;
a1old=a1;

MAXITR=10;

b1=zeros(m,1);
b1old=b1;

w=(y.*(b1-a1))'*x;
j=0;

disp('initialization complete');

tic
if isSparse
    while ~converge
        changedvariable=0;
        for i=1:m
            if nnz(x(i,:)~=0)
                qii=x(i,:)*x(i,:)';
                [Gb,Ga]=calculateGradient(y(i),x(i,:),w',lambda,n+1);
                pga=1;
                pgb=1;
                 if abs(Ga) < 1e-5
                    pga=0;
                 end
%                  if a1(i) <= 1e-6 && -Ga==min(-Ga,0)
%                     pga=0;
%                 elseif a1(i) >= c2-1e-5 && Ga==min(Ga,0)
%                     pga=0;
% %                 else
% %                     pga=Ga;
%                 end
                if pga~=0
                    t=a1(i);
                    a1old(i,:)=t;
                    a1(i)=min(max(a1(i)-Ga/qii,0),c2);
                    if abs(a1(i)-a1old(i))>=1e-5
                        changedvariable=changedvariable+1;
                    end
                end  
                if abs(Gb) < 1e-5
                    pgb=0;
                end
%                 if b1(i) <=1e-6 && -Gb==min(-Gb,0)
%                     pgb=0;
%                 elseif b1(i) >= c1-1e-5 && Gb==min(Gb,0)
%                     pgb=0;
% %         %         elseif a1(i)==c2;
% %         %             pga=max(Ga,0);
% %         %         else
% %         %             pga=Ga;
%                 end
        
                if pgb~=0
                    t=b1(i);
                    b1old(i)=t;
                    b1(i)=min(max(b1(i)-Gb/qii,0),c1);
                    if abs(b1(i)-b1old(i))>=1e-5
                        changedvariable=changedvariable+1;
                    end
                end 
                
                if pga~=0 || pgb ~= 0
                    w=w+CalculateChangesinW(b1(i)-b1old(i),a1(i)-a1old(i),y(i),x(i,:),n+1);
                end
            end
        end
        %alphas=[a1old,a1];
        %betas=[b1old,b1];

        j=j+1
        if j==MAXITR || changedvariable == 0
            converge=true;
            changedvariable
            j
        end
    end
else
    while ~converge
        for i=1:m
            qii=x(i,:)*x(i,:)';
            Gb=y(i)*x(i,:)*w'-1;
            Ga=lambda-Gb-1;
    %         if a1(i)==0
    %             pga=min(Ga,0);
    %         elseif a1(i)==c2;
    %             pga=max(Ga,0);
    %         else
    %             pga=Ga;
    %         end
    %         if pga~=0
                t=a1(i);
                a1old(i,:)=t;
                a1(i)=min(max(a1(i)-Ga/qii,0),c2);
    %         end   
    %         
    %         if d1(i)==-a1(i)
    %             pgb=min(Gd,0);
    %         elseif d1(i)==c1-a1(i)
    %             pgb=max(Gd,0);
    %         else
    %             pgb=Gd;
    %         end
    %         if pgb~=0
                t=b1(i);
                b1old(i)=t;
                %b1(i)-Gb(i)/Q(i,i)
                b1(i)=min(max(b1(i)-Gb/qii,0),c1);
                w=w+(((b1(i)-b1old(i)-a1(i)+a1old(i))*y(i))*x(i,:));
    %         end    
        end
        %alphas=[a1old,a1];
        %betas=[b1old,b1];

        j=j+1;
        changedvariable=nnz([abs(a1old-a1);abs(b1old-b1)]>=1e-5);
        if j==MAXITR || changedvariable == 0
            converge=true;
        end
    end
end
toc

sumOfAlpha=sum(a1)
pred1=calculatePrediction(xt,w,size(xt,1));
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt)
nsv=nnz(b1-a1)
margin=1/norm(w)


end

function [pred]=calculatePrediction(xt,w,m)
    pred=sign(xt*w');
end

function [Gb, Ga] = calculateGradient(y,x,w,lambda,n)
%     Gb=0;
    Gb=y*x*w-1;
%     for k=1:n
%         if x(k)~=0
%             Gb = Gb + y*x(k)*w(k);
%         end
%     end
    
%     Gb=Gb-1;
    Ga=lambda-Gb-1;
end

function [delw]=CalculateChangesinW(delb,dela,y,x,n)
%     delw=zeros(1,n);
%     for j=1:n
%         if x(j)~=0
%             delw(j)=(delb-dela)*y*x(j);
%         end
%     end
%   
    delw=(delb-dela)*y*x;
    %(((b1(i)-b1old(i)-a1(i)+a1old(i))*y(i))*x(i,:))
end
