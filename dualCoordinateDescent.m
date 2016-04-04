function [] = dualCoordinateDescent%(x,y,xt,yt)

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
    x=[x,ones(m,1)];
    xt=[xt,ones(size(xt,1),1)];
else
    filetr='/home/mouly/Documents/mtp/Data_ML/mnist38_norm_svm_full_1.train';
    filete='/home/mouly/Documents/mtp/Data_ML/mnist38_norm_svm_full_1.test';
    [y,x,n]=ReadSparse(filetr);
    [yt,xt,nt]=ReadSparse(filete);
    n
end

m=size(x,1);

converge=false;
a1=ones(m,1)/m;
a1old=a1;

MAXITR=200;

b1=zeros(m,1);
b1old=b1;

%lambda rule increase in labda decrease the sum
lambda=1.00025;

if isSparse
    w=zeros(1,n+1);
    for i=1:m
        indSize=length(x(i).ind);
        if indSize~=0
            ind=x(i).ind;
            for j=1:indSize
                w(ind(j))=w(ind(j))+y(i)*(b1(i)-a1(i))*x(i).value(j);
            end
        end
    end
    w(n+1)=sum(y.*(b1-a1));
else
    w=(y.*(b1-a1))'*x;
end
j=0;
tic
if isSparse
    %ind=linspace(1,n+1,n+1);
    while ~converge
        changedvariable=0;
        for i=1:m
            if isempty(x(i).ind)==0
                qii=calculateqii(x(i).value,x(i).ind);
                [Gb,Ga]=calculateGradient(y(i),x(i).value,w',lambda,x(i).ind);
                pga=1;
                pgb=1;
                if Ga < 1e-5
                    pga=0;
                end
        %         if a1(i)==0
        %             pga=min(Ga,0);
        %         elseif a1(i)==c2;
        %             pga=max(Ga,0);
        %         else
        %             pga=Ga;
        %         end
                if pga~=0
                    t=a1(i);
                    a1old(i,:)=t;
                    a1(i)=min(max(a1(i)-Ga/qii,0),c2);
                    if abs(a1(i)-a1old(i))>=1e-5
                        changedvariable=changedvariable+1;
                    end
                end  
                if Gb < 1e-5
                    pgb=0;
                end
        %         
        %         if d1(i)==-a1(i)
        %             pgb=min(Gd,0);
        %         elseif d1(i)==c1-a1(i)
        %             pgb=max(Gd,0);
        %         else
        %             pgb=Gd;
        %         end
                if pgb~=0
                    t=b1(i);
                    b1old(i)=t;
                    %b1(i)-Gb(i)/Q(i,i)
                    b1(i)=min(max(b1(i)-Gb/qii,0),c1);
                    if abs(b1(i)-b1old(i))>=1e-5
                        changedvariable=changedvariable+1;
                    end
                end 
                
                if pga~=0 || pgb ~= 0
                    w=w+CalculateChangesinW(b1(i)-b1old(i),a1(i)-a1old(i),y(i),x(i).value,x(i).ind,n+1);
                end
            end
        end
        %alphas=[a1old,a1];
        %betas=[b1old,b1];

        j=j+1;
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

wcd=w
sumOfAlpha=sum(a1)
pred1=calculatePrediction(xt,w,size(xt,1));
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt)
nsv=nnz(b1-a1)


end


function [pred]=calculatePrediction(xt,w,m)
pred=zeros(m,1);
nw=length(w);
for i=1:m
    predv=w(nw);
    n=length(xt(i).ind);
    for j=1:n
        predv=predv+xt(i).value(j)*w(xt(i).ind(j));
    end
    pred(i)=sign(predv);
end
%sign(xt*w');
end

function [qii]=calculateqii(x,ind)
    qii=1;
    n=length(ind);
    for i=1:n
        qii=qii+x(i)*x(i);
    end
end

function [Gb, Ga] = calculateGradient(y,x,w,lambda,ind)
    Gb=y*w(length(w));
    n=length(ind);
    for i=1:n
        Gb=Gb+y*x(i)*w(ind(i));
    end
    
    Gb=Gb-1;
    Ga=lambda-Gb-1;
end

function [delw]=CalculateChangesinW(delb,dela,y,x,ind,n)
    delw=zeros(1,n);
    indSize=length(ind);

    for j=1:indSize
        delw(ind(j))=(delb-dela)*y*x(j);
    end
    
    delw(n)=(delb-dela)*y;
    %(((b1(i)-b1old(i)-a1(i)+a1old(i))*y(i))*x(i,:))
end




%{
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
%}