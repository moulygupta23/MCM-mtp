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
%{
%------------------------------------------------------
% min h + c1*sum(qi) + c2*sum(etai)
% such that
%     yi(w'*xi+b)+qi >= 1
%     yi(w'*xi+b)<= h + etai
%     qi>=0
%     eta>=0
tic
cvx_begin
    variables w(n) b h q(m) eta(m)
    minimize( h+ c1*o*q+c2*o*eta);
    subject to
    y.*(x*w + b) >= o' - q;
    y.*(x*w + b) <= o'*h + eta;
    q>=0;
    eta>=0;
cvx_end
disp('original primal');
toc
w1primal=w;
b1primal=b;
h1primal=h;

pred1=sign(xt*w1primal+b1primal);
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt)

% ---------------------------------------------------------------------------
% max sum(betai)
% such that
%     sum(alphai)=1
%     sum((betai-alphai).*y)=0
%     0<=beta<=c1
%     0<=alpha<=c2
%
tic
cvx_begin
    variables a(m) be(m);
    dual variables u u1 u2;
    maximize(o*be);
    subject to
        u:sum(a)==1;
        u1:sum((be-a).*y)==0;
        u2:sum(repmat((be-a).*y,1,n).*x)==0;
        0<=be<= c1;
        0<=a<=c2;
cvx_end
disp('original dual');
toc
h1dual=u;
b1dual=-u1;
w1dual=-u2;
pred1=sign(xt*w1dual'+b1dual);
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt)
%------------------------------------------------------
% min 1/2* ||w||^2 + h + c1*sum(qi) + c2*sum(etai)
% such that
%     yi(w'*xi+b)+qi >= 1
%     yi(w'*xi+b)<= h + etai
%     qi>=0
%     eta>=0
tic
cvx_begin
    variables w(n) b h q(m) eta(m)
    minimize( .5*w'*w +h+ c1*o*q+c2*o*eta);
    subject to
    y.*(x*w + b) >= o' - q;
    y.*(x*w + b) <= o'*h + eta;
    q>=0;
    eta>=0;
cvx_end
toc
w2primal=w;
b2primal=b;
h2primal=h;

pred1=sign(xt*w2primal+b2primal);
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt)
%}
%%{
Q=x*x';
% ---------------------------------------------------------------------------
% max 1/2*sumationij (betai-alphai)*yi*xi*xj'*yj*(betaj-alphaj) + sum(betai)
% such that
%     sum(alphai)=1
%     sum((betai-alphai).*y)=0
%     0<=beta<=c1
%     0<=alpha<=c2
%
tic
cvx_begin
    variables a(m) be(m);
    dual variables u u1;
    maximize(-0.5*quad_form(y.*(be-a),Q)+o*be);
    subject to
        u:sum(a)==1;
        u1:sum((be-a).*y)==0;
        0<=be<= c1;
        0<=a<=c2;
cvx_end
toc
w2dual=((be-a).*y)'*x
b2dual=u1
pred1=sign(xt*w2dual'+b2dual);
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt);
sum(a);
Gbeta=Q*(be-a)-1;
Galfa=-Q*(be-a);
% [Galfa,Gbeta]
% r=repmat(y,1,n).*x;
% Q=r*r';
% gb=[];
% ga=[];
% for i=1:m
%     gb=[gb;(0.5*(Q(i,:)*(be-a)+Q(i,i)*(be(i)-a(i)))-1)];
%     ga=[ga;-0.5*(Q(i,:)*(be-a)+Q(i,i)*(be(i)-a(i)))];
% end
% ga
% gb
%}
%}
%{
%-------------------- Equivalent Form ------------------%
cvx_begin
    variables a1(m) de(m);
    dual variables u u1;
    maximize(-0.5*quad_form(y.*(de),Q)+o*(de + a1))
    subject to
        u:sum(a1)==1;
        u1:sum((de).*y)==0;
        0<=de + a1<= c1;
        0<=a1<=c2;
cvx_end
w2=(de.*y)'*x
sum(a1)
%}
%{
x=[x,ones(m,1)];

%r=repmat(y,1,n+1).*x;
%Q=r*r';
lambda=4;
%{
% -------------------------------------------------------------------------
% deltai= betai-alphai
% max 1/2*sumationij deltai*yi*xi*xj'*yj*deltai + sum(deltai) +
%       lambda*(sum(alpha)-1);
% such that
%     -alpha<=delta<=c1-alpha
%     0<=alpha<=c2
%

cvx_begin
    variables a(m) be(m);
    dual variables u u1;
    minimize(0.5*quad_form((be-a),Q)-o*(be)+lambda*(o*a-1));
    subject to
        0<=be<= c1;
        0<=a<=c2;
cvx_end

wmdualcvx=((be-a).*y)'*x
%}
xt=[xt,ones(size(xt,1),1)];
%{
pred1=sign(xt*wmdualcvx');
correct = sum((pred1-yt)== 0);
accuracy = correct*100/length(yt)
sum(a)
%}
%}
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
%{
% lambda
%pred1=sign(xt*w');
% correct = sum((pred1-yt)== 0);
% accuracy = correct*100/length(yt)

% while ~converge
%     for i=1:m
%         Gd=1/2*(Q(i,:)*d1+d1(i)*Q(i,i))-1;
%         Ga=lambda-Gd-1;
% %         if a1(i)==0
% %             pga=min(Ga,0);
% %         elseif a1(i)==c2;
% %             pga=max(Ga,0);
% %         else
% %             pga=Ga;
% %         end
% %         if pga~=0
%             t=a1(i);
%             a1old(i,:)=t;
%             a1(i)=min(max(a1(i)-Ga/Q(i,i),0),c2);
% %         end   
% %         
% %         if d1(i)==-a1(i)
% %             pgb=min(Gd,0);
% %         elseif d1(i)==c1-a1(i)
% %             pgb=max(Gd,0);
% %         else
% %             pgb=Gd;
% %         end
% %         if pgb~=0
%             t=d1(i);
%             d1old(i)=t;
%             %b1(i)-Gb(i)/Q(i,i)
%             d1(i)=min(max(d1(i)-Gd/Q(i,i),-a1(i)),c1-a1(i));
%             w=w+(((d1(i)-d1old(i))*y(i))*x(i,:));
% %         end    
%     end
%     alphas=[a1old,a1];
%     deltas=[d1old,d1];
%     
%     j=j+1
%     %[abs(a1old-a1);abs(d1old-d1)]
%     changedvariable=nnz([abs(a1old-a1);abs(d1old-d1)]>=1e-5)
%     %Gdnew=Q*d1-o';
%     objnew=(0.5*(d1'*Q*d1)-o*(d1+a1)+lambda*(o*a1-1))
%     %d=max(Gdnew-Gd)
%     %d=max(Gdnew)-min(Gdnew);
%     %d=abs(obj-objnew);
%     d=1;
%     if d<=thresh || j==MAXITR || changedvariable == 0
%         converge=true;
%     end
%     obj=objnew;
% end
%}
wcd=w
%d1
%a1
sum(a1)
%}
%% svm implementation------------------------------------------------------
%{
Q=x*x';

cvx_begin
    variable a(m);
    dual variables z z1;
    minimize(0.5*quad_form(y.*a,Q)-o*a);
    subject to
        z:0<= a <= C;
        z1:y'*a == 0;
cvx_end

w1=(a.*y)'*x
z
z1

cvx_begin
    variables w(n) b q(m)
    %dual variables u u1;
    minimize( .5*w'*w + o*q);
    subject to
    y.*(x*w + b) >= o' - q;
    q>=0;
cvx_end
w
b

x=[x,ones(m,1)];

r=repmat(y,1,n+1).*x;
Q=r*r';
a1=zeros(m,1);
w=(y.*a1)'*x;
thresh=1e-10;
a1old=zeros(m,1);
G=zeros(m,1);
Gnew=zeros(m,1);
converge=false;

while ~converge
    for i=1:m
        G(i)=y(i)*w*x(i,:)'-1;
        if a1(i)==0
            pg=min([G(i),0]);
        elseif a1==C;
            pg=max([G(i),0]);
        else
            pg=G;
        end
        if pg~=0
            a1old(i)=a1(i);
            a1(i)=min([max([a1(i)-G(i)/Q(i,i),0]),C]);
            w=w+(((a1(i)-a1old(i))*y(i))'*x(i,:));
        end
        Gnew(i)=y(i)*w*x(i,:)'-1;
    end
    d=max(Gnew-G);
    if d<=thresh
        converge=true;
    end
end

wc=w
%}
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
