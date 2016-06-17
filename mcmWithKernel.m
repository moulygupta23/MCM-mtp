function [ accuracy ] = mcmWithKernel( x,class,xt,classt,C,gamma )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

[m,n] = size(x);
k=zeros(m,m);
for i =1:m
    for j=1:m
        k(i,j)=exp(-((x(i,:)-x(j,:))*(x(i,:)-x(j,:))')*gamma);
    end
end
disp('kernel calculated');
f= cat (2, 1, zeros(1,m+1),ones(1,m)*C);

%-----h constraint--%
cons1 = cat(2,ones(m,m+2)*-1,eye(m));
u1 = zeros(m,1);

for i=1:m
    for j=2:m+2
        if j <= m+1
            cons1(i,j) = k(i,j-1)*class(i);
        else
            cons1(i,j) = class(i);
        end
    end
end

%--- >=1 constraint--%
cons2 = cat(2,zeros(m,m+2),eye(m)*-1);
u2 = ones(m,1)*-1;

for i=1:m
    for j=2:m+2
        if j <= m+1
            cons2(i,j) = -k(i,j-1)*class(i);
        else
            cons2(i,j) = -class(i);
        end
    end
end

%---- q constraints-----%
cons3 = cat(2, zeros(m,m+2),eye(m)*-1);
u3 = zeros(m,1);

%----- all constraint matrix A---%
D = cat(1,cons1,cons2,cons3);
%-----matrix B------%
U = cat(1,u1,u2,u3);

g=linprog(f,D,U);

lambda = g(2:m+1);
b = g(m+2);
q = g(m+3:size(g,1)); 

k=zeros(m,size(classt,1));
for i =1:m
    for j=1:size(classt,1)
        k(i,j)=exp(-((x(i,:)-xt(j,:))*(x(i,:)-xt(j,:))')*gamma);
    end
end
%size(k)
%size(lambda)
pred = sign(lambda'*k +b);
%size(pred)
correct = sum((pred'-classt)== 0);

accuracy = correct/length(pred)
end
