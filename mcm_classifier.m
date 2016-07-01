function [ NSV,accuracy ] = mcm_classifier(x,y,xt,yt,C)
%Basic MCM classifier you have to pass x=trainx , y = trainy ,xt=trainxt ,yt=trainyt and the value of c
%It will return the number od SV and the accuracy 
tic
[m,n] = size(x);
%{
for i = 1:n
    D(i) = mean(x(:,i));
    s(i) = std(x(:,i));
    x(:,i) = (x(:,i)-D(i));
    xt(:,i) = xt(:,i)-D(i);
end
%}
% %Normailze the data
xm = mean(x);
xs = std(x);
xs(xs==0)=1;
% 
x=(x-repmat(xm,size(x,1),1))./repmat(xs,size(x,1),1);
xt=(xt-repmat(xm,size(xt,1),1))./repmat(xs,size(xt,1),1);

%kernel calculation
k=x*x';
disp('kernel calculated');

%coefficients of valiables to be optimized
f=cat (2, 1, zeros(1,m+1),ones(1,m)*C);%[h,lembda,b,q]

%size(repmat(y,1,m))
%size(k)

%constaints setting
r=repmat(y,1,m).*k;
%             [     h     , lembda,  b,  q   ]
cons1 = cat(2,ones(m,1)*-1,  r    ,  y,eye(m));
cons2 = cat(2,zeros(m,1),-r,-y,eye(m)*-1);

%{
f= cat (2, 1, zeros(1,n+1),ones(1,m)*C);%[h,w,b,q]

%-----h constraint--%
cons1 = cat(2,ones(m,n+2)*-1,eye(m));
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
cons2 = cat(2,zeros(m,n+2),eye(m)*-1);
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

%---- q constraints-----%
cons3 = cat(2, zeros(m,n+2),eye(m)*-1);
u3 = zeros(m,1);

%----- all constraint matrix A---%
D = cat(1,cons1,cons2,cons3);
%-----matrix B------%
U = cat(1,u1,u2,u3);
options=optimset('Largescale', 'off', 'Simplex', 'on');
g=linprog(f,D,U,[],[],[0;-inf*ones(n+1,1);zeros(m,1)],+inf*ones(n+2+m,1),[],options);
%}
%%{
%constraint coeffient matrix A 
D = cat(1,cons1,cons2);
%constraint constant matrix B
U = [zeros(m,1);ones(m,1)*-1];
%size(D)
%size(U)
disp('optimization starts');
% Options for solving the MCM LP: we use the simplex method
options=optimset('Largescale', 'off', 'simplex', 'on');
%  [h   ;lambda        ;b   ;q         ]
lb=[1   ;-inf*ones(m,1);-inf;zeros(m,1)];
ub=+inf*ones(m+2+m,1);
x0=[];
Aeq=[];
Beq=[];
g=linprog(f,D,U,Aeq,Beq,lb,ub,x0,options);
%%}
%{
w = g(2:n+1)
b = g(n+2)
q = g(n+3:size(g,1)); 

%-------------------when test and train are same
%     xt=x;
%     yt=y;
%--------------- when and train are different load the test set here
%and comment above

pred = sign(xt*w +b);
%}
%%{
toc
lambda=g(2:m+1);
w=lambda'*x;
b=g(m+2);
%dlmwrite('mymcm.txt',g);

NSV=nnz(lambda)
%k= x*xt';
nt=size(xt,2);
w1=w;
if n < nt
    xt=xt(:,1:n);
elseif nt < n
    w1=[w(1:nt),w(n)];
end
disp('kernel recalculated');
% pred = sign(lambda'*k +b)';
size(w1)
size(xt)
pred= sign(w1*xt'+b)';
%size(pred)
%size(yt)
%%}
correct = sum((pred-yt)== 0);

accuracy = correct*100/length(pred)
toc
end

