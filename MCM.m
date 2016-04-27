addpath('liblinear-2.1/matlab');
datafiletrain = 'D:/Mouly/Data_ML/mnist38_norm_svm_full_1.train';
datafiletest = 'D:/Mouly/Data_ML/mnist38_norm_svm_full_1.test';
[y, x] = libsvmread(datafiletrain);
disp('train data loaded into memory');

[yt, xt] = libsvmread(datafiletest);
disp('test data loaded into memory');
C=1;

k=x*x';
disp('kernel calculated');
[m,n]=size(x);
tic
f=cat (2, 1, zeros(1,m+1),ones(1,m)*C);%[h,lembda,b,q]
disp('f done');
%size(repmat(y,1,m))
%size(k)

%constaints setting
r=repmat(y,1,m).*k;
%             [     h     , lembda,  b,  q   ]
cons1 = cat(2,ones(m,1)*-1,  r    ,  y,eye(m));
disp('cons1 done');
cons2 = cat(2,zeros(m,1),-r,-y,eye(m)*-1);
disp('cons2 done');
%constraint coeffient matrix A 
D = cat(1,cons1,cons2);
%constraint constant matrix B
U = [zeros(m,1);ones(m,1)*-1];
%size(D)
%size(U)

disp('optimization starts');
% Options for solving the MCM LP: we use the simplex method
options=optimset('Largescale', 'off', 'Simplex', 'on');
%  [h   ;lambda        ;b   ;q         ]
lb=[1   ;-inf*ones(m,1);-inf;zeros(m,1)];
ub=+inf*ones(m+2+m,1);
x0=[];
Aeq=[];
Beq=[];
toc
tic
g=linprog(f,D,U,Aeq,Beq,lb,ub,x0,options);
toc
lambda=g(2:m+1);
w=lambda'*x;
b=g(m+2);

nnz(lambda)
%k= x*xt';
nt=size(xt,2);
if n < nt
    xt=xt(:,1:n);
elseif nt < n
    w1=w(1:nt);
end
%disp('kernel recalculated');
%pred = sign(lambda'*k +b)';
pred= sign(w1'*xt+b);
%size(pred)
%size(yt)
%%}
correct = sum((pred-yt)== 0);

accuracy = correct/length(pred)
figure