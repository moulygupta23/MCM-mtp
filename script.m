% filetr='D:\Mouly\Data_ML\news20b_sparse_1.train';
% filete='D:\Mouly\Data_ML\news20b_sparse_1.test';
% [y,x,n]=ReadSparse(filetr);
% [yt,xt,nt]=ReadSparse(filete);
% n=max(n,nt);
lambda=3.7;
flag=true;
prevlambda=5;
while flag  
    lambda
    %coordinateDecent1;%(rsx,rsy,rsxt,rsyt,rsn,lambda);
    asum=sumOfAlpha;
    if abs(asum -1)<=1e-2
        flag=false;
%     elseif asum > 1
%         prevlambda=lambda;
%         lambda=lambda+2;
    else
        
        while abs(asum -1)>=1e-2
            mid=(lambda+prevlambda)/2;
            l1=lambda;
            lambda=mid
            coordinateDecent1;%(rsx,rsy,rsxt,rsyt,rsn,lambda);
            asum=sumOfAlpha;
            if asum > 1
                lambda=l1;
                prevlambda=mid;
            else
                lambda=mid;
            end
        end
    end
end