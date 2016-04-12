% filetr='E:\Mouly Work\Data_ML\news20b_sparse_1.train';
% filete='E:\Mouly Work\Data_ML\news20b_sparse_1.test';
% [y,x,n]=ReadSparse(filetr);
% [yt,xt,nt]=ReadSparse(filete);
% n=max(n,nt);
lambda=1;
flag=true;
prevlambda=0;
while flag  
    lambda
    asum=coordinateDecent(rsx,rsy,rsxt,rsyt,rsn,lambda);
    if abs(asum -1)<=1e-2
        flag=false;
    elseif asum > 1
        prevlambda=lambda;
        lambda=lambda+2;
    else
        
        while abs(asum -1)>=1e-2
            mid=(lambda+prevlambda)/2;
            mid
            asum=coordinateDecent(rsx,rsy,rsxt,rsyt,rsn,mid);
            if asum > 1
                prevlambda=mid;
            else
                lambda=mid;
            end
        end
    end
end