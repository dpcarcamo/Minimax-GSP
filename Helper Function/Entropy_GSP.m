function H = Entropy_GSP(J,h, ExactMean, ExactCorr)
    
  
    Jeff = J;
    heff = h;

    [G, D] = decimate_GSP(J);

    H = sum(Entropy(ExactMean)); %Entropy Independent model
    H2 = MI2(ExactMean,ExactCorr);

    
    c = 0 ;


    for step = 1:size(D,1)

        i = D(step, 1);
        j = D(step, 2);
        k = D(step, 3);
        
        heff(j) = heff(j) - log(exp(heff(i))+1) + log(exp(Jeff(j, i)+ heff(i))+1);

        if k ~= 0
            heff(k) = heff(k) - log(exp(heff(i))+1) + log(exp(Jeff(k, i)+ heff(i))+1);
            Jeff(j,k) = Jeff(j,k) + log(exp(heff(i))+1) - log(exp(Jeff(i,j)+heff(i))+1) - log(exp(Jeff(i,k)+heff(i))+1) + log(exp(Jeff(i,j)+ Jeff(i,k)+heff(i))+1);
            Jeff(k, j) = Jeff(j, k);
        end


    end

    D = flip(D);


    dim = size(D, 1);

    for i = 1:length(h)
        c = c + log(exp(heff(i))+1);
    end

    
    for i = 1:dim  
        xi = D(i,1);
        xj = D(i,2);
        xk = D(i,3);
        if xk == 0
            H = H - H2(xi,xj);
        else
            MI3 = -log(exp(heff(xi))+1) + Jeff(xi,xj)*ExactCorr(xi,xj)+Jeff(xi,xk)*ExactCorr(xi,xk) + heff(xi)*ExactMean(xi) - ExactMean(xi)*log(ExactMean(xi))-(1-ExactMean(xi))*log(1-ExactMean(xi))+(-log(exp(heff(xi))+1) + log(exp(Jeff(xi,xj)+heff(xi))+1) + log(exp(Jeff(xi,xk)+heff(xi))+1) - log(exp(Jeff(xi,xj)+ Jeff(xi,xk)+ heff(xi))+ 1))*ExactCorr(xj,xk) + (log(exp(heff(xi))+1) - log(exp(Jeff(xi,xj)+ heff(xi))+ 1))*ExactMean(xj)+ (log(exp(heff(xi))+1) - log(exp(Jeff(xi,xk)+ heff(xi))+ 1))*ExactMean(xk);
            H = H - MI3;
        end
    end

end
