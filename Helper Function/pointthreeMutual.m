function [deltaH]  = pointthreeMutual(J, h, mean, corr, numSpins, spins, pair)
    % Calculate the DKL drop for a set of spins connecting to a pair of
    % spins

    deltaH = zeros(numSpins,1);
    
    for i = spins
        j = pair(1);
        k = pair(2);
        if i == j || i == k
            deltaH(i, ind) = 0; 
            continue
        end

        % ----- %
        m = mean([i,j,k]);
        C = corr([i,j,k],[i,j,k]);
        
        step_size = 1/2;
        [h1, J12, J13] = inverse_Ising_GSP_01_helper(m, C, step_size);

        h(i) = h1;
        Jij = J12;
        Jik = J13;


        hjold = h(j);
        hkold = h(k);
        Jjkold = J(j,k);

        hj = h(j) + log(exp(h(i))+1) - log(exp(Jij + h(i)) + 1);
        hk = h(k) + log(exp(h(i))+1) - log(exp(Jik + h(i)) + 1);
        Jjk = J(j,k) - log(exp(h(i)) + 1 ) + log( exp(Jij + h(i))+ 1) + log( exp(Jik + h(i))+ 1) - log( exp(Jij + Jik + h(i))+ 1) ;


        % ----_ %

        deltaH(i) = -log(exp(h(i))+1) + Jij*corr(i,j) + Jik*corr(i,k) + h(i)*mean(i) - mean(i)*log(mean(i))- (1-mean(i))*log(1-mean(i)) + (Jjk - Jjkold)*corr(j,k) + (hj - hjold)*mean(j) + (hk- hkold)*mean(k); 

    
    end

end
