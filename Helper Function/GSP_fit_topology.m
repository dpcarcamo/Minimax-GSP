function [J, h, H] = GSP_fit_topology(Mean,Corr, topo)


    numSpins = length(Mean);


    [~,D] = decimate_GSP(topo);

    D = flip(D);

    
    J = zeros(numSpins,numSpins);

    h = log(Mean./(1-Mean));



    row = D(1,1);
    col = D(1,2);

    %Initialize first connection


    h(row) = log( (Mean(row) -Corr(row,col))/ (1 + Corr(row,col) - Mean(row) -Mean(col)) );
    J(row,col) = log( Corr(row,col)/ (Mean(col) - Corr(row,col))) - h(row);
    J(col, row) = J(row, col);

    h(col) = h(col) + log(exp(h(row)) + 1)- log(exp(J(row,col) + h(row)  )+ 1);



    Hind = sum(Entropy(Mean));
    H2 = MI2(Mean([row,col]),Corr([row,col],[row,col]));
    
    H = Hind - H2(1,2);         

    
    for step = 2:numSpins -1
       

        
        k = D(step,3);
        j = D(step,2);
        i = D(step,1);


        %-----%

        m = Mean([i,j,k]);
        C = Corr([i,j,k],[i,j,k]);

        step_size = 1;
        [h1, J12, J13] = inverse_Ising_GSP_01_helper(m, C, step_size);

        h(i) = h1;
        Jij = J12;
        Jik = J13;

        %-----%

        J(i, j) = Jij;
        J(j, i) = Jij;
        J(i, k) = Jik;
        J(k, i) = Jik;


        hjold = h(j);
        hkold = h(k);
        Jjkold = J(j,k);

        hj = h(j) + log(exp(h(i))+1) - log(exp(Jij + h(i)) + 1);
        hk = h(k) + log(exp(h(i))+1) - log(exp(Jik + h(i)) + 1);
        Jjk = J(j,k) - log(exp(h(i)) + 1 ) + log( exp(Jij + h(i))+ 1) + log( exp(Jik + h(i))+ 1) - log( exp(Jij + Jik + h(i))+ 1) ;


        h(j) = hj;
        h(k) = hk;
        J(j,k) = Jjk;
        J(k,j) = J(j,k);

        deltaH = -log(exp(h(i))+1) + Jij*Corr(i,j) + Jik*Corr(i,k) + h(i)*Mean(i) - Mean(i)*log(Mean(i))- (1-Mean(i))*log(1-Mean(i)) + (Jjk - Jjkold)*Corr(j,k) + (hj - hjold)*Mean(j) + (hk- hkold)*Mean(k);


        H = H - deltaH(1)  ;
    end


end 