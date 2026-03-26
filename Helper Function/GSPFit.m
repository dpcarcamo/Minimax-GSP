function [J, h, H] = GSPFit(datamean, datacorr, topology)
    % Function to fit a GSP network onto data given a topology. 

    topology = topology ~= 0;
    numSpins = length(datamean);

    [G,D] = decimate_GSP(topology);

    D = D(1:numSpins-1,:);
    D = flip(D);
    J = zeros(numSpins); % Initialize random Spin matrix
    
    h = log(datamean./(1-datamean));


    row = D(1,1);
    col = D(1,2);

    if row == col
        col = col+1;
    end

    
    h(row) = log( (datamean(row) -datacorr(row,col))/ (1 + datacorr(row,col) - datamean(row) -datamean(col)) );
    J(row,col) = log( datacorr(row,col)/ (datamean(col) - datacorr(row,col))) - h(row);
    J(col, row) = J(row, col);
    
    h(col) = h(col) + log(exp(h(row)) + 1)- log(exp(J(row,col) + h(row)  )+ 1);


    Hind = sum(Entropy(datamean));
    H2 = MI2(datamean([row,col]),datacorr([row,col],[row,col]));
    H = Hind - H2(1,2);



    for step = 2:numSpins-1


        i = D(step,1);

        j = D(step,2);
        k = D(step,3);

        %-----%
    
        m = datamean([i,j,k]);
        C = datacorr([i,j,k],[i,j,k]);
        
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
    
        deltaH = -log(exp(h(i))+1) + J(i,j)*datacorr(i,j) + J(i,k)*datacorr(i,k) + h(i)*datamean(i) - datamean(i)*log(datamean(i)) - (1-datamean(i))*log(1-datamean(i)) + (J(j,k) - Jjkold)*datacorr(j,k) + (h(j) - hjold)*datamean(j) + (h(k)- hkold)*datamean(k);
    

        
        H = H - deltaH;  

    end
    

end
