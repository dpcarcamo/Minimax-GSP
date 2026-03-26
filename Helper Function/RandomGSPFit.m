function [J, h, H, Ents] = RandomGSPFit(datamean, datacorr)
    % Function to fit a random GSP network onto data. 


    numSpins = length(datamean);


    J = zeros(numSpins); % Initialize random Spin matrix
    
    h = log(datamean./(1-datamean));


    linidx = randi((numSpins)*(numSpins-1)/2);
    p = (sqrt(1+8*linidx)-1)/2;
    idx0 = floor(p);

    row = idx0 + 1;
    col = linidx - idx0*(idx0+1)/2 + 1;

    if row == col
        col = col+1;
    end



    connectedSpins = [row, col];
    
    h(row) = log( (datamean(row) -datacorr(row,col))/ (1 + datacorr(row,col) - datamean(row) -datamean(col)) );
    J(row,col) = log( datacorr(row,col)/ (datamean(col) - datacorr(row,col))) - h(row);
    J(col, row) = J(row, col);
    
    h(col) = h(col) + log(exp(h(row)) + 1)- log(exp(J(row,col) + h(row)  )+ 1);


    spins = 1:numSpins;
    spins(connectedSpins) = [];

    Hind = sum(Entropy(datamean));
    H2 = MI2(datamean([row,col]),datacorr([row,col],[row,col]));
    H = Hind - H2(1,2);
    Ents(1) = Hind;
    Ents(2) = H;


    addedpairs = zeros(2*numSpins-3,2);
    addedpairs(1,1) = row;
    addedpairs(1,2) = col;

    for step = 3:numSpins 

        randidx = randi(2*step-5);

        i = spins(randi(numSpins-step+1));

        j = addedpairs(randidx,1);
        k = addedpairs(randidx,2);

        addedpairs(2*step-4,1) = i;
        addedpairs(2*step-4,2) = j;
        addedpairs(2*step-3,1) = i;
        addedpairs(2*step-3,2) = k;
        
        connectedSpins = [connectedSpins, i]; %Update list of already connected spins
    
            
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
    

        spins = 1:numSpins;
        spins(connectedSpins) = [];
    

        
        H = H - deltaH;  
        Ents(step) = H;

    end
    

end
