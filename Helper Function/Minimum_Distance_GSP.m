function [JGSPdist, hGSPdist, H] = Minimum_Distance_GSP(datamean,datacorr_pseudo,Locations)

    distanceMetric = squareform(pdist(Locations));

    numSpins = size(distanceMetric,1);
    spins = 1:numSpins;

    distanceMetric = distanceMetric + diag(inf*ones(numSpins,1));

    JGSPdist = zeros(numSpins,numSpins);

    hGSPdist = log(datamean./(1-datamean));

    dims = size(distanceMetric);

    deltaS = 1./distanceMetric;

    deltaS = deltaS - diag(diag(deltaS));

    [M,I] = max(deltaS, [], 'all', 'linear');

    [row, col] = ind2sub(dims,I);

    %Initialize first connection

    deltaS(row, col) = 0;
    deltaS(col, row) = 0;
    hGSPdist(row) = log( (datamean(row) -datacorr_pseudo(row,col))/ (1 + datacorr_pseudo(row,col) - datamean(row) -datamean(col)) );
    JGSPdist(row,col) = log( datacorr_pseudo(row,col)/ (datamean(col) - datacorr_pseudo(row,col))) - hGSPdist(row);
    JGSPdist(col, row) = JGSPdist(row, col);

    hGSPdist(col) = hGSPdist(col) + log(exp(hGSPdist(row)) + 1)- log(exp(JGSPdist(row,col) + hGSPdist(row)  )+ 1);
    connectedSpins = [row, col];

    count = 1;

    Hind = sum(Entropy(datamean));
    H2 = MI2(datamean([row,col]),datacorr_pseudo([row,col],[row,col]));
    H = Hind - H2(1,2);         

    Addedpairs = zeros(2*numSpins - 3, 2); 
    Addedpairs(1,1) = col;
    Addedpairs(1,2) = row;
    distanceTri = zeros(numSpins, 2*numSpins - 3);

    spins(connectedSpins) = [];
    for spin = spins
        distanceTri(spin, 1) = 1./(distanceMetric(spin, row) + distanceMetric(spin, col));
    end


    for step = 1:numSpins -2

        [M,I] = max(distanceTri, [],"all", "linear");

        dim = size(distanceTri);
        [i,col] = ind2sub(dim,I);

        jk = Addedpairs(col,:);
        j = jk(1);
        k = jk(2);


        connectedSpins = [connectedSpins, i]; %Update list of already connected spins



        %-----%

        m = datamean([i,j,k]);
        C = datacorr_pseudo([i,j,k],[i,j,k]);

        step_size = 1;
        [h1, J12, J13] = inverse_Ising_GSP_01_helper(m, C, step_size);

        hGSPdist(i) = h1;
        Jij = J12;
        Jik = J13;

        %-----%

        JGSPdist(i, j) = Jij;
        JGSPdist(j, i) = Jij;
        JGSPdist(i, k) = Jik;
        JGSPdist(k, i) = Jik;


        hjold = hGSPdist(j);
        hkold = hGSPdist(k);
        Jjkold = JGSPdist(j,k);

        hj = hGSPdist(j) + log(exp(hGSPdist(i))+1) - log(exp(Jij + hGSPdist(i)) + 1);
        hk = hGSPdist(k) + log(exp(hGSPdist(i))+1) - log(exp(Jik + hGSPdist(i)) + 1);
        Jjk = JGSPdist(j,k) - log(exp(hGSPdist(i)) + 1 ) + log( exp(Jij + hGSPdist(i))+ 1) + log( exp(Jik + hGSPdist(i))+ 1) - log( exp(Jij + Jik + hGSPdist(i))+ 1) ;


        hGSPdist(j) = hj;
        hGSPdist(k) = hk;
        JGSPdist(j,k) = Jjk;
        JGSPdist(k,j) = JGSPdist(j,k);

        deltaH = -log(exp(hGSPdist(i))+1) + JGSPdist(i,j)*datacorr_pseudo(i,j) + JGSPdist(i,k)*datacorr_pseudo(i,k) + hGSPdist(i)*datamean(i) - datamean(i)*log(datamean(i))- (1-datamean(i))*log(1-datamean(i)) + (JGSPdist(j,k) - Jjkold)*datacorr_pseudo(j,k) + (hGSPdist(j) - hjold)*datamean(j) + (hGSPdist(k)- hkold)*datamean(k);

        pair = sort([i, j]);
        Addedpairs(2*step,1) = pair(1);
        Addedpairs(2*step,2) = pair(2);
        pair = sort([i, k]);
        Addedpairs(2*step+1,1) = pair(1);
        Addedpairs(2*step+1,2) = pair(2);

        spins = 1:numSpins;
        spins(connectedSpins) = [];
        for spin = spins
            distanceTri(spin, 2*step) = 1./(distanceMetric(spin, i) + distanceMetric(spin, j));
            distanceTri(spin, 2*step+1) = 1./(distanceMetric(spin, i) + distanceMetric(spin, k));
        end


        distanceTri(i, :) = 0;
        H = H - deltaH  ;
    end


end 