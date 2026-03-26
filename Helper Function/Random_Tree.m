function [JTreeRand, hTreeRand, H, Ents] = Random_Tree(datamean,datacorr_pseudo)


    numSpins = length(datamean);
    spins = 1:numSpins;


    JTreeRand = zeros(numSpins); % Initialize random Spin matrix
    
    hTreeRand = log(datamean./(1-datamean));


    linidx = randi((numSpins)*(numSpins-1)/2);
    p = (sqrt(1+8*linidx)-1)/2;
    idx0 = floor(p);

    row = idx0 + 1;
    col = linidx - idx0*(idx0+1)/2 + 1;

    if row == col
        col = col+1;
    end

    %Initialize first connection

    deltaS(row, col) = 0;
    deltaS(col, row) = 0;
    hTreeRand(row) = log( (datamean(row) -datacorr_pseudo(row,col))/ (1 + datacorr_pseudo(row,col) - datamean(row) -datamean(col)) );
    JTreeRand(row,col) = log( datacorr_pseudo(row,col)/ (datamean(col) - datacorr_pseudo(row,col))) - hTreeRand(row);
    JTreeRand(col, row) = JTreeRand(row, col);

    hTreeRand(col) = hTreeRand(col) + log(exp(hTreeRand(row)) + 1)- log(exp(JTreeRand(row,col) + hTreeRand(row)  )+ 1);

    connectedSpins = [row, col];
    spins(connectedSpins) = [];

    count = 1;

    Hind = sum(Entropy(datamean));
    H2 = MI2(datamean,datacorr_pseudo);
    H = Hind - H2(row,col);         
    Ents = zeros(numSpins,1);
    Ents(1) = Hind;
    Ents(2) = H;

    for i= 2:numSpins-1

        row = spins(randi(numSpins - i));
        col = connectedSpins(randi(i));

        connectedSpins = [connectedSpins, row]; %Update list of already connected spins
        spins = 1:numSpins;
        spins(connectedSpins) = [];

        deltaS(row, col) = 0;
        deltaS(col, row) = 0;
        hTreeRand(row) = log( (datamean(row) -datacorr_pseudo(row,col))/ (1 + datacorr_pseudo(row,col) - datamean(row) -datamean(col)) );
        JTreeRand(row,col) = log( datacorr_pseudo(row,col)/ (datamean(col) - datacorr_pseudo(row,col))) - hTreeRand(row);
        JTreeRand(col, row) = JTreeRand(row, col);

        hTreeRand(col) = hTreeRand(col) + log(exp(hTreeRand(row)) + 1)- log(exp(JTreeRand(row,col) + hTreeRand(row)  )+ 1);
        count = count + 1;
        H = H - H2(row,col) ; 
        Ents(i+1) = H;


    end


end