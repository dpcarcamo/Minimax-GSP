function [JTreedist, hTreedist, H] = Minimum_Distance_Tree(datamean,datacorr_pseudo,Locations)

    distanceMetric = squareform(pdist(Locations));

    numSpins = size(distanceMetric,1);
    spins = 1:numSpins;

    distanceMetric = distanceMetric - diag(inf*ones(numSpins,1));

    JTreedist = zeros(numSpins,numSpins);

    hTreedist = log(datamean./(1-datamean));

    dims = size(distanceMetric);

    deltaS = 1./distanceMetric;

    deltaS = deltaS - diag(diag(deltaS));

    [M,I] = max(deltaS, [], 'all', 'linear');

    [row, col] = ind2sub(dims,I);

    %Initialize first connection

    deltaS(row, col) = 0;
    deltaS(col, row) = 0;
    hTreedist(row) = log( (datamean(row) -datacorr_pseudo(row,col))/ (1 + datacorr_pseudo(row,col) - datamean(row) -datamean(col)) );
    JTreedist(row,col) = log( datacorr_pseudo(row,col)/ (datamean(col) - datacorr_pseudo(row,col))) - hTreedist(row);
    JTreedist(col, row) = JTreedist(row, col);

    hTreedist(col) = hTreedist(col) + log(exp(hTreedist(row)) + 1)- log(exp(JTreedist(row,col) + hTreedist(row)  )+ 1);
    numSpins = numSpins -2;
    connectedSpins = [row, col];
    spins = spins(spins~=row );
    spins = spins(spins~=col );

    count = 1;

    Hind = sum(Entropy(datamean));
    H2 = MI2(datamean,datacorr_pseudo);
    H = Hind - H2(row,col);            

    for i= 1:numSpins
        [M, row, col, trow, tcol] = find_max_in_selected(deltaS, spins, connectedSpins); % Find maximum change in entropy
        connectedSpins = [connectedSpins, row]; %Update list of already connected spins
        spins(trow) = [];

        deltaS(row, col) = 0;
        deltaS(col, row) = 0;
        hTreedist(row) = log( (datamean(row) -datacorr_pseudo(row,col))/ (1 + datacorr_pseudo(row,col) - datamean(row) -datamean(col)) );
        JTreedist(row,col) = log( datacorr_pseudo(row,col)/ (datamean(col) - datacorr_pseudo(row,col))) - hTreedist(row);
        JTreedist(col, row) = JTreedist(row, col);

        hTreedist(col) = hTreedist(col) + log(exp(hTreedist(row)) + 1)- log(exp(JTreedist(row,col) + hTreedist(row)  )+ 1);
        count = count + 1;
        H = H - H2(row,col) ; 
    end


end
