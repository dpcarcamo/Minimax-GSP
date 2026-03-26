function [Jguess, hguess, Ent, Ents] = FindTree(mean, corr)
    % Find the best tree
    Ent = sum(Entropy(mean)); % Entropy of independent model

    numSpins = length(mean);
    deltaS = MI2(mean, corr);
    deltaS = deltaS - diag(diag(deltaS));
    spins = 1:numSpins;
    Jguess = zeros(numSpins,numSpins);
    Ents = zeros(numSpins,1);
    Ents(1) = Ent;

    hguess = log(mean./(1-mean));
    
    dims = size(corr);

    [M,I] = max(deltaS, [], 'all', 'linear');
        
    [row, col] = ind2sub(dims,I);
    
    %Initialize first connection

    deltaS(row, col) = 0;
    deltaS(col, row) = 0;
    hguess(row) = log( (mean(row) -corr(row,col))/ (1 + corr(row,col) - mean(row) -mean(col)) );
    Jguess(row,col) = log( corr(row,col)/ (mean(col) - corr(row,col))) - hguess(row);
    Jguess(col, row) = Jguess(row, col);

    hguess(col) = hguess(col) + log(exp(hguess(row)) + 1)- log(exp(Jguess(row,col) + hguess(row)  )+ 1);
    numSpins = numSpins -2;
    connectedSpins = [row, col];
    spins = spins(spins~=row );
    spins = spins(spins~=col );
    Ent = Ent - M;
    Ents(2) = Ent;
    count = 1;
    for i= 1:numSpins
        [M, row, col, trow, tcol] = find_max_in_selected(deltaS, spins, connectedSpins); % Find maximum change in entropy
        Ent = Ent - M;
        Ents(i + 2) = Ent;
        connectedSpins = [connectedSpins, row]; %Update list of already connected spins
        spins(trow) = [];
  
        deltaS(row, col) = 0;
        deltaS(col, row) = 0;
        hguess(row) = log( (mean(row) -corr(row,col))/ (1 + corr(row,col) - mean(row) -mean(col)) );
        Jguess(row,col) = log( corr(row,col)/ (mean(col) - corr(row,col))) - hguess(row);
        Jguess(col, row) = Jguess(row, col);

        hguess(col) = hguess(col) + log(exp(hguess(row)) + 1)- log(exp(Jguess(row,col) + hguess(row)  )+ 1);
        count = count + 1;
    end
    
end
