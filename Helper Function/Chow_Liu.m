function [threepoint] = Chow_Liu(i, j, k, mean, corr)
    % Aproximate a triangle by its subtree that maximizes the mutual info

    spins = [i, j, k];
    MI = MI2(mean(spins),corr(spins,spins));
    MI = triu(MI);
    MI = MI + tril(ones(3));
    [M, I] = min(MI, [], 'all', 'linear');
    [row, col] = ind2sub(size(MI),I);
    MI(row, col) = 0;
    MI = triu(MI) + triu(MI).' - 2*diag(diag(ones(3)));

    [leaves, node] = Findleaves(MI); % Find the remaining tree

    [GC,GR] = groupcounts([leaves, node]');
    if length(GR) == 3

        for iter = 1:3
            if GC(iter) == 2
                first = spins(mod(GR(iter) -2,3) + 1);
                middle = spins(GR(iter));
                last = spins(mod(GR(iter), 3) + 1) ;
            end
        end
     
    
        prob1 = corr(first, middle)*corr(middle, last)/mean(middle)
    elseif length(GR) == 2
        first = spins(GC(1));
        last = spins(GC(2));
        if 1 ~= GC(1) || 1 ~= GC(2)
            prob1 = corr(first,last)*mean(spins(1));
        end
        if 2 ~= GC(1) || 2 ~= GC(2)
            prob1 = corr(first,last)*mean(spins(2));
        end
        if 3 ~= GC(1) || 3 ~= GC(2)
            prob1 = corr(first,last)*mean(spins(3));
        end
    else 
        prob1 = mean(spins(1))*mean(spins(2))*mean(spins(3));
    end
    threepoint = zeros(3,3,3);

    threepoint(1,2,3) = prob1;
    threepoint(1,3,2) = threepoint(1,2,3);
    threepoint(2,1,3) = threepoint(1,2,3);
    threepoint(2,3,1) = threepoint(1,2,3);
    threepoint(3,1,2) = threepoint(1,2,3);
    threepoint(3,2,1) = threepoint(1,2,3);
    threepoint(1,1,1) = mean(spins(1));
    threepoint(2,2,2) = mean(spins(2));
    threepoint(3,3,3) = mean(spins(3));
    threepoint(1,2,2) = corr(spins(1),spins(2));
    threepoint(2,1,2) = threepoint(1,2,2);
    threepoint(2,2,1) = threepoint(1,2,2);
    threepoint(2,1,1) = threepoint(1,2,2);
    threepoint(1,1,2) = threepoint(1,2,2);
    threepoint(1,2,1) = threepoint(1,2,2);

    threepoint(1,3,3) = corr(spins(1),spins(3));
    threepoint(3,1,3) = threepoint(1,3,3);
    threepoint(3,3,1) = threepoint(1,3,3);
    threepoint(3,1,1) = threepoint(1,3,3);
    threepoint(1,1,3) = threepoint(1,3,3);
    threepoint(1,3,1) = threepoint(1,3,3);

    threepoint(3,2,2) = corr(spins(2),spins(3));
    threepoint(2,3,2) = threepoint(3,2,2);
    threepoint(2,2,3) = threepoint(3,2,2);
    threepoint(2,3,3) = threepoint(3,2,2);
    threepoint(3,3,2) = threepoint(3,2,2);
    threepoint(3,2,3) = threepoint(3,2,2);
end

