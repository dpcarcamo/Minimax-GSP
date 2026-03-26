
function [G, D, Jeff, heff] = decimate_GSP_carefully(G0, h, keep)
    % Input: nxn unweighted adjacency matrix G0 representing a GSP network.
    %
    % Output: Randomly remove nodes of degree 1 or 2. If we remove a node of
    % degree 2 then we place a new edge between the two neighbors of the
    % removed node. Continue this process until no more nodes of degree 1 or 2
    % exist and return the final adjacency matrix G. Also return the order of
    % decimations in the numDec x 3 matrix D, where D(t,1) is the node removed
    % at step t and D(t,2) and D(t,3) are the two nodes from which the node was
    % removed. We note that D(t,3) = 0 if the removed node had degree 1.
    
    % Size of network:
    N = size(G0,1);
    spins = 1:N;

    heff = h;
    Jeff = G0;
    
    % Initialize things:
    G = G0 ~= 0;
    D = zeros(N-1,3);
    
    % Find nodes with degree <= 2:
    degs = sum(G);
    inds = intersect(find(degs <= 2), find(degs >= 1));
    for ig = 1:length(keep)
        inds(inds==keep(ig)) = [];
    end
    
    % Loop until there are no more feasible nodes:
    counter = 1;
    
    while ~isempty(inds) ~= 0
        
        Dtemp = zeros(1,3);
        
        % Choose node to remove:
        i = inds(randi(length(inds))); % Remove random node
    %     i = inds(1); % Remove first node in list
    %     i = inds(end); % Remove last node in list
        Dtemp(1) = i;
        
        % Neighbors of i:
        Ni = spins(G(i,:));
        Dtemp(2) = Ni(1);
        
        % Remove node from network:
        G(i,Ni) = 0;
        G(Ni,i) = 0;

        heff(Ni(1)) = heff(Ni(1)) - log(exp(heff(i)) + 1) + log(exp(heff(i) + Jeff(i, Ni(1))) + 1);

        prevjkcon = 1;
        % If degree of i is 2 then connect two neighbors:
        if length(Ni) == 2
            Dtemp(3) = Ni(2);
            
            G(Ni(1), Ni(2)) = 1;
            G(Ni(2), Ni(1)) = 1;
            heff(Ni(2)) = heff(Ni(2)) - log(exp(heff(i)) + 1) + log(exp(heff(i) + Jeff(i, Ni(2))) + 1);
            Jeff(Ni(1), Ni(2)) = Jeff(Ni(1),Ni(2)) + log(exp(heff(i)) + 1 ) - log( exp(Jeff(i, Ni(1)) + heff(i))+ 1)- log( exp(Jeff(i, Ni(2)) + heff(i))+ 1) + log( exp(Jeff(i, Ni(1)) + Jeff(i, Ni(2))+ heff(i))+ 1);
            Jeff(Ni(2), Ni(1)) = Jeff(Ni(1),Ni(2));
        end
        
        % Remove node from network:
        Jeff(i,Ni) = 0;
        Jeff(Ni,i) = 0;

        % Compute new feasible nodes to remove:

        degs(i) = degs(i) - 2;
        if prevjkcon == 1
            degs(Ni) = degs(Ni) - 1;
        end

        binIntersect = (degs == 1) + (degs == 2);

        inds = spins(logical(binIntersect));

        % for ig = 1:length(keep)
        %     inds(inds==keep(ig)) = [];
        % end
        inds = setdiff(inds,keep);
        
        D(counter,:) = Dtemp;
        counter = counter + 1;
        
    end
end

