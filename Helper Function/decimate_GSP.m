function [G, D] = decimate_GSP(G0)
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
    
    % Initialize things:
    G = G0 ~= 0;
    D = zeros(N-1,3);
    
    % Find nodes with degree <= 2:
    degs = sum(G);
    inds = intersect(find(degs <= 2), find(degs >= 1));
    
    % Loop until there are no more feasible nodes:
    counter = 1;
    
    while ~isempty(inds) ~= 0
        
        Dtemp = zeros(1,3);
        
        % Choose node to remove:
        randindx = randi(length(inds));
        i = inds(randindx); % Remove random node
    %     i = inds(1); % Remove first node in list
    %     i = inds(end); % Remove last node in list
        Dtemp(1) = i;

        
        % Neighbors of i:
        Ni = spins(G(i,:));
        Dtemp(2) = Ni(1);
        
        % Remove node from network:
        G(i,Ni) = 0;
        G(Ni,i) = 0;

        
        prevjkcon = 1;
        % If degree of i is 2 then connect two neighbors:
        if length(Ni) == 2
            Dtemp(3) = Ni(2);

            prevjkcon = G(Ni(1), Ni(2));
            G(Ni(1), Ni(2)) = 1;
            G(Ni(2), Ni(1)) = 1;
            
            
        end
        
        % Compute new feasible nodes to remove:

        degs(i) = degs(i) - 2;
        if prevjkcon == 1
            degs(Ni) = degs(Ni) - 1;
        end

        binIntersect = (degs == 1) + (degs == 2);

        inds = spins(logical(binIntersect));
        
        D(counter,:) = Dtemp;
        counter = counter + 1;
        
        
    end
    
end
