function [child, parent1, parent2, heff, Jeff] = Decimate(J, h)
    % Decimate the GSP network defined in the connection matrix J
    % Not used in correlations_GSP_01.
    numSpins = length(h);
    G = graph(J, 'upper');
    nodeNames = cellstr(num2str((1:numSpins).')); % Convert indices to cell array of strings
    G.Nodes.Name = nodeNames; % Assign node names
    Jeff = J;

    parent1 = [];
    parent2 = [];
    child = [];
    nodenum = height(G.Nodes);

    heff = h;
    while nodenum > 1
        [m, Gind] = min(G.degree);
        Gneigh = neighbors(G,Gind);
        parents = outedges(G,Gind);
        temp = G.Nodes.Name(Gind);
        ind = str2num(temp{1}); % set as labled node
        child = [child, ind];

        temp = G.Nodes.Name(Gneigh(1));
        neigh(1) = str2num(temp{1}); % set as labled neighbor
        heff(neigh(1)) = heff(neigh(1)) - log(exp(heff(ind)) + 1) + log(exp(heff(ind) + G.Edges.Weight(parents(1))) + 1);
        
        parent1 = [parent1, neigh(1)];
        if m ==2
            temp = G.Nodes.Name(Gneigh(2));
            neigh(2) = str2num(temp{1}); % set as labled neighbor
            heff(neigh(2)) = heff(neigh(2)) - log(exp(heff(ind)) + 1) + log(exp(heff(ind) + G.Edges.Weight(parents(2))) + 1);
            parent2 = [parent2, neigh(2)];
            edgeind = findedge(G,Gneigh(1),Gneigh(2));
            if edgeind == 0
                G = addedge(G,Gneigh(1),Gneigh(2), log(exp(heff(ind)) + 1 ) - log( exp(G.Edges.Weight(parents(1)) + heff(ind))+ 1)- log( exp(G.Edges.Weight(parents(2)) + heff(ind))+ 1) + log( exp(G.Edges.Weight(parents(1))+G.Edges.Weight(parents(2)) + heff(ind))+ 1) ); 
                Jeff(neigh(1), neigh(2)) = log(exp(heff(ind)) + 1 ) - log( exp(G.Edges.Weight(parents(1)) + heff(ind))+ 1)- log( exp(G.Edges.Weight(parents(2)) + heff(ind))+ 1) + log( exp(G.Edges.Weight(parents(1))+G.Edges.Weight(parents(2)) + heff(ind))+ 1);
                Jeff(neigh(2), neigh(1)) = Jeff(neigh(1), neigh(2));
            else
                weight = G.Edges.Weight(edgeind);
                Jeff(neigh(1), neigh(2)) = weight + log(exp(heff(ind)) + 1 ) - log( exp(G.Edges.Weight(parents(1)) + heff(ind))+ 1)- log( exp(G.Edges.Weight(parents(2)) + heff(ind))+ 1) + log( exp(G.Edges.Weight(parents(1))+G.Edges.Weight(parents(2)) + heff(ind))+ 1);
                Jeff(neigh(2), neigh(1)) = Jeff(neigh(1), neigh(2));
                G.Edges.Weight(edgeind) = weight + log(exp(heff(ind)) + 1 ) - log( exp(G.Edges.Weight(parents(1)) + heff(ind))+ 1)- log( exp(G.Edges.Weight(parents(2)) + heff(ind))+ 1) + log( exp(G.Edges.Weight(parents(1))+G.Edges.Weight(parents(2)) + heff(ind))+ 1) ;     
            end
        else
            parent2 = [parent2, 0];
        end

        %remove node
        G = rmnode(G, Gind);
        nodenum = nodenum - 1;

    end

    [m, Gind] = min(G.degree);
    temp = G.Nodes.Name(Gind);
    ind = str2num(temp{1}); % set as labled node
    child = [child, ind];
    parent1 = [parent1,0];
    parent2 = [parent2,0];

end
