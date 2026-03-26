function [h, J, Ent, Ents] = find_GSP_update(mean, corr)
% FIND_GSP_UPDATE
% Infers an Ising model on a Generalized Series-Parallel (GSP) graph
% using an incremental construction procedure.
%
% INPUTS
% mean : vector of single-spin means <s_i>
% corr : matrix of pairwise correlations <s_i s_j>
%
% OUTPUTS
% h    : inferred local fields
% J    : inferred pairwise couplings
% Ent  : final entropy estimate of the model
% Ents : entropy trajectory as edges are added
%
% The algorithm starts from the independent model and sequentially adds
% interactions that maximally reduce the entropy (equivalently maximize
% mutual information contributions).

    %-------------------------------------------------------------
    % Initialize entropy with the independent-spin entropy
    %-------------------------------------------------------------
    Ent = sum(Entropy(mean)); % Entropy of independent model
    Ents(1) = Ent;

    numSpins = length(mean);

    % Preallocate entropy history
    Ents = zeros(numSpins,1);
    Ents(1) = Ent;

    %-------------------------------------------------------------
    % Initialize fields assuming independent spins
    % h_i = log(p_i / (1-p_i))
    %-------------------------------------------------------------
    h = log(mean./(1-mean));

    % Initialize coupling matrices
    J = zeros(numSpins, numSpins);      % full coupling matrix
    binaryJ = zeros(numSpins,numSpins); % adjacency structure (graph)

    %-------------------------------------------------------------
    % Compute pairwise mutual informations
    %-------------------------------------------------------------
    MIs_temp = MI2(mean,corr);

    % Select pair with largest mutual information
    [M,I] = max(MIs_temp, [], 'all', 'linear');

    % Reduce entropy by that mutual information
    Ent = Ent - M;
    Ents(2) = Ent;

    % Convert linear index to spin indices
    [start, ind] = ind2sub(size(corr),I);

    %-------------------------------------------------------------
    % Solve exact two-spin inverse problem
    %-------------------------------------------------------------
    h(ind) = log((mean(ind) - corr(ind, start))./(1 + corr(ind, start) - mean(start) - mean(ind)));

    weight = log(corr(ind,start)./(mean(start) - corr(ind,start))) - h(ind);

    % Update field of the other spin
    h(start) = h(start) + log(exp(h(ind))+ 1) - log(exp(weight + h(ind))+1);

    % Store coupling
    J(start, ind) = weight;
    J(ind, start) = J(start,ind);

    % Record graph edge
    binaryJ(start, ind) = 1;
    binaryJ(ind,start) = 1;

    %-------------------------------------------------------------
    % Data structures used during greedy construction
    %-------------------------------------------------------------

    % Stores candidate pairs that define possible GSP updates
    uniquePairsfull = zeros(2*numSpins - 3,2);

    % Stores entropy reductions for candidate triples
    deltaHs = zeros(numSpins,2*numSpins - 3);

    % First connected pair
    pair = sort([start, ind]);
    connected_spins = pair;

    % Spins not yet connected
    spins = 1:numSpins;
    spins(pair) = [];

    % Compute candidate entropy reductions for adding a third spin
    [deltaH]  = pointthreeMutual(J, h, mean, corr, numSpins, spins, pair);

    deltaHs(:,1) = deltaH;

    uniquePairsfull(1,1) = pair(1);
    uniquePairsfull(1,2) = pair(2);

    %-------------------------------------------------------------
    % Main greedy loop: add one spin at a time
    %-------------------------------------------------------------
    count = 2;

    while count < numSpins

        % Select candidate update with maximum entropy reduction
        [M,I] = max(deltaHs, [],"all", "linear");

        dim = size(deltaHs);
        [i,col] = ind2sub(dim,I);

        % Pair that defines the GSP connection
        jk = uniquePairsfull(col,:);
        j = jk(1);
        k = jk(2);

        %---------------------------------------------------------
        % Solve inverse problem for 3-spin subsystem (i,j,k)
        %---------------------------------------------------------
        m = mean([i,j,k]);
        C = corr([i,j,k],[i,j,k]);

        step_size = 1;
        [h1, J12, J13] = inverse_Ising_GSP_01_helper(m, C, step_size);

        h(i) = h1;
        Jij = J12;
        Jik = J13;

        %---------------------------------------------------------
        % Insert new couplings into global model
        %---------------------------------------------------------
        J(i, j) = Jij;
        J(j, i) = Jij;

        J(i, k) = Jik;
        J(k, i) = Jik;

        binaryJ(i, j) = 1;
        binaryJ(j, i) = 1;

        binaryJ(i, k) = 1;
        binaryJ(k, i) = 1;

        %---------------------------------------------------------
        % Renormalize existing parameters after inserting node i
        %---------------------------------------------------------
        hjold = h(j);
        hkold = h(k);
        Jjkold = J(j,k);

        hj = h(j) + log(exp(h(i))+1) - log(exp(Jij + h(i)) + 1);
        hk = h(k) + log(exp(h(i))+1) - log(exp(Jik + h(i)) + 1);

        Jjk = J(j,k) ...
            - log(exp(h(i)) + 1 ) ...
            + log( exp(Jij + h(i))+ 1) ...
            + log( exp(Jik + h(i))+ 1) ...
            - log( exp(Jij + Jik + h(i))+ 1);

        h(j) = hj;
        h(k) = hk;

        J(j,k) = Jjk;
        J(k,j) = J(j,k);

        %---------------------------------------------------------
        % Update entropy
        %---------------------------------------------------------
        Ent = Ent - M; % subtract mutual information contribution
        Ents(count + 1) = Ent;


        %---------------------------------------------------------
        % Update set of connected spins
        %---------------------------------------------------------
        connected_spins = [connected_spins, i];

        spins = 1:numSpins;
        spins(connected_spins) = [];

        %---------------------------------------------------------
        % Generate new candidate GSP extensions
        %---------------------------------------------------------

        % Connect i with j
        pair = sort([i, j]);
        [deltaH]  = pointthreeMutual(J, h, mean, corr, numSpins, spins, pair);

        deltaHs(:,2*(count-1)) = deltaH;

        uniquePairsfull(2*(count-1),1) = pair(1);
        uniquePairsfull(2*(count-1),2) = pair(2);

        % Connect i with k
        pair = sort([i, k]);
        [deltaH]  = pointthreeMutual(J, h, mean, corr, numSpins, spins, pair);

        deltaHs(:,2*(count-1) + 1) = deltaH;

        uniquePairsfull(2*(count-1)+1,1) = pair(1);
        uniquePairsfull(2*(count-1)+1,2) = pair(2);

        % Prevent reusing spin i again
        deltaHs(i, :) = 0;

        count = count + 1;
    end

end





























% 
% 
% 
% 
% 
% 
% 
% 
% function [h, J,Ent, Ents] = find_GSP_update(mean, corr)
%     % Updated with new inverse calculator
%     % Updated with new change in entropy calculation
% 
%     Ent = sum(Entropy(mean)); % Entropy of independent model
%     Ents(1) = Ent;
% 
%     numSpins = length(mean);
%     Ents = zeros(numSpins,1);
%     Ents(1) = Ent;
% 
% 
%     h = log(mean./(1-mean));
%     J = zeros(numSpins, numSpins);
%     binaryJ = zeros(numSpins,numSpins);
% 
%     MIs_temp = MI2(mean,corr);
% 
%     [M,I] = max(MIs_temp, [], 'all', 'linear');
%     Ent = Ent - M;
%     Ents(2) = Ent;
%     [start, ind] = ind2sub(size(corr),I);
% 
%     h(ind) = log((mean(ind) - corr(ind, start))./(1 + corr(ind, start) - mean(start) - mean(ind)));
% 
% 
%     weight = log(corr(ind,start)./(mean(start) - corr(ind,start))) - h(ind);
% 
%     h(start) = h(start) + log(exp(h(ind))+ 1) - log(exp(weight + h(ind))+1);
% 
% 
%     J(start, ind) = weight;
%     J(ind, start) = J(start,ind);
%     binaryJ(start, ind) = 1;
%     binaryJ(ind,start) = 1;
% 
% 
% 
%     % Initilize list of added pairs
%     uniquePairsfull = zeros(2*numSpins - 3,2);
% 
%     deltaHs = zeros(numSpins,2*numSpins - 3);
% 
%     pair = sort([start, ind]);
%     connected_spins = pair;
% 
%     spins = 1:numSpins;
%     spins(pair) = [];
%     [deltaH]  = pointthreeMutual(J, h, mean, corr, numSpins, spins, pair);
% 
%     deltaHs(:,1) = deltaH; 
%     uniquePairsfull(1,1) = pair(1);
%     uniquePairsfull(1,2) = pair(2);
% 
% 
%     count = 2;
%     while count < numSpins
%         [M,I] = max(deltaHs, [],"all", "linear");
% 
%         dim = size(deltaHs);
%         [i,col] = ind2sub(dim,I);
% 
% 
%         jk = uniquePairsfull(col,:);
%         j = jk(1);
%         k = jk(2);
% 
%         %-----%
% 
%         m = mean([i,j,k]);
%         C = corr([i,j,k],[i,j,k]);
% 
%         step_size = 1;
%         [h1, J12, J13] = inverse_Ising_GSP_01_helper(m, C, step_size);
% 
%         h(i) = h1;
%         Jij = J12;
%         Jik = J13;
% 
%         %-----%
% 
%         J(i, j) = Jij;
%         J(j, i) = Jij;
%         J(i, k) = Jik;
%         J(k, i) = Jik;
%         binaryJ(i, j) = 1;
%         binaryJ(j, i) = 1;
%         binaryJ(i, k) = 1;
%         binaryJ(k, i) = 1;
% 
% 
%         hjold = h(j);
%         hkold = h(k);
%         Jjkold = J(j,k);
% 
%         hj = h(j) + log(exp(h(i))+1) - log(exp(Jij + h(i)) + 1);
%         hk = h(k) + log(exp(h(i))+1) - log(exp(Jik + h(i)) + 1);
%         Jjk = J(j,k) - log(exp(h(i)) + 1 ) + log( exp(Jij + h(i))+ 1) + log( exp(Jik + h(i))+ 1) - log( exp(Jij + Jik + h(i))+ 1) ;
% 
% 
%         h(j) = hj;
%         h(k) = hk;
%         J(j,k) = Jjk;
%         J(k,j) = J(j,k);
% 
% 
%         Ent = Ent - M; % Subtract mutual info from entropy
%         Ents(count + 1) =Ent;
%         deltaH = -log(exp(h(i))+1) + J(i,j)*corr(i,j) + J(i,k)*corr(i,k) + h(i)*mean(i) - mean(i)*log(mean(i))- (1-mean(i))*log(1-mean(i)) + (J(j,k) - Jjkold)*corr(j,k) + (h(j) - hjold)*mean(j) + (h(k)- hkold)*mean(k);
% 
% 
% 
%         connected_spins = [connected_spins, i];
% 
%         spins = 1:numSpins;
%         spins(connected_spins) = [];
% 
%         % Connect i, j
%         pair = sort([i, j]);
%         [deltaH]  = pointthreeMutual(J, h, mean, corr, numSpins, spins, pair);
% 
%         deltaHs(:,2*(count-1)) = deltaH; 
%         uniquePairsfull(2*(count-1),1) = pair(1);
%         uniquePairsfull(2*(count-1),2) = pair(2);
% 
%         % Connect i, k
%         pair = sort([i, k]);
%         [deltaH]  = pointthreeMutual(J, h, mean, corr, numSpins, spins, pair);
% 
%         deltaHs(:,2*(count-1) + 1) = deltaH; 
%         uniquePairsfull(2*(count-1)+1,1) = pair(1);
%         uniquePairsfull(2*(count-1)+1,2) = pair(2);
% 
%         deltaHs(i, :) = 0;
%         count = count + 1;
%     end
% 
% end