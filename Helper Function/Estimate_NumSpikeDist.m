function [spikes, probs] = Estimate_NumSpikeDist(J, h, kT)
    %  USING 0,1 FOR SPINS
    % NEED TO ADD THERMALIZATION

    numSpins = length(h);

    % initilize glauber with spin set with three up spins
    spin_set = zeros(numSpins, 1);
    spin_set(randperm(numel(spin_set), 3)) = 1;

    numIters = 2^12 ;

    probs = zeros(1,numel(spin_set)+1);
    for iter = 1 : numIters
        iter
        for termal =  1:numel(spin_set)
            % Pick a random spin
            Index = randi(numel(spin_set));
            
            % Calculate energy change if this spin is flipped
            dE = ((2*spin_set(Index)-1)*J(Index, :)* spin_set - J(Index, Index) + h(Index)*(2*spin_set(Index)-1));
            
            % Boltzmann probability of flipping
            prob = exp(-dE / kT)/(1+exp(-dE / kT));
            
            % Spin flip condition
            if rand() <= prob
                spin_set(Index) = 1 - spin_set(Index);
            end
        end
        probs(sum(spin_set)+1) = probs(sum(spin_set)+1) + 1;
    end
    probs = probs/sum(probs);
    spikes = 0:numel(spin_set);
end
