function [means, corrs, specific_heat] = glauber(J, h, kT)
    %  USING 0,1 FOR SPINS
    % NEED TO ADD THERMALIZATION

    numSpins = length(h);

    % initilize glauber with spin set with three up spins
    spin_set = zeros(numSpins, 1);
    spin_set(randperm(numel(spin_set), ceil(numSpins*0.1))) = 1;

    numIters = 2^16 ;

    means = zeros(numSpins,1);
    corrs = zeros(numSpins,numSpins);
    Energy = 0;
    Energysq = 0;
    for iter = 1 : numIters
        % Thermailze state
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
        
        means = means + spin_set;
        corrs = corrs + spin_set.*spin_set.';
        Energy = Energy +  energyIsing(spin_set, J, h);
        Energysq = Energysq + energyIsing(spin_set, J, h)^2;

    end
    means = means/numIters;
    corrs = corrs/numIters;
    Energy = Energy/numIters;
    Energysq = Energysq/numIters;
    specific_heat = (Energysq - Energy^2)/kT^2;
end