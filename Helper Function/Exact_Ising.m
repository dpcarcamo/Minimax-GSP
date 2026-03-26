function [meanspin, corr, C,threepoint] = Exact_Ising(J, h, kT)
    %  USING 0,1 FOR SPINS

    numSpins = length(h);
   
    
    z = 0; % Partition Function
    probs = zeros(2^numSpins, 1);
    for i = 0:2^numSpins -1
    
        spin_set = decimalToBinaryVector(i, numSpins).';
        z = z + exp(-energyIsing(spin_set, J, h)/kT);

    end 
    
    meanspin = zeros(numSpins, 1);
    corr = zeros(numSpins, numSpins);
    threepoint = zeros(numSpins, numSpins,numSpins);
    Energy = 0;
    C = 0;
    for i = 1:2^numSpins
        spin_set = decimalToBinaryVector(i -1, numSpins).';
        
        probs(i) =  exp(-energyIsing(spin_set, J, h)/kT)/z;
        meanspin = meanspin + spin_set.*probs(i);
        corr =  corr + probs(i)*(spin_set)*spin_set.';

        
        % Create a 3D grid of matrices from the input vectors
        [AA, BB, CC] = meshgrid(spin_set, spin_set, spin_set);
        
        % Multiply the corresponding elements element-wise
        tensor = AA .* BB .* CC;

        threepoint = threepoint + probs(i)*tensor;
        En = energyIsing(spin_set, J, h);
        Energy = Energy + probs(i)*En;
        C = C + probs(i)*En^2;

    end 
    C = C - Energy^2;
    C = C / kT^2;
    
end