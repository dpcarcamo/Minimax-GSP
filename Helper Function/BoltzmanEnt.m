function [entropy] = BoltzmanEnt(J, h, kT)
    %  USING 0,1 FOR SPINS

    numSpins = length(h);
   
    
    z = 0; % Partition Function
    probs = zeros(2^numSpins, 1);
    for i = 0:2^numSpins -1
    
        spin_set = decimalToBinaryVector(i, numSpins).';
        z = z + exp(-energyIsing(spin_set, J, h)/kT);

    end 
    
    entropy = 0;
    for i = 1:2^numSpins
        spin_set = decimalToBinaryVector(i -1, numSpins).';
        
        probs(i) =  exp(-energyIsing(spin_set, J, h)/kT)/z;

        entropy = entropy -  probs(i)*log(probs(i));
    end

    
    
end