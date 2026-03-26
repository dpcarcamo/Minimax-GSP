function [J, h] = RandomGSPIsing(numSpins)
    % Function to create a randome GSP network defined in an Ising
    % interaction matrix J and external field vector h. 

    J = zeros(numSpins); % Initialize random Spin matrix
    
    h = randn(numSpins,1);
    
    J(1,2) = randn();
%     J(1,3) = randn();
%     J(2,3) = randn();

    Jcons = (J ~= 0 ); % Use this to account for nodes we "dont" add. I.e phantom nodes
    p1 = [1];
    p2 = [2];

    
    for i = 3:numSpins
        randidx = randi(length(p1));

        if randi(8) ~= -2 % Connect two nodes
            J(p1(randidx),i) = randn();
            J(p2(randidx),i) = randn();
            Jcons(p1(randidx),i) = 1;
            Jcons(p2(randidx),i) = 1; 
            p2 = [p1(randidx),p2(randidx) , p2];
            p1 = [i, i, p1];
            
            
        else % Connect to one
            if randi(2) == 2
                J(p1(randidx),i) = randn();
                Jcons(p1(randidx),i) = 1;
                Jcons(p2(randidx),i) = 1; 
                p2 = [p1(randidx),p2(randidx) , p2];
                p1 = [i, i, p1];
            else
                J(p2(randidx),i) = randn(); 
                Jcons(p1(randidx),i) = 1;
                Jcons(p2(randidx),i) = 1; 
                p2 = [p1(randidx),p2(randidx) , p2];
                p1 = [i, i, p1];
            end
        end
    end



    J = J + J.';

end
