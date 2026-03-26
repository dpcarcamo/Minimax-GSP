function [m, C, X, Z, Trip, Jcon, dC_dh, dm_dh, dh_dh,dJ_dh,dh_dJ,Jpair, pairs] = correlations_GSP_03(J, h, m, C, Trip, dC_dh, dm_dh, dh_dh,dJ_dh,dh_dJ,Jpair, keep )
    % Updated to be more memory efficient for all derivitives 11/3/23
    % Inputs: nxn matirx J and nx1 col vector h of external fields for a system
    % with 0,1 variables. We require that the interaction matrix J represents a
    % GSP network. We also take as input the order of node decimations (as
    % output by 'decimate_GSP'), where decimations(t,1) is the t^th node to be 
    % decimated and decimations(t,[2,3]) are its two neighbors. We note that
    % decimations(t,3) = 0 if the t^th node only has one neighbor.
    %
    % Output: nx1 vector m of magnetizations, calculated in two steps: (i)
    % decimate nodes of degree 1 or 2 down to a single node, and (ii)
    % recursivley calculate the magnetizations back out in reverse order. We
    % also compute the correlations C(i,j) between all nodes i and j and the
    % partition function Z. See Report 2 for details.
    
    % NOTE: The difference between this function and 'correlations_GSP' is that
    % here consider systems with 0,1 variables.
    
    % Number of nodes:
    n = length(h);


    bJ = tril(J) ~= 0 ;
    [x,y] = find(bJ);
    
    Jcon = zeros(size(J));
    for i = 1:length(x)
    
        Jcon(x(i), y(i)) = i;
    
        Jcon(y(i), x(i)) = i;
    end

    connum = length(x);
    pairs = [x,y];
    
    
    %Compute decimation order:
    [Jdec, D1, Jeff, heff] = decimate_GSP_carefully(J, h, keep);

    % h = heff;
    % J = Jeff;
    Nleft = sum(sum(Jdec) ~=0) ;
    
    [Jdec, D] = decimate_GSP(Jeff);
    
    D = D+ flip(D1);
    D = flip(D);

    % Initialize effective parameters:
    J_eff = J;
    h_eff = h;
    c = 0;
    
    % Loop through node decimations:
    for t = 1:size(D,1)
        
        % Node to decimate and its neighbors:
        i = D(t,1);
        j = D(t,2);
        k = D(t,3);
        
        % Update effective parameters on first neighbor:
        c = c + log(exp(h_eff(i)) + 1);
        h_eff(j) = h_eff(j) - log(exp(h_eff(i)) + 1) + log(exp(J_eff(i,j) + h_eff(i)) + 1);
        
        % If node i has two neighbors:
        if k ~= 0
            
            h_eff(k) = h_eff(k) - log(exp(h_eff(i)) + 1) + log(exp(J_eff(i,k) + h_eff(i)) + 1);
            J_eff(j,k) = J_eff(j,k) + log(exp(h_eff(i)) + 1) - log(exp(J_eff(i,j) + h_eff(i)) + 1)...
                - log(exp(J_eff(i,k) + h_eff(i)) + 1) + log(exp(J_eff(i,j) + J_eff(i,k) + h_eff(i)) + 1);
            J_eff(k,j) = J_eff(j,k);
            
        end
    end
    
    % Things we are going to need to compute:
    m = zeros(n,1); % Magnetizations
    C = zeros(n); % Correlations between nodes that interact
    dm_dh = zeros(n); % Susceptibilities
    dC_dh = zeros(connum,n); % Derivatives of correlations with respect to external fields
    dh_dh = eye(n); % Derivatives of external fields with respect to external fields
    dJ_dh = zeros(connum,n); % Derivatives of interactions with respect to external fields
    dh_dJ = zeros(n,connum); % Derivatives of external fields with respect to interactions
    Jpair = zeros(connum);
    %dJ_dJ = zeros(n,n,n,n); % Derivatives of interactions with respect to interactions
    %Trip = zeros(connum,n); %Triplets

    % Make self-derivatives unity:
    for ind1 = 1:n
        for ind2 = 1:n
            if Jcon(ind1,ind2) ~= 0
                Jpair(Jcon(ind1,ind2),Jcon(ind1,ind2)) = 1;
                Jpair(Jcon(ind2,ind1),Jcon(ind1,ind2)) = 1;
                Jpair(Jcon(ind1,ind2),Jcon(ind2,ind1)) = 1;
                Jpair(Jcon(ind2,ind1),Jcon(ind2,ind1)) = 1;
            end
%             dJ_dJ(ind1,ind2,ind1,ind2) = 1;
%             dJ_dJ(ind1,ind2,ind2,ind1) = 1;
%             dJ_dJ(ind2,ind1,ind1,ind2) = 1;
%             dJ_dJ(ind2,ind1,ind2,ind1) = 1;
        end
    end
    
    % Compute things for final node:
    i0 = j ;%child(end);
    m(i0) = 1/(1 + exp(-h_eff(i0)));
    C(i0,i0) = m(i0);
    dm_dh(i0,i0) = exp(-h_eff(i0))/(1 + exp(-h_eff(i0)))^2;
    
    % Compute partition function:
    Z = exp(c)*(exp(h_eff(i0)) + 1);
    
    % Loop over nodes in reverse order from which they were decimated:
    for t_j = size(D,1):-1:1
        
        % Node to take derivative with respect to and its decimation neighbors:
        j1 = D(t_j,1);
        j2 = D(t_j,2);
        j3 = D(t_j,3);
        
        % Compute magnetizations and derivatives:
        
        % If j1 only has one neighbor:
        if j3 == 0
            
            % Compute magnetizations and correlations:
            m(j1) = (1 - m(j2))/(1 + exp(-h_eff(j1))) + m(j2)/(1 + exp(-J_eff(j1,j2) - h_eff(j1)));
            C(j1,j1) = m(j1);
            
            C(j1,j2) = m(j2)/(1 + exp(-J_eff(j1,j2) - h_eff(j1)));
            C(j2,j1) = C(j1,j2);
            
            % Compute derivatives of h(j2) with respect to h(j1) and J(j1,j2):
            dh_dh(j2,j1) = -1/(1 + exp(-h_eff(j1))) + 1/(1 + exp(-J_eff(j1,j2) - h_eff(j1)));
            dh_dJ(j2,Jcon(j1,j2)) = 1/(1 + exp(-J_eff(j1,j2) - h_eff(j1)));
            dh_dJ(j2,Jcon(j2,j1)) = dh_dJ(j2,Jcon(j1,j2));
            
            % Compute derivative of external field of final node with respect
            % to h(j1) and J(j1,j2). Dependence goes through h(j2):
            dh_dh(i0,j1) = dh_dh(i0,j2)*dh_dh(j2,j1);
            dh_dJ(i0,Jcon(j1,j2)) = dh_dh(i0,j2)*dh_dJ(j2,Jcon(j1,j2));
            dh_dJ(i0,Jcon(j2,j1)) = dh_dJ(i0,Jcon(j1,j2));
            
            % Loop over nodes between final node and j1:
            for t_i = size(D,1):-1:(t_j + 1)
                
                % Node to take derivative of and its decimation neighbors:
                i1 = D(t_i,1);
                i2 = D(t_i,2);
                i3 = D(t_i,3);
                
                % Compute derivatives of h(i1), J(i1,i2), and J(i1,i3) with
                % respect to h(j1) and J(j1,j2). All dependencies go through h(j2):
                dh_dh(i1,j1) = dh_dh(i1,j2)*dh_dh(j2,j1);
                dh_dJ(i1,Jcon(j1,j2)) = dh_dh(i1,j2)*dh_dJ(j2,Jcon(j1,j2));
                dh_dJ(i1,Jcon(j2,j1)) = dh_dJ(i1,Jcon(j1,j2));
                
                dJ_dh(Jcon(i1,i2),j1) = dJ_dh(Jcon(i1,i2),j2)*dh_dh(j2,j1);
                dJ_dh(Jcon(i2,i1),j1) = dJ_dh(Jcon(i1,i2),j1);


                Jpair(Jcon(i1,i2),Jcon(j1,j2)) = dJ_dh(Jcon(i1,i2),j2)*dh_dJ(j2,Jcon(j1,j2));
                Jpair(Jcon(i2,i1),Jcon(j1,j2)) = Jpair(Jcon(i1,i2),Jcon(j1,j2));
                Jpair(Jcon(i1,i2),Jcon(j2,j1)) = Jpair(Jcon(i1,i2),Jcon(j1,j2));
                Jpair(Jcon(i2,i1),Jcon(j2,j1)) = Jpair(Jcon(i1,i2),Jcon(j1,j2));

%                 dJ_dJ(i1,i2,j1,j2) = dJ_dh(i1,i2,j2)*dh_dJ(j2,j1,j2);
%                 dJ_dJ(i2,i1,j1,j2) = dJ_dJ(i1,i2,j1,j2);
%                 dJ_dJ(i1,i2,j2,j1) = dJ_dJ(i1,i2,j1,j2);
%                 dJ_dJ(i2,i1,j2,j1) = dJ_dJ(i1,i2,j1,j2);
%                 
                % If i1 has two neighbors:
                if i3 ~= 0
                    
                    dJ_dh(Jcon(i1,i3),j1) = dJ_dh(Jcon(i1,i3),j2)*dh_dh(j2,j1);
                    dJ_dh(Jcon(i3,i1),j1) = dJ_dh(Jcon(i1,i3),j1);

                    Jpair(Jcon(i1,i3),Jcon(j1,j2)) = dJ_dh(Jcon(i1,i3),j2)*dh_dJ(j2,Jcon(j1,j2));
                    Jpair(Jcon(i3,i1),Jcon(j1,j2)) = Jpair(Jcon(i1,i3),Jcon(j1,j2));
                    Jpair(Jcon(i1,i3),Jcon(j2,j1)) = Jpair(Jcon(i1,i3),Jcon(j1,j2));
                    Jpair(Jcon(i3,i1),Jcon(j2,j1)) = Jpair(Jcon(i1,i3),Jcon(j1,j2));

%                     dJ_dJ(i1,i3,j1,j2) = dJ_dh(i1,i3,j2)*dh_dJ(j2,j1,j2);
%                     dJ_dJ(i3,i1,j1,j2) = dJ_dJ(i1,i3,j1,j2);
%                     dJ_dJ(i1,i3,j2,j1) = dJ_dJ(i1,i3,j1,j2);
%                     dJ_dJ(i3,i1,j2,j1) = dJ_dJ(i1,i3,j1,j2);
                    
                end
      
            end
            
        % If j1 has two neighbors:
        else
            
            % Compute magnetizations and correlations:
            m(j1) = (1 - m(j2) - m(j3) + C(j2,j3))/(1 + exp(-h_eff(j1)))...
                + (m(j2) - C(j2,j3))/(1 + exp(-J_eff(j1,j2) - h_eff(j1)))...
                + (m(j3) - C(j2,j3))/(1 + exp(-J_eff(j1,j3) - h_eff(j1)))...
                + C(j2,j3)/(1 + exp(-J_eff(j1,j2) - J_eff(j1,j3) - h_eff(j1)));
            C(j1,j1) = m(j1);
            
            C(j1,j2) = (m(j2) - C(j2,j3))/(1 + exp(-J_eff(j1,j2) - h_eff(j1)))...
                + C(j2,j3)/(1 + exp(-J_eff(j1,j2) - J_eff(j1,j3) - h_eff(j1)));
            C(j2,j1) = C(j1,j2);
            
            C(j1,j3) = (m(j3) - C(j2,j3))/(1 + exp(-J_eff(j1,j3) - h_eff(j1)))...
                + C(j2,j3)/(1 + exp(-J_eff(j1,j2) - J_eff(j1,j3) - h_eff(j1)));
            C(j3,j1) = C(j1,j3);
            
            % Compute derivatives of h(j2), h(j3), and J(j2,j3) with respect to
            % h(j1), J(j1,j2), and J(j1,j3):
            dh_dh(j2,j1) = -1/(1 + exp(-h_eff(j1))) + 1/(1 + exp(-J_eff(j1,j2) - h_eff(j1)));
            dh_dJ(j2,Jcon(j1,j2)) = 1/(1 + exp(-J_eff(j1,j2) - h_eff(j1)));
            dh_dJ(j2,Jcon(j2,j1)) = dh_dJ(j2,Jcon(j1,j2));
            
            dh_dh(j3,j1) = -1/(1 + exp(-h_eff(j1))) + 1/(1 + exp(-J_eff(j1,j3) - h_eff(j1)));
            dh_dJ(j3,Jcon(j1,j3)) = 1/(1 + exp(-J_eff(j1,j3) - h_eff(j1)));
            dh_dJ(j3,Jcon(j3,j1)) = dh_dJ(j3,Jcon(j1,j3));
            
            dJ_dh(Jcon(j2,j3),j1) = 1/(1 + exp(-h_eff(j1))) - 1/(1 + exp(-J_eff(j1,j2) - h_eff(j1)))...
                - 1/(1 + exp(-J_eff(j1,j3) - h_eff(j1))) + 1/(1 + exp(-J_eff(j1,j2) - J_eff(j1,j3) - h_eff(j1)));
            dJ_dh(Jcon(j3,j2),j1) = dJ_dh(Jcon(j2,j3),j1);


            Jpair(Jcon(j2,j3),Jcon(j1,j2)) = -1/(1 + exp(-J_eff(j1,j2) - h_eff(j1)))...
                + 1/(1 + exp(-J_eff(j1,j2) - J_eff(j1,j3) - h_eff(j1)));
            Jpair(Jcon(j2,j3),Jcon(j2,j1)) = Jpair(Jcon(j2,j3),Jcon(j1,j2));
            Jpair(Jcon(j3,j2),Jcon(j1,j2)) = Jpair(Jcon(j2,j3),Jcon(j1,j2));
            Jpair(Jcon(j3,j2),Jcon(j2,j1)) = Jpair(Jcon(j2,j3),Jcon(j1,j2));


%             dJ_dJ(j2,j3,j1,j2) = -1/(1 + exp(-J_eff(j1,j2) - h_eff(j1)))...
%                 + 1/(1 + exp(-J_eff(j1,j2) - J_eff(j1,j3) - h_eff(j1)));
%             dJ_dJ(j2,j3,j2,j1) = dJ_dJ(j2,j3,j1,j2);
%             dJ_dJ(j3,j2,j1,j2) = dJ_dJ(j2,j3,j1,j2);
%             dJ_dJ(j3,j2,j2,j1) = dJ_dJ(j2,j3,j1,j2);


            Jpair(Jcon(j2,j3),Jcon(j1,j3)) = -1/(1 + exp(-J_eff(j1,j3) - h_eff(j1)))...
                + 1/(1 + exp(-J_eff(j1,j2) - J_eff(j1,j3) - h_eff(j1)));
            Jpair(Jcon(j2,j3),Jcon(j3,j1)) = Jpair(Jcon(j2,j3),Jcon(j1,j3));
            Jpair(Jcon(j3,j2),Jcon(j1,j3)) = Jpair(Jcon(j2,j3),Jcon(j1,j3));
            Jpair(Jcon(j3,j2),Jcon(j3,j1)) = Jpair(Jcon(j2,j3),Jcon(j1,j3));

%             dJ_dJ(j2,j3,j1,j3) = -1/(1 + exp(-J_eff(j1,j3) - h_eff(j1)))...
%                 + 1/(1 + exp(-J_eff(j1,j2) - J_eff(j1,j3) - h_eff(j1)));
%             dJ_dJ(j2,j3,j3,j1) = dJ_dJ(j2,j3,j1,j3);
%             dJ_dJ(j3,j2,j1,j3) = dJ_dJ(j2,j3,j1,j3);
%             dJ_dJ(j3,j2,j3,j1) = dJ_dJ(j2,j3,j1,j3);
            
            % Compute derivative of external field of final node with respect
            % to h(j1), J(j1,j2), and J(j1,j3). Dependencies go through h(j2),
            % h(j3), and J(j2,j3):
            dh_dh(i0,j1) = dh_dh(i0,j2)*dh_dh(j2,j1) + dh_dh(i0,j3)*dh_dh(j3,j1)...
                + dh_dJ(i0,Jcon(j2,j3))*dJ_dh(Jcon(j2,j3),j1);
            dh_dJ(i0,Jcon(j1,j2)) = dh_dh(i0,j2)*dh_dJ(j2,Jcon(j1,j2)) + dh_dJ(i0,Jcon(j2,j3))*Jpair(Jcon(j2,j3),Jcon(j1,j2));  %dJ_dJ(j2,j3,j1,j2);
            dh_dJ(i0,Jcon(j2,j1)) = dh_dJ(i0,Jcon(j1,j2));
            dh_dJ(i0,Jcon(j1,j3)) = dh_dh(i0,j3)*dh_dJ(j3,Jcon(j1,j3)) + dh_dJ(i0,Jcon(j2,j3))*Jpair(Jcon(j2,j3),Jcon(j1,j3));  %dJ_dJ(j2,j3,j1,j3);
            dh_dJ(i0,Jcon(j3,j1)) = dh_dJ(i0,Jcon(j1,j3));
            
            % Loop over nodes between final node and j1:
            for t_i = size(D,1):-1:(t_j + 1)
                
                % Node to take derivative of and its decimation neighbors:
                i1 = D(t_i,1);
                i2 = D(t_i,2);
                i3 = D(t_i,3);
                
                % Compute derivatives of h(i1), J(i1,i2), and J(i1,i3) with
                % respect to h(j1), J(j1,j2) and J(j1,j3). All dependencies go
                % through h(j2), h(j3), and J(j2,j3):
                dh_dh(i1,j1) = dh_dh(i1,j2)*dh_dh(j2,j1) + dh_dh(i1,j3)*dh_dh(j3,j1)...
                    + dh_dJ(i1,Jcon(j2,j3))*dJ_dh(Jcon(j2,j3),j1);
                dh_dJ(i1,Jcon(j1,j2)) = dh_dh(i1,j2)*dh_dJ(j2,Jcon(j1,j2)) + dh_dJ(i1,Jcon(j2,j3))*Jpair(Jcon(j2,j3),Jcon(j1,j2)); %dJ_dJ(j2,j3,j1,j2);
                dh_dJ(i1,Jcon(j2,j1)) = dh_dJ(i1,Jcon(j1,j2));
                dh_dJ(i1,Jcon(j1,j3)) = dh_dh(i1,j3)*dh_dJ(j3,Jcon(j1,j3)) + dh_dJ(i1,Jcon(j2,j3))*Jpair(Jcon(j2,j3),Jcon(j1,j3)); %dJ_dJ(j2,j3,j1,j3);
                dh_dJ(i1,Jcon(j3,j1)) = dh_dJ(i1,Jcon(j1,j3));
                
                dJ_dh(Jcon(i1,i2),j1) = dJ_dh(Jcon(i1,i2),j2)*dh_dh(j2,j1) + dJ_dh(Jcon(i1,i2),j3)*dh_dh(j3,j1)...
                    + dJ_dh(Jcon(j2,j3),j1)*Jpair(Jcon(i1,i2),Jcon(j2,j3)); %dJ_dJ(i1,i2,j2,j3)*
                dJ_dh(Jcon(i2,i1),j1) = dJ_dh(Jcon(i1,i2),j1);

                Jpair(Jcon(i1,i2),Jcon(j1,j2)) = dJ_dh(Jcon(i1,i2),j2)*dh_dJ(j2,Jcon(j1,j2)) + Jpair(Jcon(i1,i2),Jcon(j2,j3))*Jpair(Jcon(j2,j3),Jcon(j1,j2));
                Jpair(Jcon(i1,i2),Jcon(j2,j1)) = Jpair(Jcon(i1,i2),Jcon(j1,j2));
                Jpair(Jcon(i2,i1),Jcon(j1,j2)) = Jpair(Jcon(i1,i2),Jcon(j1,j2));
                Jpair(Jcon(i2,i1),Jcon(j2,j1)) = Jpair(Jcon(i1,i2),Jcon(j1,j2));


                Jpair(Jcon(i1,i2),Jcon(j1,j3)) = dJ_dh(Jcon(i1,i2),j3)*dh_dJ(j3,Jcon(j1,j3)) + Jpair(Jcon(i1,i2),Jcon(j2,j3))*Jpair(Jcon(j2,j3),Jcon(j1,j3));
                Jpair(Jcon(i1,i2),Jcon(j3,j1)) = Jpair(Jcon(i1,i2),Jcon(j1,j3));
                Jpair(Jcon(i2,i1),Jcon(j1,j3)) = Jpair(Jcon(i1,i2),Jcon(j1,j3));
                Jpair(Jcon(i2,i1),Jcon(j3,j1)) = Jpair(Jcon(i1,i2),Jcon(j1,j3));


                
                % If i1 has two neighbors:
                if i3 ~= 0
                    
                    dJ_dh(Jcon(i1,i3),j1) = dJ_dh(Jcon(i1,i3),j2)*dh_dh(j2,j1) + dJ_dh(Jcon(i1,i3),j3)*dh_dh(j3,j1)...
                        + dJ_dh(Jcon(j2,j3),j1)*Jpair(Jcon(i1,i3),Jcon(j2,j3)); %dJ_dJ(i1,i3,j2,j3)*
                    dJ_dh(Jcon(i3,i1),j1) = dJ_dh(Jcon(i1,i3),j1);

                    Jpair(Jcon(i1,i3),Jcon(j1,j2)) = dJ_dh(Jcon(i1,i3),j2)*dh_dJ(j2,Jcon(j1,j2)) + Jpair(Jcon(i1,i3),Jcon(j2,j3))*Jpair(Jcon(j2,j3),Jcon(j1,j2));
                    Jpair(Jcon(i1,i3),Jcon(j2,j1)) = Jpair(Jcon(i1,i3),Jcon(j1,j2));
                    Jpair(Jcon(i3,i1),Jcon(j1,j2)) = Jpair(Jcon(i1,i3),Jcon(j1,j2));
                    Jpair(Jcon(i3,i1),Jcon(j2,j1)) = Jpair(Jcon(i1,i3),Jcon(j1,j2));




                    Jpair(Jcon(i1,i3),Jcon(j1,j3)) = dJ_dh(Jcon(i1,i3),j3)*dh_dJ(j3,Jcon(j1,j3)) + Jpair(Jcon(i1,i3),Jcon(j2,j3))*Jpair(Jcon(j2,j3),Jcon(j1,j3));
                    Jpair(Jcon(i1,i3),Jcon(j3,j1)) = Jpair(Jcon(i1,i3),Jcon(j1,j3));
                    Jpair(Jcon(i3,i1),Jcon(j1,j3)) = Jpair(Jcon(i1,i3),Jcon(j1,j3));
                    Jpair(Jcon(i3,i1),Jcon(j3,j1)) = Jpair(Jcon(i1,i3),Jcon(j1,j3));

                
                end
                
            end
            
        end
        
        % Compute susceptibilities:
        
        % Compute susceptibility with final node (derivative of m(i0) with
        % respect to h(j1)). Dependence goes through h(i0):
        dm_dh(i0,j1) = dm_dh(i0,i0)*dh_dh(i0,j1);
        dm_dh(j1,i0) = dm_dh(i0,j1);
        
        % Loop over nodes between final node and j1:
        for t_i = size(D,1):-1:(t_j + 1)
            
            % Node to take derivative of and its decimation neighbors:
            i1 = D(t_i,1);
            i2 = D(t_i,2);
            i3 = D(t_i,3);
                
            % If i1 only has one neighbor:
            if i3 == 0
                
                % Useful quantities:
                l_h = 1/(1 + exp(-h_eff(i1)));
                l_hJ = 1/(1 + exp(-J_eff(i1,i2) - h_eff(i1)));
                dl_h = exp(-h_eff(i1))/(1 + exp(-h_eff(i1)))^2;
                dl_hJ = exp(-J_eff(i1,i2) - h_eff(i1))/(1 + exp(-J_eff(i1,i2) - h_eff(i1)))^2;
                
                % Compute susceptibility (derivative of m(i1) with respect to h(j1)):
                dm_dh(i1,j1) = dl_h*(1 - m(i2))*dh_dh(i1,j1) + dl_hJ*m(i2)*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i2),j1))...
                    + (-l_h + l_hJ)*dm_dh(i2,j1);
                dm_dh(j1,i1) = dm_dh(i1,j1);
                
                % Compute derivative of C(i1,i2) with respect to h(j1):
                dC_dh(Jcon(i1,i2),j1) = dl_hJ*m(i2)*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i2),j1)) + l_hJ*dm_dh(i2,j1);
                dC_dh(Jcon(i2,i1),j1) = dC_dh(Jcon(i1,i2),j1);
                
                Trip(Jcon(i1,i2),j1) = dC_dh(Jcon(i1,i2),j1) - m(i1)*dm_dh(i2,j1) - m(i2)*dm_dh(i1,j1) ...
                    + m(i1)*(dm_dh(i2,j1) + m(i2)*m(j1)') + m(i2)*(dm_dh(i1,j1) + m(i1)*m(j1)')+m(j1)*(dm_dh(i2,i1) + m(i2)*m(i1)') - 2*m(i1)*m(i2)*m(j1);
                Trip(Jcon(i2,i1),j1) = Trip(Jcon(i1,i2),j1);

                if Jcon(i1,j1) ~= 0
                    Trip(Jcon(i1,j1),i2) = Trip(Jcon(i1,i2),j1);
                end
                if Jcon(i2,j1) ~= 0
                    Trip(Jcon(i2,j1),i1) = Trip(Jcon(i1,i2),j1);
                end
           
            % If i1 has two neighbors:
            else
                
                % Useful quantities:
                l_h = 1/(1 + exp(-h_eff(i1)));
                l_hJ2 = 1/(1 + exp(-J_eff(i1,i2) - h_eff(i1)));
                l_hJ3 = 1/(1 + exp(-J_eff(i1,i3) - h_eff(i1)));
                l_hJJ = 1/(1 + exp(-J_eff(i1,i2) - J_eff(i1,i3) - h_eff(i1)));
                dl_h = exp(-h_eff(i1))/(1 + exp(-h_eff(i1)))^2;
                dl_hJ2 = exp(-J_eff(i1,i2) - h_eff(i1))/(1 + exp(-J_eff(i1,i2) - h_eff(i1)))^2;
                dl_hJ3 = exp(-J_eff(i1,i3) - h_eff(i1))/(1 + exp(-J_eff(i1,i3) - h_eff(i1)))^2;
                dl_hJJ = exp(-J_eff(i1,i2) - J_eff(i1,i3) - h_eff(i1))/(1 + exp(-J_eff(i1,i2) - J_eff(i1,i3) - h_eff(i1)))^2;
                
                % Compute susceptibility (derivative of m(i1) with respect to h(j1)):
                dm_dh(i1,j1) = dl_h*(1 - m(i2) - m(i3) + C(i2,i3))*dh_dh(i1,j1)...
                    + dl_hJ2*(m(i2) - C(i2,i3))*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i2),j1))...
                    + dl_hJ3*(m(i3) - C(i2,i3))*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i3),j1))...
                    + dl_hJJ*C(i2,i3)*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i2),j1) + dJ_dh(Jcon(i1,i3),j1))...
                    + (-l_h + l_hJ2)*dm_dh(i2,j1) + (-l_h + l_hJ3)*dm_dh(i3,j1)...
                    + (l_h - l_hJ2 - l_hJ3 + l_hJJ)*dC_dh(Jcon(i2,i3),j1);
                dm_dh(j1,i1) = dm_dh(i1,j1);
                
                % Compute derivative of C(i1,i2) with respect to h(j1):
                dC_dh(Jcon(i1,i2),j1) = dl_hJ2*(m(i2) - C(i2,i3))*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i2),j1))...
                    + dl_hJJ*C(i2,i3)*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i2),j1) + dJ_dh(Jcon(i1,i3),j1))...
                    + l_hJ2*dm_dh(i2,j1) + (-l_hJ2 + l_hJJ)*dC_dh(Jcon(i2,i3),j1);
                dC_dh(Jcon(i2,i1),j1) = dC_dh(Jcon(i1,i2),j1);

                Trip(Jcon(i1,i2),j1) = dC_dh(Jcon(i1,i2),j1) - m(i1)*dm_dh(i2,j1) - m(i2)*dm_dh(i1,j1) ...
                    + m(i1)*(dm_dh(i2,j1) + m(i2)*m(j1)') + m(i2)*(dm_dh(i1,j1) + m(i1)*m(j1)')+m(j1)*(dm_dh(i2,i1) + m(i2)*m(i1)') - 2*m(i1)*m(i2)*m(j1);
                Trip(Jcon(i2,i1),j1) = Trip(Jcon(i1,i2),j1);

                if Jcon(i1,j1) ~= 0
                    Trip(Jcon(i1,j1),i2) = Trip(Jcon(i1,i2),j1);
                end
                if Jcon(i2,j1) ~= 0
                    Trip(Jcon(i2,j1),i1) = Trip(Jcon(i1,i2),j1);
                end
                
                % Compute derivative of C(i1,i3) with respect to h(j1):
                dC_dh(Jcon(i1,i3),j1) = dl_hJ3*(m(i3) - C(i2,i3))*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i3),j1))...
                    + dl_hJJ*C(i2,i3)*(dh_dh(i1,j1) + dJ_dh(Jcon(i1,i2),j1) + dJ_dh(Jcon(i1,i3),j1))...
                    + l_hJ3*dm_dh(i3,j1) + (-l_hJ3 + l_hJJ)*dC_dh(Jcon(i2,i3),j1);
                dC_dh(Jcon(i3,i1),j1) = dC_dh(Jcon(i1,i3),j1);

                Trip(Jcon(i1,i3),j1) = dC_dh(Jcon(i1,i3),j1) - m(i1)*dm_dh(i3,j1) - m(i3)*dm_dh(i1,j1) ...
                    + m(i1)*(dm_dh(i3,j1) + m(i3)*m(j1)') + m(i3)*(dm_dh(i1,j1) + m(i1)*m(j1)')+m(j1)*(dm_dh(i3,i1) + m(i3)*m(i1)') - 2*m(i1)*m(i3)*m(j1);
                Trip(Jcon(i3,i1),j1) = Trip(Jcon(i1,i3),j1);

                if Jcon(i1,j1) ~= 0
                    Trip(Jcon(i1,j1),i3) = Trip(Jcon(i1,i3),j1);
                end
                if Jcon(i3,j1) ~= 0
                    Trip(Jcon(i3,j1),i1) = Trip(Jcon(i1,i3),j1);
                end
                
            end
            
        end
        
    end
    
    % Compute correlations between all nodes:
    C = dm_dh + m*m';
    C(logical(eye(n))) = m;
    
    % Susceptibility:
    X = C - m*m';
end
