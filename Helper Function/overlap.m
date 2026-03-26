function [overlapfrac,overlapfracresp,overlapfracspont, fracs] = overlap(respJ,spontJ)
    %Function for calculating the overlap between two networks
    

    neruon_num_spont = size(spontJ,1);
    trials = (2*neruon_num_spont-3);
    overlapfrac = zeros(1,trials);
    overlapfracresp = zeros(1,trials);
    overlapfracspont = zeros(1,trials);

    respJ = abs(respJ);
    spontJ = abs(spontJ);
    
    sortResp = sort(respJ(triu(respJ)~=0));
    sortSpont = sort(spontJ(triu(spontJ)~=0));
    
    sortResp = [0,sortResp.'];
    sortSpont = [0,sortSpont.'];
    
    for count = 1:trials
        count
        brespJ = respJ ~= 0;
        bspontJ = spontJ ~=0;

        brespJthres = respJ > sortResp(count);
        bspontJthres = spontJ > sortSpont(count);
        
        samecons = brespJthres + bspontJthres > 1;
        samecons = samecons(1:neruon_num_spont,1:neruon_num_spont);
        overlapfrac(count) = full(sum(sum(samecons)))/(2*(2*neruon_num_spont-3-count+1));
        samecons = brespJthres + bspontJ > 1;
        samecons = samecons(1:neruon_num_spont,1:neruon_num_spont);
        overlapfracresp(count) = full(sum(sum(samecons)))/(2*(2*neruon_num_spont-3-count+1));
        samecons = brespJ + bspontJthres > 1;
        samecons = samecons(1:neruon_num_spont,1:neruon_num_spont);
        overlapfracspont(count) = full(sum(sum(samecons)))/(2*(2*neruon_num_spont-3-count+1));
    end
    
    fracs = (2*neruon_num_spont-3-(1:trials)+1)/((2*neruon_num_spont-3));
end