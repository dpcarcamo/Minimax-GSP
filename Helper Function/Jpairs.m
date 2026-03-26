function [Js, connum, pairs] =  Jpairs(J, D)

    
    Jcon = J ~= 0;
    Jcon = triu(Jcon);

    pairs = zeros(size(Jcon,1),2);


    Js = zeros(size(Jcon)); % I have no idea why i cant update Jcon
    count = 0;
    for iter = 1:size(D,1)
        
        i = D(iter,1);
        j = D(iter,2);
        k = D(iter,3);

        count = count + 1;
        Js(i, j) = count;
        pairs(count,1) = i;
        pairs(count,2) = j;
        if k ~=0 
            count = count + 1;
            Js(i, k) = count; 
            pairs(count, 1) = i;
            pairs(count, 2) = k;
        end

    end

    Js = Js + Js.';
    connum = count;

end
