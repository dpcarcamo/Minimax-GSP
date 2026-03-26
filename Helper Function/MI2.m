function deltaS = MI2(mean,corr)
    
    a = -mean.*log(mean) - (1-mean).*log(1-mean) - (mean.*log(mean)).' - ((1-mean).*log(1-mean)).';
    b = corr.*log(corr);
    c = (mean - corr).*log(mean - corr) + ((mean - corr).*log(mean - corr)).';
    d = (1 - mean - mean.' +corr).*log((1 - mean - mean.' +corr));
    deltaS = real(a + b + c + d);

    for i = 1:length(mean)
        deltaS(i,i) = 0; %-mean(i)*log(mean(i)) - (1-mean(i))*log(1-mean(i));
    end

    deltaS(isnan(deltaS)) = 0;
end
