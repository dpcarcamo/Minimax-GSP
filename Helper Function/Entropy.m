function [H] = Entropy(mean)
    H = -mean.*log(mean)-(1-mean).*log(1-mean);
    H(mean==0) = 0;
    H(mean==1) = 0;
end
