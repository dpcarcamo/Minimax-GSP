function [Energy, Specific_heat, D]= Thermodynamics(J,h, T)
    
    Jeff = J;
    heff = h;
    f = 0;
    df = 0;
    dh = zeros(size(h));
    dJ = zeros(size(J));
    ddf = 0;
    ddh = zeros(size(h));
    ddJ = zeros(size(J));

    [G, D] = decimate_GSP(J);


    for step = 1:size(D,1)

        i = D(step, 1);
        j = D(step, 2);
        k = D(step, 3);
        
        heff(j) = heff(j) - T*log(exp(heff(i)/T)+1) + T*log(exp((Jeff(j, i)+ heff(i))/T)+1);

        df = df + log(exp(heff(i)/T)+1) + 1/(exp(-heff(i)/T)+1)*(dh(i) - heff(i)/T);

        dh(j) = dh(j) + log((exp((heff(i)+Jeff(i,j))/T)+1)/(exp(heff(i)/T)+1)) - 1/(exp(-heff(i)/T)+1)*(dh(i) - heff(i)/T) + 1/(exp(-(heff(i)+Jeff(i,j))/T)+1)*(dh(i) + dJ(i,j) - (heff(i)+Jeff(i,j))/T);
        
        ddf = ddf + 1/(exp(-heff(i)/T)+1)*ddh(i) + exp(-heff(i)/T)/(T*(exp(-heff(i)/T)+1)^2)*((dh(i) - heff(i)/T)^2);
        
        ddh(j) = ddh(j) + 1/(exp(-(heff(i)+Jeff(i,j))/T)+1)*(ddh(i)+ddJ(i,j)) - 1/(exp(-heff(i)/T)+1)*ddh(i) - exp(-heff(i)/T)/((exp(-heff(i)/T)+1)^2)*((dh(i) - heff(i)/T)^2)/T + exp(-(heff(i)+Jeff(i,j))/T)/((exp(-(heff(i)+Jeff(i,j))/T)+1)^2)*((dh(i) + dJ(i,j) - (heff(i)+Jeff(i,j))/T)^2)/T;

        if k ~= 0
            
            heff(k) = heff(k) - T*log(exp(heff(i)/T)+1) + T*log(exp((Jeff(i, k)+ heff(i))/T)+1);
            
            Jeff(j,k) = Jeff(j,k) + T*log(exp(heff(i)/T)+1) - T*log(exp((Jeff(i,j)+heff(i))/T)+1) - T*log(exp((Jeff(i,k)+heff(i))/T)+1) + T*log(exp((Jeff(i,j)+ Jeff(i,k)+heff(i))/T)+1);
            
            Jeff(k, j) = Jeff(j, k);
            
            dh(k) = dh(k) + log((exp((heff(i)+Jeff(i,k))/T)+1)/(exp(heff(i)/T)+1)) - 1/(exp(-heff(i)/T)+1)*(dh(i) - heff(i)/T) + 1/(exp(-(heff(i)+Jeff(i,k))/T)+1)*(dh(i) + dJ(i,k) - (heff(i)+Jeff(i,k))/T);
            
            dJ(j,k) = dJ(j,k) + log(((exp((heff(i)+Jeff(i,k)+Jeff(i,j))/T)+1)*(exp((heff(i))/T)+1))/((exp((heff(i)+Jeff(i,k))/T)+1)*(exp((heff(i)+Jeff(i,j))/T)+1))) + 1/(exp(-heff(i)/T)+1)*(dh(i) - heff(i)/T) - 1/(exp(-(heff(i)+Jeff(i,j))/T)+1)*(dh(i) + dJ(i,j) - (heff(i)+Jeff(i,j))/T) - 1/(exp(-(heff(i)+Jeff(i,k))/T)+1)*(dh(i) + dJ(i,k) - (heff(i)+Jeff(i,k))/T) + 1/(exp(-(heff(i)+Jeff(i,j)+Jeff(i,k))/T)+1)*(dh(i) + dJ(i,j) + dJ(i,k)  - (heff(i)+Jeff(i,j)+Jeff(i,k))/T);
            
            dJ(k,j) = dJ(j,k);

            ddh(k) = ddh(k) + 1/(exp(-(heff(i)+Jeff(i,k))/T)+1)*(ddh(i)+ddJ(i,k)) - 1/(exp(-heff(i)/T)+1)*ddh(i) - exp(-heff(i)/T)/((exp(-heff(i)/T)+1)^2)*((dh(i) - heff(i)/T)^2)/T + exp(-(heff(i)+Jeff(i,k))/T)/((exp(-(heff(i)+Jeff(i,k))/T)+1)^2)*((dh(i) + dJ(i,k) - (heff(i)+Jeff(i,k))/T)^2)/T;

            ddJ(j,k) = ddJ(j,k) + 1/(exp(-heff(i)/T)+1)*ddh(i) + exp(-heff(i)/T)/((exp(-heff(i)/T)+1)^2)*((dh(i) - heff(i)/T)^2)/T - 1/(exp(-(heff(i)+Jeff(i,j))/T)+1)*(ddh(i)+ddJ(i,j)) - exp(-(heff(i)+Jeff(i,j))/T)/((exp(-(heff(i)+Jeff(i,j))/T)+1)^2)*((dh(i) + dJ(i,j) - (heff(i)+Jeff(i,j))/T)^2)/T     - 1/(exp(-(heff(i)+Jeff(i,k))/T)+1)*(ddh(i)+ddJ(i,k)) - exp(-(heff(i)+Jeff(i,k))/T)/((exp(-(heff(i)+Jeff(i,k))/T)+1)^2)*((dh(i) + dJ(i,k) - (heff(i)+Jeff(i,k))/T)^2)/T   + 1/(exp(-(heff(i)+Jeff(i,j)+Jeff(i,k))/T)+1)*(ddh(i)+ddJ(i,j)+ddJ(i,k)) + exp(-(heff(i)+Jeff(i,j)+Jeff(i,k))/T)/((exp(-(heff(i)+Jeff(i,j)+Jeff(i,k))/T)+1)^2)*((dh(i) + dJ(i,j) + dJ(i,k) - (heff(i)+Jeff(i,j)+Jeff(i,k))/T)^2)/T;

            ddJ(k,j) = ddJ(j,k);

        end
        

    end

    
    df = df + log(exp(heff(j)/T)+1) + 1/(exp(-heff(j)/T)+1)*(dh(j) - heff(j)/T);
    ddf = ddf + 1/(exp(-heff(j)/T)+1)*ddh(j) + exp(-heff(j)/T)/(T*(exp(-heff(j)/T)+1)^2)*((dh(j) - heff(j)/T)^2);

    dim = size(D, 1);

    for i = 1:length(h)
        f = f + T*log(exp(heff(i)/T)+1);
    end



    Energy = - f + T*df;

    Specific_heat = T * ddf;


end
