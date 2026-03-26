function Emean = energyIsing(spin, J, h)
    %ENERGYISING Mean energy per spin.
    %   Emean = ENERGYISING(spin, J) returns the mean energy per spin of the
    %   configuration |spin|. |spin| is a matrix of +/- 1's. |J| is a scalar.

    Emean = -(1./2).*(spin.' * J * spin)  - h.'*spin;
end