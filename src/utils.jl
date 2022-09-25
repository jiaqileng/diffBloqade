function σ(z::Float64)
    return 1.0/(1.0 + exp(-z))
end;

function dσ(z::Float64)
    return σ(z) * (1 - σ(z))
end;

function sine(z::Float64, k::Int)
    # Args: 
    #   z: variable, 0 <= z <= 1
    #   k: index of basis, 0 <= k <= N-1
    # Return: a sine wave basis supported on [0,1].

    return sin(k*pi*z)
end;

function bspline(z::Float64, N::Int, k::Int)
    # Args: 
    #   z: variable, 0 <= z <= 1
    #   N: number of basis
    #   k: index of basis, 0 <= k <= N-1
    # Return: a quadratic B-Spline basis supported on [(k-1)/(N+2), (k+2)/(N+2)].

    l = k/(N+2)
    r = (k+3)/(N+2)
    norm_factor = 4/(l+r)^2

    if z < l || z > r
        return 0;
    else
        return norm_factor * (z-l)*(z-r);
    end
end;
