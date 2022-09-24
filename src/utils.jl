function σ(z::Float64)
    return 1.0/(1.0 + exp(-z))
end;

function dσ(z::Float64)
    return σ(z) * (1 - σ(z))
end;

function basis_eval!(base_vec::Vector, basis::String, num_basis::Int, tf::Float64, t::Float64)
    # Output: Array [b_j(t)]_{j=0,1,...,num_basis-1}
    
    if basis == "spectral"
        base_vec .= broadcast(k->sin(k*pi*t/tf), 0:num_basis-1)
    elseif basis == "legendre"
        base_vec .= broadcast(k->Pl(2*t/tf - 1, k), 0:num_basis-1)
    else
        return error("No basis found.")
    end
    return nothing;
end

function inner_pulse(basis::String, vv::Vector, num_basis::Int, tf::Float64, t::Float64)
    # Output: Real f(t) = Σ v_j * b_j(t)
    
    base_vec = Array{Float64}(undef, num_basis)
    basis_eval!(base_vec, basis, num_basis, tf, t)
    
    return dot(vv, base_vec);
end;
    
function pulse_constructor(basis::String, vv::Vector, num_basis::Int, tf::Float64, vmin::Float64, vmax::Float64, t::Float64)
    # Output: Real full_pulse(t) = (vmax - vmin) * σ(Σ v_j * b_j(t)) + vmin
    
    inner_val = inner_pulse(basis, vv, num_basis, tf, t);
    return (vmax - vmin) * σ(inner_val) + vmin;
end

function dpdv!(dpdv::Vector, basis::String, vv::Vector, num_basis::Int, tf::Float64, vmin::Float64, vmax::Float64, t::Float64)
    # Output: Array dpdv = ∇_vv full_pulse, evaluated at t
    
    dfdv = Array{Float64}(undef, num_basis)
    basis_eval!(dfdv, basis, num_basis, tf, t)
    inner_val = inner_pulse(basis, vv, num_basis, tf, t)
    dpdv .= (vmax - vmin) * dσ(inner_val) * dfdv

end