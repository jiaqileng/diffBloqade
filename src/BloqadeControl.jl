module BloqadeControl

using Bloqade
using Distributions
using LegendrePolynomials
#using PythonCall
using LinearAlgebra
using Printf

include("utils.jl")

export BloqadeModel, WaveformGenerator, Optimizer, update_waveform!, compute_loss, backward!, train_loss!, grad_fdm



mutable struct BloqadeModel
    atoms::AtomList{1, Float64}
    local_detuning::Vector
    global_detuning::Float64
    n_samples::Int
    init_states::Vector{Vector{ComplexF64}}
    #observables::Vector{GeneralMatrixBlock{2, ComplexF64, Matrix{ComplexF64}}}
    observables::Vector{Any}

    C6::Float64
    rabi_min::Float64
    rabi_max::Float64
    delta_min::Float64
    delta_max::Float64
    
    function BloqadeModel(atoms::AtomList{1, Float64}, local_detuning::Vector, global_detuning::Float64, n_samples::Int, init_states::Vector{Vector{ComplexF64}}, observables::Vector{Any})
        if length(atoms) != length(local_detuning)
            return error("Atoms size does not match local detuning.")
        end

        C6 = 862690*2*pi
        rabi_min = 0.0
        rabi_max = 8*pi
        delta_min = 0.0
        delta_max = 40*pi
        
        new(atoms, local_detuning, global_detuning, n_samples, init_states, observables,
            C6, rabi_min, rabi_max, delta_min, delta_max)
    end
end

mutable struct WaveformGenerator
    n_sites::Int
    basis::String
    n_basis::Int
    use_sigmoid_envolope::Bool
    f0::Vector{Function}
    tf::Float64

    Ω1::Waveform
    Δ1::Vector{Waveform}

    function WaveformGenerator(n_sites::Int, basis::String, n_basis::Int, use_sigmoid_envolope::Bool, f0::Vector{Function}, tf::Float64)
        Ω1 = Waveform(t->0, duration=tf);
        Δ1 = map(1:n_sites) do idx
            Waveform(t->0, duration=tf);
        end

        new(n_sites, basis, n_basis, use_sigmoid_envolope, f0, tf, Ω1, Δ1)
    end
end

mutable struct Optimizer
    params::Matrix
    lr::Float64
    w_l2::Float64
    n_epochs::Int
    method::String

    g::Matrix
    m::Matrix
    v::Matrix
    
    function Optimizer(params::Matrix, lr::Float64, w_l2::Float64, n_epochs::Int, method::String)
        g = zeros(Float64, size(params))
        m = zeros(Float64, size(params))
        v = zeros(Float64, size(params))
        new(params, lr, w_l2, n_epochs, method, g, m, v)
    end
end



function basis_eval!(base_vec::Vector, generator::WaveformGenerator, t::Float64)
    # Return: Array [b_j(t)]_{j=0,1,...,num_basis-1}
    
    if generator.basis == "spectral"
        base_vec .= broadcast(k->sine(t/generator.tf, k), 0:generator.n_basis-1)

    elseif generator.basis == "legendre"
        base_vec .= broadcast(k->Pl(2*t/generator.tf - 1, k), 0:generator.n_basis-1)

    elseif generator.basis == "bspline"
        base_vec .= broadcast(k->bspline(t/generator.tf, generator.n_basis, k), 0:generator.n_basis-1)

    else
        return error("No basis found.")
    end
    return nothing;
end

function raw_pulse(generator::WaveformGenerator, tag::Char, vv::Vector, t::Float64)
    # Args:
    #   f0: initial guess pulse, to be fine tuned; specified as a function of t for 0 <= t <= tf
    # Return: Real f(t) = Σ v_j * b_j(t)
    
    base_vec = Array{Float64}(undef, generator.n_basis)
    basis_eval!(base_vec, generator, t)

    if tag == 'x'
        f0 = generator.f0[1]
    elseif tag == 'z'
        f0 = generator.f0[2]
    end

    return f0(t) + dot(vv, base_vec);
end;
    
function pulse_constructor(generator::WaveformGenerator, tag::Char, vv::Vector, vmin::Float64, vmax::Float64, t::Float64)
    # Return: Real full_pulse(t) = (vmax - vmin) * σ(Σ v_j * b_j(t)) + vmin
    
    raw_pulse_val = raw_pulse(generator, tag, vv, t);
    if generator.use_sigmoid_envolope == true
        return (vmax - vmin) * σ(raw_pulse_val) + vmin;
    else
        return raw_pulse_val;
    end
end

function dpdv!(dpdv::Vector, generator::WaveformGenerator, tag::Char, vv::Vector, vmin::Float64, vmax::Float64, t::Float64)
    # Output: Array dpdv = ∇_vv full_pulse, evaluated at t
    
    dfdv = Array{Float64}(undef, generator.n_basis)
    basis_eval!(dfdv, generator, t)
    raw_val = raw_pulse(generator, tag, vv, t)

    if generator.use_sigmoid_envolope == true
        dpdv .= (vmax - vmin) * dσ(raw_val) * dfdv
    else
        dpdv .= dfdv
    end

    return nothing;
end

function update_waveform!(model::BloqadeModel, generator::WaveformGenerator, params::Matrix)
    generator.Ω1 = Waveform(t->pulse_constructor(generator, 'x', params[1,:], model.rabi_min, model.rabi_max, t), duration=generator.tf);
    generator.Δ1 .= map(1:generator.n_sites) do idx
            Waveform(t->model.local_detuning[idx]*pulse_constructor(generator, 'z', params[2,:], model.delta_min, model.delta_max, t)+model.global_detuning, duration=generator.tf);
    end

    return nothing;
end

function eval_ps_onsite(model::BloqadeModel, generator::WaveformGenerator, tag::Char, site_id::Int, reg_τ::AbstractRegister, M::GeneralMatrixBlock, τ::Float64, sign::Char, params::Matrix)
    if sign == '+'
        s = 1.0
    elseif sign == '-'
        s = -1.0
    end
    
    hamiltonian = rydberg_h(model.atoms; Ω=generator.Ω1, Δ=generator.Δ1, ϕ=0)
    reg = copy(reg_τ) # pass in the quantum state at time τ

    # Apply Pauli rotations
    if tag == 'x'
        # Apply X rotations on the site
        apply!(reg, put(site_id=>Rx(2*(1 + s*3/4)*pi)))
    elseif tag == 'z'
        # Apply Z rotations at a site
        apply!(reg, put(site_id=>Rz(2*(1 + s*3/4)*pi)))
    else 
        return error("Tag must be x or z.")
    end
    
    # Phase 2 emulation
    prob = SchrodingerProblem(reg, (τ, generator.tf), hamiltonian)
    integrator = init(prob, Vern8());
    emulate!(prob)
    
    ps = expect(M, reg)
    
    return real(ps);
end

function eval_ps!(ps::Vector, model::BloqadeModel, generator::WaveformGenerator, tag::Char, reg_τ::AbstractRegister, M::GeneralMatrixBlock, τ::Float64, sign::Char, params::Matrix)
    n_sites = length(model.atoms)
    ps .= broadcast(site_id->eval_ps_onsite(model, generator, tag, site_id, reg_τ, M, τ, sign, params), 1:n_sites)

    return nothing;
end

function compute_loss(model::BloqadeModel, generator::WaveformGenerator, optimizer::Optimizer)
    a, b = size(optimizer.params)
    if a != 2 || b != generator.n_basis
        return error("Parameter size does not match.")
    end
    
    # Hamiltonian creation
    hamiltonian = rydberg_h(model.atoms; Ω = generator.Ω1, Δ=generator.Δ1, ϕ=0)
    
    num_states = length(model.init_states)
    loss = 0

    for j in 1:num_states
        #reg = zero_state(length(model.atoms)) # initialization to all-zero states
        ψ0 = copy(model.init_states[j])
        reg = ArrayReg(ψ0; nlevel=2)
        prob = SchrodingerProblem(reg, (0, generator.tf), hamiltonian)
        integrator = init(prob, Vern8())
        emulate!(prob)
        loss += real(expect(model.observables[j], reg))
    end

    # Add l2 regularization
    loss_l2 = optimizer.w_l2 * 0.5 * norm(optimizer.params)^2 / (a*b)

    return loss/num_states + loss_l2;
end

function backward!(model::BloqadeModel, generator::WaveformGenerator, optimizer::Optimizer, ψ0::Vector, M::GeneralMatrixBlock)
    a, b = size(optimizer.g)
    if a != 2 || b != generator.n_basis
        return error("Gradient size does not match.")
    end

    # Erase existing grad info
    optimizer.g .= zeros(Float64, size(optimizer.params))
    
    n_sites = length(model.atoms)
    T = generator.tf
    hamiltonian = rydberg_h(model.atoms; Ω=generator.Ω1, Δ=generator.Δ1, ϕ=0)
    ∇Ω_τ = Array{Float64}(undef, generator.n_basis)
    ∇Δ_τ = Array{Float64}(undef, generator.n_basis)
    p_m = Array{Float64}(undef, n_sites)
    p_p = Array{Float64}(undef, n_sites)

    # Monte Carlo sampling for gradient
    counter = 1
    while counter <= model.n_samples
        #τ = Uniform(0, T)
        τ = T * rand()
        dpdv!(∇Ω_τ, generator, 'x', optimizer.params[1,:], model.rabi_min, model.rabi_max, τ)
        dpdv!(∇Δ_τ, generator, 'z', optimizer.params[2,:], model.delta_min, model.delta_max, τ)

        # Phase 1 emulation
        #reg0 = zero_state(n_sites) # initialization to all-zero states
        state0 = copy(ψ0)
        reg0 = ArrayReg(state0; nlevel=2)
        prob = SchrodingerProblem(reg0, (0, τ), hamiltonian)
        integrator = init(prob, Vern8());
        emulate!(prob)

        # Update grad for x
        eval_ps!(p_m, model, generator, 'x', reg0, M, τ, '-', optimizer.params)
        eval_ps!(p_p, model, generator, 'x', reg0, M, τ, '+', optimizer.params)
        optimizer.g[1,:] .+= ∇Ω_τ * dot(ones(n_sites), p_m - p_p)

        # Update grad for z
        eval_ps!(p_m, model, generator, 'z', reg0, M, τ, '-', optimizer.params)
        eval_ps!(p_p, model, generator, 'z', reg0, M, τ, '+', optimizer.params)
        optimizer.g[2,:] .+= ∇Δ_τ * dot(model.local_detuning, p_m - p_p)
        
        counter += 1
    end

    optimizer.g .= 0.5 * T * optimizer.g ./ model.n_samples

    # Add gradient from l2 regularization
    optimizer.g .+= optimizer.w_l2 * optimizer.params ./ (a*b)

    return nothing;
end

function step!(optimizer::Optimizer, epoch::Int)
    if optimizer.method == "sgd"
        optimizer.params .-= optimizer.lr * optimizer.g
    elseif optimizer.method == "adam"
        # Adam algorithm -- ArXiv:1412.6980
        β1 = 0.9
        β2 = 0.99
        ϵ = 1e-8
        optimizer.m .= β1 * optimizer.m + (1 - β1) * optimizer.g
        optimizer.v .= β2 * optimizer.v + (1 - β2) * optimizer.g.^2
        m_ = optimizer.m / (1 - β1^epoch)
        v_ = optimizer.v / (1 - β2^epoch)
        optimizer.params .-= optimizer.lr * (m_ ./ sqrt.(v_ .+ ϵ))
    end
    
    return nothing;
end

function train_loss!(model::BloqadeModel, generator::WaveformGenerator, optimizer::Optimizer)
    a, b = size(optimizer.params)
    if a != 2 || b != generator.n_basis
        return error("Parameter size does not match.")
    end
    
    num_states = length(model.init_states)
    batch_losses = Array{Float64}(undef, num_states)

    losses = Array{Float64}([epoch for epoch in 1:optimizer.n_epochs if epoch % 10 == 0])
    count = 1
    for epoch in 1:optimizer.n_epochs
        for state_id in 1:num_states
            ψ0 = copy(model.init_states[state_id])
            M = copy(model.observables[state_id])
            update_waveform!(model, generator, optimizer.params)
            backward!(model, generator, optimizer, ψ0, M)
            step!(optimizer, epoch)
            batch_losses[state_id] = compute_loss(model, generator, optimizer)
        end
        
        if mod(epoch, 10) == 0
            loss_val = compute_loss(model, generator, optimizer)
	    losses[count] = loss_val
	    count = count + 1
            batch_loss = mean(batch_losses)
            #loss_l2 = optimizer.w_l2 * 0.5 * norm(optimizer.params)^2 / (a*b)
            #print("epoch: ", epoch, ", loss = ", loss_val, ", loss_l2 = ", loss_l2, ".\n");
            @printf "epoch: %i, loss = %.4f, batch_loss = %.4f\n" epoch loss_val batch_loss
            #@printf "epoch: %i, loss = %.4f, batch_loss = %f\n" epoch loss_val batch_losses
        end
    end
    return losses
end

function grad_fdm(model::BloqadeModel, generator::WaveformGenerator, optim::Optimizer, dx::Float64)
    grad_fdm = zeros((2, generator.n_basis))
    params0 = copy(optim.params)

    for j in 1:2
        for k in 1:generator.n_basis
            params_p = copy(params0)
            params_p[j,k] += dx
            optim.params = params_p
            update_waveform!(model, generator, optim.params)
            f_p = compute_loss(model, generator, optim)

            params_m = copy(params0)
            params_m[j,k] -= dx
            optim.params = params_m
            update_waveform!(model, generator, optim.params)
            f_m = compute_loss(model, generator, optim)

            grad_fdm[j,k] += (f_p - f_m)/(2*dx)
        end
    end
    
    return grad_fdm = grad_fdm;
end


end
