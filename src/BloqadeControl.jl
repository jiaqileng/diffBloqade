module BloqadeControl

using Bloqade
using Distributions
using LegendrePolynomials
#using PythonCall
using LinearAlgebra

include("utils.jl")

export BloqadeModel, Optimizer, compute_loss, backward!, train_loss!

mutable struct BloqadeModel
    n_sites::Int
    distance::Float64
    local_detuning::Vector
    n_basis::Int
    basis::String
    tf::Float64
    M::GeneralMatrixBlock
    n_samples::Int
    
    C6::Float64
    rabi_min::Float64
    rabi_max::Float64
    delta_min::Float64
    delta_max::Float64
    
    atoms::AtomList{1, Float64}
    
    function BloqadeModel(n_sites, distance, local_detuning, n_basis, basis, tf, M, n_samples)
        C6 = 862690*2*pi
        rabi_min = 0.0
        rabi_max = 25.0
        delta_min = -125.0
        delta_max = 125.0
        atoms = generate_sites(ChainLattice(), n_sites, scale=distance);
        
        new(n_sites, distance, local_detuning, n_basis, basis, tf, M, n_samples, 
            C6, rabi_min, rabi_max, delta_min, delta_max, atoms)
    end
end


mutable struct Optimizer
    params::Matrix
    lr::Float64
    n_iter::Int
    method::String

    g::Matrix
    m::Matrix
    v::Matrix
    
    function Optimizer(params, lr, n_iter, method)
        g = zeros(Float64, size(params))
        m = zeros(Float64, size(params))
        v = zeros(Float64, size(params))
        new(params, lr, n_iter, method, g, m, v)
    end
end


function waveform_synthesizer(model::BloqadeModel, params::Matrix)
    Ω1 = Waveform(t->pulse_constructor(model.basis, params[1,:], model.n_basis, model.tf, model.rabi_min, model.rabi_max, t), duration=model.tf);
    Δ1 = map(1:model.n_sites) do idx
            Waveform(t->model.local_detuning[idx]*pulse_constructor(model.basis, params[2,:], model.n_basis, model.tf, model.delta_min, model.delta_max, t), duration=model.tf);
    end
    
    return Ω1, Δ1;
end

function eval_ps_x(model::BloqadeModel, site_id::Int, τ::Float64, sign::Char, params::Matrix)
    if sign == '+'
        s = 1.0
    elseif sign == '-'
        s = -1.0
    end
    
    Ω1, Δ1 = waveform_synthesizer(model, params)
    hamiltonian = rydberg_h(model.atoms; Ω=Ω1, Δ=Δ1, ϕ=0)
    reg = zero_state(model.n_sites) # initialization to all-zero states
    
    # Phase 1 emulation
    prob = SchrodingerProblem(reg, (0, τ), hamiltonian)
    integrator = init(prob, Vern8());
    emulate!(prob)
    
    # Apply X rotations at a site
    apply!(reg, put(site_id=>Rx(2*(1 + s*3/4)*pi)))
    
    # Phase 2 emulation
    prob = SchrodingerProblem(reg, (τ, model.tf), hamiltonian)
    integrator = init(prob, Vern8());
    emulate!(prob)
    
    ps = expect(model.M, reg)
    
    return real(ps);
end

function eval_ps_z(model::BloqadeModel, site_id::Int, τ::Float64, sign::Char, params::Matrix)
    if sign == '+'
        s = 1.0
    elseif sign == '-'
        s = -1.0
    end
    
    Ω1, Δ1 = waveform_synthesizer(model, params)
    hamiltonian = rydberg_h(model.atoms; Ω=Ω1, Δ=Δ1, ϕ=0)
    reg = zero_state(model.n_sites) # initialization to all-zero states
    
    # Phase 1 emulation
    prob = SchrodingerProblem(reg, (0, τ), hamiltonian)
    integrator = init(prob, Vern8());
    emulate!(prob)
    
    # Apply Z rotations at a site
    apply!(reg, put(site_id=>Rz(2*(1 + s*3/4)*pi)))
    
    # Phase 2 emulation
    prob = SchrodingerProblem(reg, (τ, model.tf), hamiltonian)
    integrator = init(prob, Vern8());
    emulate!(prob)
    
    ps = expect(model.M, reg)
    
    return real(ps);
end

function compute_loss(model::BloqadeModel, optimizer::Optimizer)
    a, b = size(optimizer.params)
    if a != 2 || b != model.n_basis
        return error("Parameter size does not match.")
    end
    
    # Waveform creation
    Ω1, Δ1 = waveform_synthesizer(model, optimizer.params)
    hamiltonian = rydberg_h(model.atoms; Ω = Ω1, Δ=Δ1, ϕ=0)
    
    # Simulator setup
    reg = zero_state(model.n_sites) # initialization to all-zero states
    prob = SchrodingerProblem(reg, (0, model.tf), hamiltonian)
    integrator = init(prob, Vern8())
    emulate!(prob)

    return real(expect(model.M, reg));
end

function backward!(model::BloqadeModel, optimizer::Optimizer)
    # Erase existing grad info
    optimizer.g .= zeros(Float64, size(optimizer.params))
    
    # Generate random seeds for τ
    #Tau = model.tf * rand(Float64, model.n_samples)
    
    # Monte Carlo sampling for Ω
    for site_id in 1:model.n_sites
        g = zeros(model.n_basis)
        ∇Ω_τ = Array{Float64}(undef, model.n_basis)
        for j in 1:model.n_samples
            τ = model.tf * rand()
            dpdv!(∇Ω_τ, model.basis, optimizer.params[1,:], model.n_basis, model.tf, model.rabi_min, model.rabi_max, τ)
            p_m = eval_ps_x(model, site_id, τ, '-', optimizer.params)
            p_p = eval_ps_x(model, site_id, τ, '+', optimizer.params)
            g .+= ∇Ω_τ * (p_m - p_p)
        end
        optimizer.g[1,:] .+= 0.5 * g / model.n_samples
    end
    
    # Monte Carlo sampling for Δ
    for site_id in 1:model.n_sites
        g = zeros(model.n_basis)
        ∇Δ_τ = Array{Float64}(undef, model.n_basis)
        for j in 1:model.n_samples
            #τ = Tau[j]
            τ = model.tf * rand()
            basis_eval!(∇Δ_τ, model.basis, model.n_basis, model.tf, τ)
            dpdv!(∇Δ_τ, model.basis, optimizer.params[2,:], model.n_basis, model.tf, model.delta_min, model.delta_max, τ)
            p_m = eval_ps_z(model, site_id, τ, '-', optimizer.params)
            p_p = eval_ps_z(model, site_id, τ, '+', optimizer.params)
            g .+=  ∇Δ_τ * (p_m - p_p)
        end
        optimizer.g[2,:] .+= 0.5 * model.local_detuning[site_id] * g / model.n_samples
    end
    
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

function train_loss!(model::BloqadeModel, optimizer::Optimizer)
    a, b = size(optimizer.params)
    if a != 2 || b != model.n_basis
        return error("Parameter size does not match.")
    end
    
    for epoch in 1:optimizer.n_iter
        backward!(model, optimizer)
        step!(optimizer, epoch)
        
        if mod(epoch, 10) == 0
            loss_val = compute_loss(model, optimizer)
            print("epoch: ", epoch, ", loss = ", loss_val, ".\n");
        end
    end
    
end


end