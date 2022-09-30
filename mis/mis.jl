using Bloqade
using Distributions
using LegendrePolynomials
using PythonCall
using LinearAlgebra
using Printf

import PyPlot:savefig

include("../src/BloqadeControl.jl")
using .BloqadeControl

NARGS = size(ARGS)[1]
if NARGS != 2
	error("NARGS == 2")
end
n_sites = parse(Int64, ARGS[1])
lr = parse(Float64, ARGS[2])

# Specify parametrization of waveforms
##n_sites = 4
basis = "legendre"
n_basis = 10
use_sigmoid_envolope = true
tf = 1.0
F0 = [(x)->0.,(x)->0.]

# Specify training parameters
params = 2. * rand(Float64, (2, n_basis)) .- 1.
#lr = 5e-2
w_l2 = 1e-3
n_iter = 500
method = "adam"

# Specify parameters of Rydberg atoms
distance = 5.95
atoms = generate_sites(ChainLattice(), n_sites, scale=distance);
local_detuning = [1 for _ in 1:n_sites]
global_detuning = 0.0
n_samples = 20

# Specify initial & target state
state_dict = Dict(
		  "0"=>Array{ComplexF64}([1.0+0.0im,0.0+0.0im]), 
		  "1"=>Array{ComplexF64}([0.0+0.0im,1.0+0.0im])
		  )
zero_state = reduce(kron, [state_dict["0"] for _ in 1:n_sites]; init=1.0)
state1 = reduce(kron, [state_dict[string(ind % 2)] for ind in 1:n_sites]; init=1.0)
state2 = reduce(kron, [state_dict[string((ind + 1) % 2)] for ind in 1:n_sites]; init=1.0)
target_state = (state1 + state2) ./sqrt(2)
#target_state = Array{ComplexF64}([1, 0, 0, 1]./sqrt(2))

init_states = [zero_state]
observables = []

J = UniformScaling(1.)
target_state = ArrayReg(target_state; nlevel=2)
tar_op = density_matrix(target_state).state
M = GeneralMatrixBlock(J - tar_op; nlevel=2);
append!(observables, [M])

# Setup optimizer
generator = WaveformGenerator(n_sites, basis, n_basis, use_sigmoid_envolope, F0, tf)
model = BloqadeModel(atoms, local_detuning, global_detuning, n_samples, init_states, observables);
optim = Optimizer(params, lr, w_l2, n_iter, method)

# Plot waveforms
update_waveform!(model, generator, optim.params)
Bloqade.plot(generator.Ω1)
savefig("./wave0.png")

update_waveform!(model, generator, optim.params)
@printf "Current loss function value is %.3f.\n" compute_loss(model, generator, optim)
losses = train_loss!(model, generator, optim)
npzwrite("GHZ_N"*string(n_sites)*"lr"*string(lr)*".npz", losses)
Bloqade.plot(generator.Ω1)
savefig("./wave1.png")
