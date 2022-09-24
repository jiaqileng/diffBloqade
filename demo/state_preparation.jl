using Bloqade
using Distributions
using LegendrePolynomials
using PythonCall
using LinearAlgebra

include("BloqadeControl.jl")
using .BloqadeControl


# Build Bloqade Model
n_sites = 2
distance = 5.95
local_detuning = [-1, 1]

n_basis = 10
basis = "legendre"
tf = 1.0
n_samples = 20

#target_state = arrayreg(bit"11")
#target_state = uniform_state(n_sites)
bell_state = Array{ComplexF64}([1, 0, 0, 1]./sqrt(2))
target_state = ArrayReg(bell_state; nlevel=2)
J = UniformScaling(1.)
tar_op = density_matrix(target_state).state
M = J - tar_op;
M = GeneralMatrixBlock(M; nlevel=2);

model = BloqadeModel(n_sites, distance, local_detuning, n_basis, basis, tf, M, n_samples);

# Build optimzier
params = rand(Float64, (2, model.n_basis))
lr = 5e-2
n_iter = 500
method = "adam"

optim = Optimizer(params, lr, n_iter, method)
train_loss!(model, optim)
