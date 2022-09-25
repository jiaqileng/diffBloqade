# Differentiable Analog Pulse Learning for Rydberg Atoms

We introduce **BloqadeControl**, a Julia module developed for differentiable analog quantum computing on neutral-atom quantum computers. We use **Bloqade** from QuEraComputing to simulate Rydberg atoms. 

## Setup
The module is developed with Julia 1.7.3. It is also required to pre-install the following packages:
```bash
using Bloqade
using Distributions
using LegendrePolynomials
using PythonCall
using LinearAlgebra
```

## Demos
**Note: To run notebook with Julia, you will need to add the package IJulia.**

1. State preparation: see the notebook 'demo-state-preparation.ipynb'. Several examples of state preparation are provided, including the all-one state, the uniform superposition state, and the 2-qubit Bell state.

2. Pulse fine tuning: see the notebook 'demo-fine-tuning.ipynb'.

3. Unnitary synthesis
TBA. 


## Documentations
The ansatz we use in **BloqadeControl** is as the following:
![](/docs/ansatz.png)

And the continuous-time pulses/waveforms are parametrized as
![](/docs/parametrization.png)

The total number of parameters is 2 x n_basis, in which the first half parameters are for the (global) Rabi frequency, and the second half is for the global detuning. The local detunings are specified by the user. 

TBA. 

