# Differentiable Analog Pulse Learning for Rydberg Atoms

We introduce **BloqadeControl**, a Julia module developed for differentiable analog quantum computing on neutral-atom quantum computers. We use **Bloqade** from QuEraComputing to simulate Rydberg atoms. 

For more details, see [this note](https://www.overleaf.com/read/vcgrwrkrrgqp).

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