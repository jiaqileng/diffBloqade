# Differentiable Analog Pulse Learning for Rydberg Atoms

This project is an extension of my previous work *Differentiable Analog Quantum Computing for Optimization and Control*. [[ArXiv]](https://arxiv.org/abs/2210.15812)

We introduce **BloqadeControl**, a Julia module developed for differentiable analog quantum computing on neutral-atom quantum computers. We use **Bloqade** from QuEraComputing to simulate Rydberg atoms. 

For more technical details, see [this note](https://www.overleaf.com/read/vcgrwrkrrgqp).

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

1. Comparison between finite difference method and our method for gradient computation: see the notebook 'demo-compare-grad.ipynb'. As the number of samples increases, our method (based on Monte Carlo sampling) will converge to the true gradient. We also observe that, even with a small number of samples, the direction of the estimated gradient is very close to the true gradient!

2. State preparation: see the notebook 'demo-state-preparation.ipynb'. Several examples of state preparation are provided, including the all-one state, the uniform superposition state, and the 2-qubit Bell state.

3. Pulse fine tuning: see the notebook 'demo-fine-tuning.ipynb'.

4. Unitary synthesis
TBA. 
