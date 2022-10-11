from pickle import FALSE
from signal import SIGRTMAX
from qutip import *
import numpy as np
from math import *

def heis_1d_H(_N):
    zero_qubit = qeye(2) - qeye(2)
    hamilt_list = [qeye(2)] * _N
    H = tensor([zero_qubit] * _N)
    component_list = [sigmaz(), sigmax(), sigmay()]
    for i in range(_N-1):
        for sigma in component_list:
            h_list = hamilt_list.copy()
            h_list[i] = sigma
            h_list[i+1] = sigma
            H += tensor(h_list)
    return H

def heis_cycle_H(_N):
    zero_qubit = qeye(2) - qeye(2)
    hamilt_list = [qeye(2)] * _N
    H = tensor([zero_qubit] * _N)
    component_list = [sigmaz(), sigmax(), sigmay()]
    edge_list = [(0, _N - 1)]
    for i in range(_N - 1):
        edge_list.append((i, i+1))
    for edge in edge_list:
        for sigma in component_list:
            h_list = hamilt_list.copy()
            h_list[edge[0]] = sigma
            h_list[edge[1]] = sigma
            H += tensor(h_list)
    return H    

psi0 = tensor([basis(2, 1), basis(2, 0)] * 3)
H = heis_cycle_H(N)
time = 1.8
times = [0.0, time]
result = mesolve(H, psi0, times, [], [])
state0 = result.states[1]
print(state0, H)
print(expect(tensor([sigmaz()] + [qeye(2)] * (N - 1)), state0)) 