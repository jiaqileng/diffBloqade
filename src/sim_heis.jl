using Bloqade
function foo(τ, θ, n) #Simulate 1-d Heisenberg exp(iθ(XX+YY+ZZ)) in time τ with 1~20 oscillations
    local nsites, r, f, c, t, s, d, atoms, reg, h, prob, integrator, list, i
    list = []
    C6 = 862690 * 2 * π
    nsites = n
    r = θ / τ * 12.0
    d = (C6 / r)^(1/6)
    R = (d / 2) / sin(π / n)
    atom_list = [(0.0, 0.0)]
    for i in 1:n
        push!(atom_list, (R * cos(2 * π * i / n), R * sin(2 * π * i / n)))
    end
    atom_list = last(atom_list, n)
    println(atom_list)
    for m in 1:20
        s = m * 2 * π / τ
        c = 0.28867513 * s
        # atoms = generate_sites(ChainLattice(), nsites, scale = d)
        atoms = AtomList(atom_list)
        reg = product_state(bit"101010")
        h = rydberg_h(atoms; Δ = 2 * c + 0.5 * r * 2, Ω = 2 * sqrt(2) * c, ϕ = π * 1.25)
        prob = SchrodingerProblem(reg, τ, h)
        integrator = init(prob, Vern8());
        emulate!(prob);
        push!(list, expect(put(n,1=>Z), prob.reg))
    end
    return list
end

print(foo(1.0, 1.8, 6))