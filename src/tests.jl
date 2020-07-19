function test_hubbard_matrix()
    m = SSModel(seed=0, N=10, Δτ=0.1, ω₀=1., λ=sqrt(2), μ=-2.)
    randomize!(m)

    r = randn(m.rng, m.N)
    v1 = copy(r)
    v2 = similar(r)

    apply_hubbard_matrix(m, r, v1, transpose=false)
    apply_hubbard_inverse(m, v1, v2, transpose=false)
    println("checking inverse: ", r ≈ v2)

    @. v1 = r
    apply_hubbard_matrix(m, r, v1, transpose=true)
    apply_hubbard_inverse(m, v1, v2, transpose=true)
    println("checking inverse transpose: ", r ≈ v2)

    @. v1 = r
    apply_hubbard_inverse(m, r, v1, transpose=true)
    apply_hubbard_matrix(m, v1, v2, transpose=true)
    println("checking reverse order: ", r ≈ v2)
end

function calc_F_brute!(model::SSModel, dyn::AbstractDynamics)
    """ Simplest way possible to calculate the HMC force by explicitly
         constructing M, dM/dx and performing all matrix multiplications
         directly.
    """
    @unpack F, x = dyn

    update_Bτ!(model)

    function make_M(m::SSModel)
        M = zeros(Float64, m.N, m.N)
        for τ in 1:m.N
            M[τ, τ] = 1.0
            if τ == 1
                M[τ, m.N] = m.B[τ]
            else
                M[τ, τ-1] = -m.B[τ]
            end
        end
        return M
    end
    function make_dM(m::SSModel, τ::Int)
        dM = zeros(Float64, m.N, m.N)
        if τ == 1
            dM[1, m.N] = -m.Δτ * m.λ * m.B[τ]
        else
            dM[τ, τ-1] = m.Δτ * m.λ * m.B[τ]
        end
        return dM
    end

    # Zero out forces
    fill!(F, 0.0)

    ### Add phonon part of force
    for τ = 1:model.N
        x  = model.x[τ]
        x′ = model.x[mod1(τ+1, model.N)]
        x″ = model.x[mod1(τ-1, model.N)]
        F[τ] += -model.Δτ * (
            model.ω₀^2 * x                # phonon potential
            - (x′ - 2x + x″) / model.Δτ^2  # phonon kinetic 
        )
    end

    ### Auxiliary field forces
    M = make_M(model)
    O = inv(M' * M)

    for τ = 1:model.N
        dM = make_dM(model, τ)
        F[τ] += ϕ₁' * O * M' * dM * O * ϕ₁
        F[τ] += ϕ₂' * O * M' * dM * O * ϕ₂
    end
end

function test_force!(model::SSModel, dyn::AbstractDynamics)
    for _ in 1:100
        randomize!(model)
        calc_F_brute!(model, dyn)
        Fbrute = copy(model.F)
        calc_F!(model, dyn)
        F = copy(model.F)
        @assert F ≈ Fbrute
    end
end

function test_energy_conservation(model::SSModel, dyn::MultiStepFAHamiltonianDynamics, nsteps=30)
    @unpack dt, faststeps, ϕ₁, ϕ₂, F, x, p, Qp, R₁, R₂ = dyn

    # Perform a burn-in first
    burnin_samples = 25000
    for step in 1:burnin_samples
        sample!(model, dyn)
    end

    update_Bτ!(model)

    # Sample an auxiliary field configurations ϕ₁, ϕ₂
    randn!(model.rng, R₁)
    apply_hubbard_matrix(model, R₁, ϕ₁; transpose=true)
    randn!(model.rng, R₂)
    apply_hubbard_matrix(model, R₂, ϕ₂; transpose=true)

    # Sample an initial momentum configuration
    randn!(model.rng, p)
    fourier_accelerate!(model, p, -0.5)

    # Calculate forces, and update B's, ψ's
    calc_F_aux_fields!(model, dyn)
    copy!(Qp, p)
    fourier_accelerate!(model, Qp, 1.0)

    Es = Vector{Float64}()
    Eboses = Vector{Float64}()
    Ekins = Vector{Float64}()
    Eints = Vector{Float64}()
    push!(Es, S_total(model, dyn))

    fastdt = dt / faststeps

    # Evolve x, p under Hamiltonian dynamics using the leapfrog integrator
    for step in 1:nsteps
        # Step interaction hamiltonian
        @. p += F * dt / 2

        # Perform a sub-leapfrog integration on the phonon action for faststeps # of steps
        calc_F_phonon!(model, dyn)

        for faststep in 1:faststeps
             @. p += F * fastdt / 2

            copy!(Qp, p)
            fourier_accelerate!(model, Qp, 1.0)
            @. model.x += Qp * fastdt

            calc_F_phonon!(model, dyn)

            @. p += F * fastdt / 2
        end

        # Step interaction hamiltonian
        calc_F_aux_fields!(model, dyn)
        @. p += F * dt / 2

        # Re-update Qp for Hamiltonian evaluation
        copy!(Qp, p)
        fourier_accelerate!(model, Qp, 1.0)

        push!(Es, S_total(model, dyn))
        push!(Eboses, S_bose(model, dyn))
        push!(Ekins, S_kinetic(model, dyn))
        push!(Eints, S_inter(model, dyn))
    end
    return Es, Eboses, Ekins, Eints
end

function test_stochastic_measurements(;seed::Int, β=2., μ=-2.0, dt=0.1, nsteps=10, nfaststeps=4, m_reg=0.4)
    Δτ_target = 0.1
    N = round(Int, β/Δτ_target)
    Δτ = β / N
    if Δτ ≠ Δτ_target
        println("Modified Δτ from $Δτ_target to $Δτ")
    end
    nbins = 10

    model = SSModel(seed=seed, N=N, Δτ=Δτ, ω₀=1., λ=√2, μ=μ, m_reg=m_reg)
    randomize!(model)

    dyn = MultiStepFAHamiltonianDynamics(
        ;N=N, steps=nsteps, faststeps=nfaststeps,
        dt=dt
    )

    burnin_samples = 50000
    for step in 1:burnin_samples
        sample!(model, dyn)
    end

    run_samples = 500000
    total_num_rejects = 0

    occ_meas = zeros(Float64, run_samples)
    occ_sq_meas = zeros(Float64, run_samples)
    sto_occ_meas = zeros(Float64, run_samples)
    sto_occ_sq_meas = zeros(Float64, run_samples)

    for step in 1:run_samples
        total_num_rejects += !sample!(model, dyn)

        occ_meas[step] = measure_mean_occupation(model)
        occ_sq_meas[step] = measure_mean_sq_occupation(model)
        (sto_N, sto_N²) = sto_measure_occupations(model)
        sto_occ_meas[step] = sto_N
        sto_occ_sq_meas[step] = sto_N²
    end

    observable_names = ["Occupancy", "Occupancy²", "Sto. Occupancy", "Sto. Occupancy²"]
    measurements = [occ_meas, occ_sq_meas, sto_occ_meas, sto_occ_sq_meas]
    exact_functions = [mean_occupancy, mean_occupancy_sq, mean_occupancy, mean_occupancy_sq]

    accept_prob = (run_samples - total_num_rejects) / run_samples
    measurement_dict = Dict{String, Tuple{Float64, Float64}}()

    for (obs, meas, exact) in zip(observable_names, measurements, exact_functions)
        (mu, sigma) = binned_statistics(meas, nbins)
        measurement_dict[obs] = (mu, sigma)
        println(obs)
        println("\tMeasured: $mu ± $sigma")
        println("\tExact: $(exact(model))")
    end
    println("Average Acceptance Probability: $accept_prob")
end

function test_stochastic_correctness(;seed::Int, β=2., n_avg=10)
    dt = 0.1
    nsteps = 10
    nfaststeps = 4
    m_reg = 0.4
    μ_init = -1.0
end