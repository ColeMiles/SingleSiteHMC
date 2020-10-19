module SingleSiteHMC

include("model.jl")
include("dynamics.jl")
include("measurements.jl")
include("plotting.jl")
include("mufinder.jl")
include("tests.jl")

# Export some stuff

export SSModel, randomize!, S_bose, S_inter, S_kinetic, S_total, update_Bτ!
export HamiltonianDynamics, HamiltonianFADynamics
export MultiStepHamiltonianDynamics, MultiStepFAHamiltonianDynamics
export sample!
export simulate, multistep_simulate
export measure_potential_energy, measure_mean_occupation, measure_mean_sq_occupation
export measure_kinetic_energy, measure_double_occupation
export mean_potential, mean_occupancy, mean_occupancy_sq, mean_double_occupancy, mean_kinetic
export find_μ, plot_μ_finding

# Full runs

function simulate(; seed::Int, β=2., dt=0.1, nsteps=10, m_reg=0.4)
    Δτ_target = 0.1
    N = round(Int, β/Δτ_target)
    Δτ = β / N
    if Δτ ≠ Δτ_target
        println("Modified Δτ from $Δτ_target to $Δτ")
    end
    nbins = 10

    model = SSModel(seed=seed, N=N, Δτ=Δτ, ω₀=1., λ=√2, μ=-2.5, m_reg=m_reg)
    randomize!(model)

    dyn = HamiltonianFADynamics(; N=N, steps=nsteps, dt=dt)

    burnin_samples = 50000
    for step in 1:burnin_samples
        sample!(model, dyn)
    end

    run_samples = 2000000
    total_num_rejects = 0

    pot_meas = zeros(Float64, run_samples)
    kin_meas = zeros(Float64, run_samples)
    occ_meas = zeros(Float64, run_samples)
    occ_sq_meas = zeros(Float64, run_samples)
    doub_occ_meas = zeros(Float64, run_samples)
    green_meas = zeros(Float64, N, run_samples)

    for step in 1:run_samples
        total_num_rejects += !sample!(model, dyn)

        pot_meas[step] = measure_potential_energy(model)
        kin_meas[step] = measure_kinetic_energy(model)
        occ_meas[step] = measure_mean_occupation(model)
        occ_sq_meas[step] = measure_mean_sq_occupation(model)
        doub_occ_meas[step] = measure_double_occupation(model)
        green_meas[1:end, step] .= measure_electron_greens(model)
    end

    observable_names = ["Phonon Potential", "Phonon Kinetic", "Occupancy", "Occupancy²", "Doub. Occupancy"]
    measurements = [pot_meas, kin_meas, occ_meas, occ_sq_meas, doub_occ_meas]
    exact_functions = [mean_potential, mean_kinetic, mean_occupancy, mean_occupancy_sq, mean_double_occupancy]

    accept_prob = (run_samples - total_num_rejects) / run_samples
    measurement_dict = Dict{String, Any}()

    # Measurement reporting
    for (obs, meas, exact) in zip(observable_names, measurements, exact_functions)
        (mu, sigma) = binned_statistics(meas, nbins)
        measurement_dict[obs] = (mu, sigma)
        println(obs)
        println("\tMeasured: $mu ± $sigma")
        println("\tExact: $(exact(model))")
    end
    println("Average Acceptance Probability: $accept_prob")

    # Green's measurement reporting
    G_μ, G_σ = zeros(N), zeros(N)
    println("Green's Functions")
    println("\tΔ\tG(Δ)\t\t\t\t\t\tExact")
    for Δ in 0:N-1
        GΔ = green_meas[Δ+1, 1:end]
        (mu, sigma) = binned_statistics(GΔ, nbins)
        G_μ[Δ+1] = mu
        G_σ[Δ+1] = sigma
        exact = exact_greens(model, Δ * Δτ)
        println("\t$(Δ)\t$(mu) ± $(sigma)\t$(exact)")
    end
    measurement_dict["Electron Greens"] = (G_μ, G_σ)

    return accept_prob, measurement_dict
end

function multistep_simulate(; seed::Int, β=2., μ=-2.5, dt=0.1, nsteps=10, nfaststeps=4,
                              m_reg=0.4, use_fa=true, dτ_target=0.1, run_samples=500000)
    N = round(Int, β/dτ_target)
    Δτ = β / N
    if Δτ ≠ dτ_target
        println("Modified Δτ from $dτ_target to $Δτ")
    end
    nbins = 10

    model = SSModel(seed=seed, N=N, Δτ=Δτ, ω₀=1., λ=√2, μ=μ, m_reg=m_reg)
    randomize!(model)

    if use_fa
        dyn = MultiStepFAHamiltonianDynamics(
            ;N=N, steps=nsteps, faststeps=nfaststeps,
            dt=dt
        )
    else
        dyn = MultiStepHamiltonianDynamics(
            ;N=N, steps=nsteps, faststeps=nfaststeps,
            dt=dt
        )
    end

    burnin_samples = 100000
    for step in 1:burnin_samples
        sample!(model, dyn)
    end

    total_num_rejects = 0

    accepts = zeros(Bool, run_samples)
    pot_meas = zeros(Float64, run_samples)
    kin_meas = zeros(Float64, run_samples)
    occ_meas = zeros(Float64, run_samples)
    occ_sq_meas = zeros(Float64, run_samples)
    doub_occ_meas = zeros(Float64, run_samples)
    electron_green_meas = zeros(Float64, N, run_samples)
    phonon_green_meas = zeros(Float64, N, run_samples)

    for step in 1:run_samples
        accepts[step] = sample!(model, dyn)
        pot_meas[step] = measure_potential_energy(model)
        kin_meas[step] = measure_kinetic_energy(model)
        occ_meas[step] = measure_mean_occupation(model)
        occ_sq_meas[step] = measure_mean_sq_occupation(model)
        doub_occ_meas[step] = measure_double_occupation(model)
        electron_green_meas[1:end, step] .= measure_electron_greens(model)
        phonon_green_meas[1:end, step] .= measure_phonon_greens(model)
    end

    observable_names = ["Phonon Potential", "Phonon Kinetic", "Occupancy", "Occupancy²", "Doub. Occupancy"]
    measurements = [pot_meas, kin_meas, occ_meas, occ_sq_meas, doub_occ_meas]
    exact_functions = [mean_potential, mean_kinetic, mean_occupancy, mean_occupancy_sq, mean_double_occupancy]

    accept_prob = sum(accepts) / run_samples
    measurement_dict = Dict{String, Any}()

    for (obs, meas, exact) in zip(observable_names, measurements, exact_functions)
        (mu, sigma) = binned_statistics(meas, nbins)
        measurement_dict[obs] = (mu, sigma)
        println(obs)
        println("\tMeasured: $mu ± $sigma")
        println("\tExact: $(exact(model))")
    end
    println("Average Acceptance Probability: $accept_prob")

    # Electron Green's measurement reporting
    G_μ, G_σ = zeros(N), zeros(N)
    println("Electron Green's Functions")
    println("\tΔ\tG(Δ)\t\t\t\t\t\tExact")
    for Δ in 0:N-1
        GΔ = electron_green_meas[Δ+1, 1:end]
        (mu, sigma) = binned_statistics(GΔ, nbins)
        G_μ[Δ+1] = mu
        G_σ[Δ+1] = sigma
        exact = exact_greens(model, Δ * Δτ)
        println("\t$(Δ)\t$(mu) ± $(sigma)\t$(exact)")
    end
    measurement_dict["Electron Greens"] = (copy(G_μ), copy(G_σ))

    # Phonon Green's measurement reporting
    println("Phonon Green's Functions")
    println("\tΔ\tG(Δ)\t\t\t\t\t\tExact")
    for Δ in 0:N-1
        GΔ = phonon_green_meas[Δ+1, 1:end]
        (mu, sigma) = binned_statistics(GΔ, nbins)
        G_μ[Δ+1] = mu
        G_σ[Δ+1] = sigma
        println("\t$(Δ)\t$(mu) ± $(sigma)\t?")
    end
    measurement_dict["Phonon Greens"] = (copy(G_μ), copy(G_σ))

    # Package full trajectories into "Results" structure
    # For comparison to μ-tuning, compute some extra statistics
    res = create_results(
        μ, β, occ_meas, pot_meas, kin_meas, occ_sq_meas, accepts
    )

    return accept_prob, measurement_dict, res
end

function create_results(μ, β, N_traj, pot_traj, kin_traj, N²_traj, accepts; c=0.5)
    """ Given statistics from a fixed-μ simulation, create a Results object
         with statistics comparable to a μ-tuning simulation
    """
    N̄ = N_traj[1]
    κ̄ = 0.0
    num_samples = length(N_traj)
    μ_traj = fill(μ, num_samples)
    μ̄_traj = fill(μ, num_samples)
    N̄_traj = Vector{Float64}()
    κ_traj = Vector{Float64}()
    κ̄_traj = Vector{Float64}()
    sizehint!(N̄_traj, num_samples)
    sizehint!(κ_traj, num_samples)
    sizehint!(κ̄_traj, num_samples)

    for i in 1:num_samples
        N̄ = forgetful_mean(N_traj[1:i], c, N̄)
        push!(N̄_traj, N̄)
        κ = β * (N²_traj[i] - 2 * N_traj[i] * N̄ + N̄^2)
        push!(κ_traj, κ)
        κ̄ = forgetful_mean(κ_traj[1:i], c, κ̄)
        push!(κ̄_traj, κ̄)
    end

    res = Results(
        μ_traj,
        N_traj,
        κ_traj,
        μ̄_traj,
        N̄_traj,
        κ̄_traj,
        pot_traj,
        kin_traj,
        N²_traj,
        accepts
    )

    return res
end

end
