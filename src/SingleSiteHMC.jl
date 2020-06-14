module SingleSiteHMC

include("model.jl")
include("dynamics.jl")
include("measurements.jl")
include("plotting.jl")
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

# Full runs

function simulate(; seed::Int, dt=0.1, nsteps=10, m_reg=0.4)
    β = 2
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

    run_samples = 250000
    total_num_rejects = 0

    pot_meas = zeros(Float64, run_samples)
    kin_meas = zeros(Float64, run_samples)
    occ_meas = zeros(Float64, run_samples)
    occ_sq_meas = zeros(Float64, run_samples)
    doub_occ_meas = zeros(Float64, run_samples)

    for step in 1:run_samples
        total_num_rejects += !sample!(model, dyn)

        pot_meas[step] = measure_potential_energy(model)
        kin_meas[step] = measure_kinetic_energy(model)
        occ_meas[step] = measure_mean_occupation(model)
        occ_sq_meas[step] = measure_mean_sq_occupation(model)
        doub_occ_meas[step] = measure_double_occupation(model)
    end

    observable_names = ["Phonon Potential", "Phonon Kinetic", "Occupancy", "Occupancy²", "Doub. Occupancy"]
    measurements = [pot_meas, kin_meas, occ_meas, occ_sq_meas, doub_occ_meas]
    exact_functions = [mean_potential, mean_kinetic, mean_occupancy, mean_occupancy_sq, mean_double_occupancy]

    for (obs, meas, exact) in zip(observable_names, measurements, exact_functions)
        (mu, sigma) = binned_statistics(meas, nbins)
        println(obs)
        println("\tMeasured: $mu ± $sigma")
        println("\tExact: $(exact(model))")
    end
    println("Average Acceptance Probability: $((run_samples - total_num_rejects) / run_samples)")
end

function multistep_simulate(; seed::Int, dt=0.1, nsteps=10, nfaststeps=4, m_reg=0.4)
    β = 2
    Δτ_target = 0.1
    N = round(Int, β/Δτ_target)
    Δτ = β / N
    if Δτ ≠ Δτ_target
        println("Modified Δτ from $Δτ_target to $Δτ")
    end
    nbins = 10

    model = SSModel(seed=seed, N=N, Δτ=Δτ, ω₀=1., λ=√2, μ=-2.5, m_reg=m_reg)
    randomize!(model)

    integrator_dt = 0.1
    integrator_nsteps = 10
    integrator_faststeps = 10
    dyn = MultiStepFAHamiltonianDynamics(
        ;N=N, steps=nsteps, faststeps=nfaststeps,
        dt=dt
    )

    burnin_samples = 50000
    for step in 1:burnin_samples
        sample!(model, dyn)
    end

    run_samples = 250000
    total_num_rejects = 0

    pot_meas = zeros(Float64, run_samples)
    kin_meas = zeros(Float64, run_samples)
    occ_meas = zeros(Float64, run_samples)
    occ_sq_meas = zeros(Float64, run_samples)
    doub_occ_meas = zeros(Float64, run_samples)

    for step in 1:run_samples
        total_num_rejects += !sample!(model, dyn)

        pot_meas[step] = measure_potential_energy(model)
        kin_meas[step] = measure_kinetic_energy(model)
        occ_meas[step] = measure_mean_occupation(model)
        occ_sq_meas[step] = measure_mean_sq_occupation(model)
        doub_occ_meas[step] = measure_double_occupation(model)
    end

    observable_names = ["Phonon Potential", "Phonon Kinetic", "Occupancy", "Occupancy²", "Doub. Occupancy"]
    measurements = [pot_meas, kin_meas, occ_meas, occ_sq_meas, doub_occ_meas]
    exact_functions = [mean_potential, mean_kinetic, mean_occupancy, mean_occupancy_sq, mean_double_occupancy]

    for (obs, meas, exact) in zip(observable_names, measurements, exact_functions)
        (mu, sigma) = binned_statistics(meas, nbins)
        println(obs)
        println("\tMeasured: $mu ± $sigma")
        println("\tExact: $(exact(model))")
    end
    println("Average Acceptance Probability: $((run_samples - total_num_rejects) / run_samples)")
end

end