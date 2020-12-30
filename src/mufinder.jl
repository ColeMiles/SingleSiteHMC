using Statistics
using Plots
using Parameters
using Roots
using LaTeXStrings

struct Results
    μ_traj          :: Vector{Float64}
    N_traj          :: Vector{Float64}
    κ_traj          :: Vector{Float64}
    μ̄_traj          :: Vector{Float64}
    N̄_traj          :: Vector{Float64}
    κ̄_traj          :: Vector{Float64}
    phonon_pot_traj :: Vector{Float64}
    phonon_kin_traj :: Vector{Float64}
    N²_traj         :: Vector{Float64}
    accepts         :: Vector{Bool}
end

mutable struct MuTuner
    μ_traj      :: Vector{Float64}
    N_traj      :: Vector{Float64}
    N²_traj     :: Vector{Float64}
    κ̄_traj      :: Vector{Float64}
    forgetful_c :: Float64
    μ           :: Float64
    β           :: Float64
    target_N    :: Float64
    μ̄           :: Float64
    N̄           :: Float64
    N̄²          :: Float64  # Avg (N²), not (Avg N)²
    κ̄           :: Float64
    κ_min       :: Float64

    function MuTuner(init_μ::Float64, target_N::Float64, β::Float64, forgetful_c::Float64; κ_min::Float64=0.1)
        μ_traj = [init_μ]
        N_traj = Vector{Float64}()
        N²_traj = Vector{Float64}()
        κ̄_traj = Vector{Float64}()

        return new(
            μ_traj,
            N_traj,
            N²_traj,
            κ̄_traj,
            forgetful_c,
            init_μ,
            β,
            target_N,
            init_μ,
            -1.0,
            -1.0,
            0.0,
            κ_min
        )
    end
end

function estimate_μ(tuner::MuTuner)::Tuple{Float64,Float64}
    """ Returns the best guess at the correct μ, and an estimate
         of our uncertainty in this μ.
    """
    μ̄ = tuner.μ̄

    # Run through and reconstruct the N̄ trajectories
    N̄ = tuner.N_traj[1]
    κ̄ = tuner.κ̄_traj[1]
    N̄_traj = Vector{Float64}()
    sizehint!(N̄_traj, length(tuner.N_traj))
    for i in 1:length(tuner.N_traj)
        N̄ = forgetful_mean((@view tuner.N_traj[1:i]), tuner.c, N̄)
        push!(N̄_traj, N̄)
    end

    μ_corrections = @. (tuner.target_N - N̄_traj) / tuner.κ̄_traj
    err_μ = sqrt(mean(μ_corrections[c*length(μ_correction):end] .^ 2))
    return (μ̄, err_μ)
end

function update_μ!(tuner::MuTuner, N::Float64, N²::Float64) :: Float64
    """ Given a MuTuner, and a new set of measurements for N, N²,
         updates the MuTuner and returns the new value of μ.
    """
    @unpack μ_traj, N_traj, N²_traj, κ̄_traj, forgetful_c, β, target_N, κ_min = tuner

    tuner.μ̄ = forgetful_mean(μ_traj, forgetful_c, tuner.μ̄)

    push!(N_traj, N)
    tuner.N̄ = forgetful_mean(N_traj, forgetful_c, tuner.N̄)

    push!(N²_traj, N²)
    tuner.N̄² = forgetful_mean(N²_traj, forgetful_c, tuner.N̄²)

    κ̄ = β * (tuner.N̄² - tuner.N̄^2)
    push!(κ̄_traj, κ̄)
    tuner.κ̄ = κ̄
    κ_update = max(tuner.κ̄, κ_min / sqrt(length(κ̄_traj)))

    new_μ = tuner.μ̄ + (target_N - tuner.N̄) / κ_update
    tuner.μ = new_μ
    push!(μ_traj, new_μ)

    return new_μ
end

function update_μ!(tuner::MuTuner, R::Vector{Float64}, M⁻¹R::Vector{Float64}, M⁻ᵀR::Vector{Float64}) :: Float64
    """ Given a MuTuner, a random Gaussian vector R, and M⁻¹R, M⁻ᵀR,
         updates the MuTuner and returns the new value of μ.
    """
    L = length(R)
    N = (2 / L) * (R' * (R - M⁻¹R))
    N² = N^2 + (2 / L^2) * (-sum((2R - M⁻¹R - M⁻ᵀR).^2))
    return update_μ!(tuner, N, N²)
end

function cat_results(res_a::Results, res_b::Results)
    μ_traj = cat(res_a.μ_traj, res_b.μ_traj)
    N_traj = cat(res_a.N_traj, res_b.N_traj)
    κ_traj = cat(res_a.κ_traj, res_b.κ_traj)
    μ̄_traj = cat(res_a.μ̄_traj, res_b.μ̄_traj)
    N̄_traj = cat(res_a.N̄_traj, res_b.N̄_traj)
    κ̄_traj = cat(res_a.κ̄_traj, res_b.κ̄_traj)
    phonon_pot_traj = cat(res_a.phonon_pot_traj, res_b.phonon_pot_traj)
    phonon_kin_traj = cat(res_a.phonon_kin_traj, res_b.phonon_kin_traj)
    N²_traj = cat(res_a.N²_traj, res_b.N²_traj)
    accepts = cat(res_a.accepts, res_b.accepts)
    Results(
        μ_traj, N_traj, κ_traj,
        μ̄_traj, N̄_traj, κ̄_traj,
        phonon_pot_traj,
        phonon_kin_traj, N²_traj,
        accepts
    )
end

#= Linear-time "forgetful" mean and variance : For comparison =#

function forgetful_mean(data::AbstractVector{Float64}, c::Float64)
    cutoff = 1 + floor(Int64, (1.0 - c) * length(data))
    return mean(data[cutoff:end])
end

function forgetful_var(data::AbstractVector{Float64}, c::Float64)
    cutoff = 1 + floor(Int64, (1.0 - c) * length(data))
    return var(data[cutoff:end]; corrected=true)
end

#= Constant-time "forgetful" mean and variance =# 
#= These assume that only one update has happened since the last call =#

function forgetful_mean(data::AbstractVector{Float64}, c::Float64, prev_mean::Float64)
    # Short-circuit if this is the first element of the series
    if length(data) == 1
        return data[1]
    end

    cutoff = 1 + floor(Int64, (1.0 - c) * length(data))
    prev_cutoff = 1 + floor(Int64, (1.0 - c) * (length(data) - 1))

    new_mean = (length(data) - prev_cutoff) * prev_mean
    if prev_cutoff != cutoff
        new_mean -= data[prev_cutoff]
    end
    new_mean += data[end]

    return new_mean / (length(data) - cutoff + 1)
end

# Welford's online algorithm
function forgetful_mean_var(data::AbstractVector{Float64}, c::Float64, prev_mean::Float64, prev_var::Float64)
    # Short-circuit if this is the first element of the series
    if length(data) == 1
        return (data[1], 0.0)
    end

    cutoff = 1 + floor(Int64, (1.0 - c) * length(data))
    prev_cutoff = 1 + floor(Int64, (1.0 - c) * (length(data) - 1))

    new_pt = data[end]

    # Add the new point, update mean and sample variance
    new_length = length(data) - prev_cutoff + 1
    new_mean = prev_mean + (new_pt - prev_mean) / new_length
    new_var = prev_var * (new_length - 2) + (new_pt - prev_mean) * (new_pt - new_mean)
    new_var /= new_length - 1

    # If we need to drop a point off the back, update mean and sample variance again
    if prev_cutoff != cutoff
        new_length = length(data) - cutoff + 1
        drop_pt = data[prev_cutoff]
        prev_mean = new_mean
        prev_var = new_var

        new_mean = prev_mean - (drop_pt - prev_mean) / new_length
        new_var = prev_var * new_length - (drop_pt - prev_mean) * (drop_pt - new_mean)
        # Guard against new_length == 1
        if new_length == 1
            new_var = 0.0
        else
            new_var /= new_length - 1
        end
    end

    return (new_mean, new_var)
end

function find_μ(model::SSModel, dyn::AbstractDynamics, targetN::Float64, 
                nsteps::Int64; stochastic=false, n_sto_avg=1,
                forgetful_c::Float64=0.5, κ_min::Float64=nothing)
    """ Performs the full μ-finding algorithm to attempt to discover the μ
         which places the given model at the desired N. Progresses only for 
         the given number of steps.
             
        Returns a Results struct containing full traces of several observables.
    """
    β = model.Δτ * model.N

    tuner = MuTuner(model.μ, targetN, β, forgetful_c; κ_min=κ_min)

    μ̄_traj = Vector{Float64}()
    N̄_traj = Vector{Float64}()
    κ̄_traj = Vector{Float64}()
    phonon_pot_traj = Vector{Float64}()
    phonon_kin_traj = Vector{Float64}()
    N²_traj = Vector{Float64}()
    accepts = Vector{Bool}()
    sizehint!(μ̄_traj, nsteps)
    sizehint!(N̄_traj, nsteps)
    sizehint!(κ̄_traj, nsteps)
    sizehint!(phonon_pot_traj, nsteps)
    sizehint!(phonon_kin_traj, nsteps)
    sizehint!(N²_traj, nsteps)
    sizehint!(accepts, nsteps)

    μ̄ = model.μ
    N̄ = targetN
    κ̄ = 1.0
    κ_var = 0.0

    # Main μ-tuning algorithm
    for it in 1:nsteps
        accept = sample!(model, dyn)
        push!(accepts, accept)

        if stochastic
            (N, N²) = sto_measure_occupations(model; n_avg=n_sto_avg)
        else
            N = measure_mean_occupation(model)
            N² = measure_mean_sq_occupation(model)
        end

        new_μ = update_μ!(tuner, N, N²)
        model.μ = new_μ

        # Record a lot of measurements we want to track
        push!(μ̄_traj, tuner.μ̄)
        push!(N̄_traj, tuner.N̄)
        push!(κ̄_traj, tuner.κ̄)
        push!(phonon_pot_traj, measure_potential_energy(model))
        push!(phonon_kin_traj, measure_kinetic_energy(model))
        push!(N²_traj, N²)
    end

    return Results(
        tuner.μ_traj, tuner.N_traj, tuner.κ̄_traj,
        μ̄_traj, N̄_traj, κ̄_traj,
        phonon_pot_traj, phonon_kin_traj, N²_traj,
        accepts
    )
end

function const_μ_traj(model::SSModel, dyn::AbstractDynamics, nsteps::Int64; stochastic=false,
                      n_sto_avg=1, forgetful_c::Float64=0.5)
    """ Just like multistep_simulate, but records more details about observable trajectories.
    """
    β = model.N * model.Δτ
    μ = model.μ
    N̄ = measure_mean_occupation(model)
    N̄² = measure_mean_sq_occupation(model)
    κ̄ = 1.0
    κ_var = 0.0

    μ_traj = fill(model.μ, nsteps)
    N_traj = Vector{Float64}()
    κ_traj = Vector{Float64}()
    μ̄_traj = fill(model.μ, nsteps)
    N̄_traj = Vector{Float64}()
    κ̄_traj = Vector{Float64}()
    phonon_pot_traj = Vector{Float64}()
    phonon_kin_traj = Vector{Float64}()
    N²_traj = Vector{Float64}()
    accept_traj = Vector{Bool}()
    sizehint!(N_traj, nsteps+1)
    sizehint!(κ_traj, nsteps+1)
    sizehint!(N̄_traj, nsteps+1)
    sizehint!(κ̄_traj, nsteps+1)
    sizehint!(phonon_pot_traj, nsteps+1)
    sizehint!(phonon_kin_traj, nsteps+1)
    sizehint!(N²_traj, nsteps+1)
    sizehint!(accept_traj, nsteps+1)

    for step in 1:nsteps
        accept = sample!(model, dyn)

        if stochastic
            (N, N²) = sto_measure_occupations(model; n_avg=n_sto_avg)
        else
            N = measure_mean_occupation(model)
            N² = measure_mean_sq_occupation(model)
        end

        push!(N_traj, N)
        push!(N²_traj, N²)
        N̄ = forgetful_mean(N_traj, forgetful_c, N̄)
        N̄² = forgetful_mean(N²_traj, forgetful_c, N̄²)
        push!(N̄_traj, N̄)
        push!(κ_traj, N - N²)
        push!(κ̄_traj, N̄² - N̄^2)


        push!(phonon_pot_traj, measure_potential_energy(model))
        push!(phonon_kin_traj, measure_kinetic_energy(model))
        push!(accept_traj, accept)
    end

    return Results(
        μ_traj, N_traj, κ_traj,
        μ̄_traj, N̄_traj, κ̄_traj,
        phonon_pot_traj, phonon_kin_traj, N²_traj,
        accept_traj
    )
end

function plot_μ_finding(N_target::Float64, μ_target::Float64, κ_target::Float64,
                        results::Results; instant=false)
    @unpack μ_traj, N_traj, κ_traj, μ̄_traj, N̄_traj, κ̄_traj = results
    nsteps = length(μ̄_traj)

    plot(xlims=(1, nsteps), xscale=:log10, xlabel="Step", size=(800, 600))

    # if instant
    #     plot!(κ_traj, label=L"\kappa", linecolor=:lightgreen)
    # end
    plot!(κ̄_traj, label=L"\bar{\kappa}", linecolor=:darkgreen)
    hline!([κ_target], linestyle=:dash,
          label=L"\kappa^\ast", linecolor=:limegreen, lw=2)

    if instant
        plot!(N_traj, label=L"N", linecolor=:red)
    end
    plot!(N̄_traj, label=L"\bar{N}", linecolor=:darkred)
    hline!([N_target], linestyle=:dash,
          label=L"N^\ast", linecolor=:red, lw=2)

    if instant
        plot!(μ_traj, label=L"\mu", linecolor=:lightblue)
    end
    plot!(μ̄_traj, label=L"\bar{\mu}", linecolor=:navyblue)
    hline!([μ_target], linestyle=:dash,
          label=L"\mu^\ast", linecolor=:purple, lw=2)
    return plot!()
end

function test_μ_find(;seed::Int64=12345, β=2., ω₀=1., λ=√2, μinit=-1.0, m_reg=0.4,
                      dt=0.8, nsteps=4, nfaststeps=8, N_target=1.0, nsearch=500000, plot=true,
                      stochastic=false, n_sto_avg=1, forgetful_c=0.5, κ_min=0.1)
    Δτ_target = 0.1
    N = round(Int, β/Δτ_target)
    Δτ = β / N
    if Δτ ≠ Δτ_target
        println("Modified Δτ from $Δτ_target to $Δτ")
    end

    model = SSModel(seed=seed, N=N, Δτ=Δτ, ω₀=ω₀, λ=λ, μ=μinit, m_reg=m_reg)
    randomize!(model)

    dyn = MultiStepFAHamiltonianDynamics(
        ;N=N, steps=nsteps, faststeps=nfaststeps,
        dt=dt
    )

    results = find_μ(model, dyn, N_target, nsearch,
                     stochastic=stochastic, n_sto_avg=n_sto_avg,
                     forgetful_c=forgetful_c, κ_min=κ_min)

    μ_target = numeric_μ(β, N_target; ω₀=ω₀, λ=λ)
    κ_target = analytic_κ(β, μ_target; ω₀=ω₀, λ=λ)
    if plot
        plot_μ_finding(N_target, μ_target, κ_target, results)
    end

    return results
end

function analytic_n(β, μ; ω₀=1., λ=√2)
    μ₀ = -λ^2 / ω₀^2
    numer = 2 * exp(β * (μ - μ₀/2)) + 2 * exp(2 * β * (μ - μ₀))
    denom = 1 + 2 * exp(β * (μ - μ₀/2)) + exp(2 * β * (μ - μ₀))
    return numer / denom
end

function analytic_κ(β, μ; ω₀=1., λ=√2)
    exp1 = exp(β * (λ^2 / (2 * ω₀^2) + μ))
    exp2 = exp(2 * β * (λ^2 / ω₀^2 + μ))
    numer = 2β * ((1 + 2exp1 + exp2) * (exp1 + 2exp2) - (2exp1 + 2exp2) * (exp1 + exp2))
    denom = (1 + 2exp1 + exp2)^2
    return numer / denom
end

function numeric_μ(β, n; ω₀=1., λ=√2, left=-10, right=10)
    f = μ -> analytic_n(β, μ; ω₀=ω₀, λ=λ) - n
    res_μ = find_zero(f, (left, right), Bisection())
    return res_μ
end
