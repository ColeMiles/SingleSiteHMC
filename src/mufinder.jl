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
    κ_var_traj      :: Vector{Float64}
    phonon_pot_traj :: Vector{Float64}
    phonon_kin_traj :: Vector{Float64}
    N²_traj         :: Vector{Float64}
    accepts         :: Vector{Bool}
end

#= Linear-time "forgetful" mean and variance : For comparison =#

function forgetful_mean(data::Vector{Float64}, c::Float64)
    cutoff = ceil(Int64, c * length(data))
    return mean(data[cutoff:end])
end

function forgetful_var(data::Vector{Float64}, c::Float64)
    cutoff = ceil(Int64, c * length(data))
    return var(data[cutoff:end]; corrected=true)
end

#= Constant-time "forgetful" mean and variance =# 
#= These assume that only one update has happened since the last call =#

function forgetful_mean(data::Vector{Float64}, c::Float64, prev_mean::Float64)
    cutoff = ceil(Int64, c * length(data))
    prev_cutoff = ceil(Int64, c * (length(data) - 1))

    new_mean = (length(data) - prev_cutoff) * prev_mean
    if prev_cutoff != cutoff
        new_mean -= data[prev_cutoff]
    end
    new_mean += data[end]

    return new_mean / (length(data) - cutoff + 1)
end

# Welford's online algorithm
function forgetful_mean_var(data::Vector{Float64}, c::Float64, prev_mean::Float64, prev_var::Float64)
    cutoff = ceil(Int64, c * length(data))
    prev_cutoff = ceil(Int64, c * (length(data) - 1))

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
        new_var /= new_length - 1
    end

    return (new_mean, new_var)
end

# Dummy mean function with extra argument to match what's expected by find_μ
function dummy_mean(data::Vector{Float64}, prev_val::Float64)
    return mean(data)
end

function find_μ(model::SSModel, dyn::AbstractDynamics, targetN::Float64, 
                nsteps::Int64; ninit::Int64=10, stochastic=false, n_sto_avg=1,
                forgetful_c::Float64=0.5)
    """ Performs the full μ-finding algorithm to attempt to discover the μ
         which places the given model at the desired N. Progresses only for 
         the given number of steps.
             
        Returns a Results struct containing full traces of several observables.
    """
    β = model.Δτ * model.N

    # Initialization routine
    μ_traj = fill(model.μ, ninit)
    N_traj = fill(targetN, ninit)
    κ_traj = fill(1.0, ninit)
    sizehint!(μ_traj, ninit+nsteps)
    sizehint!(N_traj, ninit+nsteps)
    sizehint!(κ_traj, ninit+nsteps)

    μ̄_traj = Vector{Float64}()
    N̄_traj = Vector{Float64}()
    κ̄_traj = Vector{Float64}()
    κ_var_traj = Vector{Float64}()
    phonon_pot_traj = Vector{Float64}()
    phonon_kin_traj = Vector{Float64}()
    N²_traj = Vector{Float64}()
    accepts = Vector{Bool}()
    sizehint!(μ̄_traj, nsteps)
    sizehint!(N̄_traj, nsteps)
    sizehint!(κ̄_traj, nsteps)
    sizehint!(κ_var_traj, nsteps)
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
        μ̄ = forgetful_mean(μ_traj, forgetful_c, μ̄)
        N̄ = forgetful_mean(N_traj, forgetful_c, N̄)
        (κ̄, κ_var) = forgetful_mean_var(κ_traj, forgetful_c, κ̄, κ_var)

        push!(μ̄_traj, μ̄)
        push!(N̄_traj, N̄)
        push!(κ̄_traj, κ̄)
        push!(κ_var_traj, κ_var)

        new_μ = μ̄ + (targetN - N̄) / max(κ̄, sqrt(κ_var) / sqrt(length(κ̄_traj)))
        # new_μ = μ̄ + (targetN - N̄) / κ̄
        model.μ = new_μ
        push!(μ_traj, new_μ)

        accept = sample!(model, dyn)
        push!(accepts, accept)

        if stochastic
            (N, N²) = sto_measure_occupations(model; n_avg=n_sto_avg)
        else
            N = measure_mean_occupation(model)
            N² = measure_mean_sq_occupation(model)
        end

        push!(N_traj, N)
        push!(κ_traj, β * (N² - 2 * N * N̄ + N̄^2))
        push!(phonon_pot_traj, measure_potential_energy(model))
        push!(phonon_kin_traj, measure_kinetic_energy(model))
        push!(N²_traj, N²)
    end

    return Results(
        μ_traj, N_traj, κ_traj,
        μ̄_traj, N̄_traj, κ̄_traj, κ_var_traj,
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
    κ̄ = 1.0
    κ_var = 0.0

    μ_traj = fill(model.μ, nsteps)
    N_traj = [N̄]
    κ_traj = [κ̄]
    μ̄_traj = fill(model.μ, nsteps)
    N̄_traj = [N̄]
    κ̄_traj = [κ̄]
    κ_var_traj = [0.0]
    phonon_pot_traj = [measure_potential_energy(model)]
    phonon_kin_traj = [measure_kinetic_energy(model)]
    N²_traj = [measure_mean_sq_occupation(model)]
    accept_traj = [true]
    sizehint!(N_traj, nsteps+1)
    sizehint!(κ_traj, nsteps+1)
    sizehint!(N̄_traj, nsteps+1)
    sizehint!(κ̄_traj, nsteps+1)
    sizehint!(κ̄_var_traj, nsteps)
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
        κ = β * (N² - 2 * N * N̄ + N̄^2)
        push!(N_traj, N)
        push!(κ_traj, β * (N² - 2 * N * N̄ + N̄^2))

        N̄ = forgetful_mean(N_traj, forgetful_c, N̄)
        (κ̄, κ_var) = forgetful_mean_var(κ_traj, forgetful_c, κ̄, κ_var)
        push!(N̄_traj, N̄)
        push!(κ̄_traj, κ̄)
        push!(κ_var_traj, κ_var)

        push!(phonon_pot_traj, measure_potential_energy(model))
        push!(phonon_kin_traj, measure_kinetic_energy(model))
        push!(N²_traj, N²)
        push!(accept_traj, accept)
    end

    return Results(
        μ_traj, N_traj, κ_traj,
        μ̄_traj, N̄_traj, κ̄_traj, κ̄_var_traj,
        phonon_pot_traj, phonon_kin_traj, N²_traj,
        accept_traj
    )
end

function plot_μ_finding(N_target::Float64, μ_target::Float64, κ_target::Float64,
                        results::Results; instant=false)
    pyplot()
    @unpack μ_traj, N_traj, κ_traj, μ̄_traj, N̄_traj, κ̄_traj = results
    nburn = length(μ_traj) - length(μ̄_traj)
    nsteps = length(μ_traj)

    x_min = instant ? 1 : nburn + 1
    plot(xlims=(x_min, nsteps), xscale=:log10, xlabel="Step", size=(800, 600))

    # if instant
    #     plot!(κ_traj, label=L"\kappa", linecolor=:lightgreen)
    # end
    plot!(nburn+1:nsteps, κ̄_traj, label=L"\bar{\kappa}", linecolor=:darkgreen)
    plot!([κ_target], seriestype=:hline, linestyle=:dash, label=L"\kappa^\ast", linecolor=:limegreen)

    if instant
        plot!(N_traj, label=L"N", linecolor=:red)
    end
    plot!(nburn+1:nsteps, N̄_traj, label=L"\bar{N}", linecolor=:darkred)
    plot!([N_target], seriestype=:hline, linestyle=:dash, label=L"N^\ast", linecolor=:red)

    if instant
        plot!(μ_traj, label=L"\mu", linecolor=:lightblue)
    end
    plot!(nburn+1:nsteps, μ̄_traj, label=L"\bar{\mu}", linecolor=:navyblue)
    plot!([μ_target], seriestype=:hline, linestyle=:dash, label=L"\mu^\ast", linecolor=:purple)
    if instant
        plot!([nburn], seriestype=:vline, linestyle=:dash, label="End Init", linecolor=:black) 
    end
end

function test_μ_find(;seed::Int64=12345, β=2., ω₀=1., λ=√2, μinit=-1.0, m_reg=0.4, ninit=10,
                      dt=0.8, nsteps=4, nfaststeps=8, N_target=1.0, nsearch=500000, plot=true,
                      stochastic=false, n_sto_avg=1)
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

    forgetful_c = 0.5
    results = find_μ(model, dyn, N_target, nsearch,
                     ninit=ninit, stochastic=stochastic,
                     n_sto_avg=n_sto_avg, forgetful_c=forgetful_c)

    μ_target = numeric_μ(β, N_target; ω₀=ω₀, λ=λ)
    κ_target = mean_kappa(model)
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

function numeric_μ(β, n; ω₀=1., λ=√2, left=-10, right=10)
    f = μ -> analytic_n(β, μ; ω₀=ω₀, λ=λ) - n
    res_μ = find_zero(f, (left, right), Bisection())
    return res_μ
end
