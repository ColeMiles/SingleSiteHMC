using Statistics
using Plots
using Parameters
using Roots

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

function forgetful_avg(data::Vector{Float64}, c::Float64)
    cutoff = ceil(Int64, c * length(data))
    return mean(data[cutoff:end])
end

# Constant time update version. Requires previous avg value, assumes only one update has been done.
function forgetful_avg(data::Vector{Float64}, c::Float64, prev_val::Float64)
    cutoff = ceil(Int64, c * length(data))
    old_cutoff = ceil(Int64, c * (length(data) - 1))

    new_val = (length(data) - old_cutoff) * prev_val
    if old_cutoff != cutoff
        new_val -= data[old_cutoff]
    end
    new_val += data[end]

    return new_val / (length(data) - cutoff + 1)
end

# Dummy mean function with extra argument to match what's expected by find_μ
function mean(data::Vector{Float64}, prev_val::Float64)
    return mean(data)
end

function find_μ(model::SSModel, dyn::AbstractDynamics, targetN::Float64, 
                nsteps::Int64; ninit::Int64=10,
                weight_fn::Function=mean)
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

    # Main μ-tuning algorithm
    for it in 1:nsteps
        μ̄ = weight_fn(μ_traj, μ̄)
        N̄ = weight_fn(N_traj, N̄)
        κ̄ = weight_fn(κ_traj, κ̄)

        push!(μ̄_traj, μ̄)
        push!(N̄_traj, N̄)
        push!(κ̄_traj, κ̄)

        new_μ = μ̄ + (targetN - N̄) / κ̄
        model.μ = new_μ
        push!(μ_traj, new_μ)

        accept = sample!(model, dyn)
        push!(accepts, accept)

        N = measure_mean_occupation(model)
        N² = measure_mean_sq_occupation(model)

        push!(N_traj, N)
        push!(κ_traj, β * (N² - 2 * N * N̄ + N̄^2))
        push!(phonon_pot_traj, measure_potential_energy(model))
        push!(phonon_kin_traj, measure_kinetic_energy(model))
        push!(N²_traj, N²)
    end

    return Results(
        μ_traj, N_traj, κ_traj,
        μ̄_traj, N̄_traj, κ̄_traj,
        phonon_pot_traj, phonon_kin_traj, N²_traj,
        accepts
    )
end

function const_μ_traj(model::SSModel, dyn::AbstractDynamics, nsteps::Int64; weight_fn::Function=mean)
    """ Just like multistep_simulate, but records more details about observable trajectorie trajectories.
    """
    β = model.N * model.Δτ
    μ = model.μ
    N̄ = measure_mean_occupation(model)
    κ̄ = 0.0

    μ_traj = fill(model.μ, nsteps)
    N_traj = [N̄]
    κ_traj = [κ̄]
    μ̄_traj = fill(model.μ, nsteps)
    N̄_traj = [N̄]
    κ̄_traj = [κ̄]
    phonon_pot_traj = [measure_potential_energy(model)]
    phonon_kin_traj = [measure_kinetic_energy(model)]
    N²_traj = [measure_mean_sq_occupation(model)]
    accept_traj = [true]
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

        N = measure_mean_occupation(model)
        N² = measure_mean_sq_occupation(model)
        κ = β * (N² - 2 * N * N̄ + N̄^2)
        push!(N_traj, N)
        push!(κ_traj, β * (N² - 2 * N * N̄ + N̄^2))

        N̄ = weight_fn(N_traj, N̄)
        κ̄ = weight_fn(κ_traj, κ̄)
        push!(N̄_traj, N̄)
        push!(κ̄_traj, κ̄)

        push!(phonon_pot_traj, measure_potential_energy(model))
        push!(phonon_kin_traj, measure_kinetic_energy(model))
        push!(N²_traj, N²)
        push!(accept_traj, accept)
    end

    return Results(
        μ_traj, N_traj, κ_traj,
        μ̄_traj, N̄_traj, κ̄_traj,
        phonon_pot_traj, phonon_kin_traj, N²_traj,
        accept_traj
    )
end

function plot_μ_finding(N_target::Float64, μ_target::Float64, results::Results)
    pyplot()
    @unpack μ_traj, N_traj, κ_traj, μ̄_traj, N̄_traj, κ̄_traj = results
    nburn = length(μ_traj) - length(μ̄_traj)
    nsteps = length(μ_traj)

    plot(μ_traj, label="μ", linecolor=:lightblue, xlabel="Step",
         legend=:bottomright, xscale=:log10)
    plot!(N_traj, label="N", linecolor=:red)
    plot!(κ_traj, label="κ", linecolor=:lightgreen)
    plot!(nburn:nsteps, μ̄_traj, label="μ̄", linecolor=:navyblue)
    plot!(nburn:nsteps, N̄_traj, label="N̄", linecolor=:darkred)
    plot!(nburn:nsteps, κ̄_traj, label="κ̄", linecolor=:darkgreen)
    plot!([N_target], seriestype=:hline, linestyle=:dash, label="N∗", linecolor=:red)
    plot!([μ_target], seriestype=:hline, linestyle=:dash, label="μ∗", linecolor=:purple)
    plot!([nburn], seriestype=:vline, linestyle=:dash, label="End Burn", linecolor=:black) 
end

function test_μ_find(;seed::Int64=12345, β=2., ω₀=1., λ=√2, μ=0.0, m_reg=0.4, ninit=10,
                      dt=0.8, nsteps=4, nfaststeps=8, N_target=1.0, nsearch=500000)
    Δτ_target = 0.1
    N = round(Int, β/Δτ_target)
    Δτ = β / N
    if Δτ ≠ Δτ_target
        println("Modified Δτ from $Δτ_target to $Δτ")
    end

    model = SSModel(seed=seed, N=N, Δτ=Δτ, ω₀=ω₀, λ=λ, μ=μ, m_reg=m_reg)
    randomize!(model)

    dyn = MultiStepFAHamiltonianDynamics(
        ;N=N, steps=nsteps, faststeps=nfaststeps,
        dt=dt
    )

    forgetful_c = 0.5
    avg_func = (data, prev_val) -> forgetful_avg(data, forgetful_c, prev_val)
    results = find_μ(model, dyn, N_target, nsearch,
                     ninit=ninit, weight_fn=avg_func)

    μ_target = numeric_μ(β, N_target; ω₀=ω₀, λ=λ)
    plot_μ_finding(N_target, μ_target, results)

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
