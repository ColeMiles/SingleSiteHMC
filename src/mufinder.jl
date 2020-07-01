using Statistics
using Plots

function forgetful_avg(data::Vector{Float64}, c::Float64)
	cutoff = ceil(Int64, c * length(data))
	return mean(data[cutoff:end])
end

function find_μ(model::SSModel, dyn::AbstractDynamics, targetN::Float64, 
				nsteps::Int64; nburn::Int64=10000,
				weight_fn::Function=mean)
	""" Performs the full μ-finding algorithm to attempt to discover the μ
	     which places the given model at the desired N. Progresses only for 
	     the given number of steps.
	         
	    Returns full traces of (μ, ⟨N⟩, ⟨κ⟩).
	"""
	β = model.Δτ * model.N

	N_measured = [measure_mean_occupation(model)]
	κ_measured = Vector{Float64}()
	sizehint!(N_measured, nburn)
	sizehint!(κ_measured, nburn)

	# Perform a burn-in
	for it in 1:nburn
		sample!(model, dyn)

		N = measure_mean_occupation(model)
		N² = measure_mean_sq_occupation(model)
		N̄ = weight_fn(N_measured)

		push!(N_measured, N)
		push!(κ_measured, β * (N² - 2 * N * N̄ + N̄^2))
	end

	μ_traj = [model.μ]
	sizehint!(N_measured, nsteps)
	sizehint!(κ_measured, nsteps)
	sizehint!(μ_traj, nsteps)

	# Main μ-tuning algorithm
	for it in 1:nsteps
		μ̄ = weight_fn(μ_traj)
		N̄ = weight_fn(N_measured)
		κ̄ = weight_fn(κ_measured)

		new_μ = μ̄ + (targetN - N̄) / κ̄
		model.μ = new_μ
		push!(μ_traj, new_μ)

		sample!(model, dyn)

		N = measure_mean_occupation(model)
		N² = measure_mean_sq_occupation(model)

		push!(N_measured, N)
		push!(κ_measured, β * (N² - 2 * N * N̄ + N̄^2))
	end

	return (μ_traj, N_measured, κ_measured)
end

function plot_μ_finding(μ_traj, N_measured, κ_measured, μ_init, N_target)
	pyplot()

	running_mean_N = cumsum(N_measured) ./ (1:length(N_measured))

	plot(μ_traj, label="μ", linecolor=:blue, xlabel="Step", legend=:topright, xscale=:log)
	plot!(running_mean_N, label="Running Mean ⟨N⟩", linecolor=:red)
	# plot(κ_measured, label="⟨κ⟩", linecolor=:green)
	plot!([N_target], seriestype=:hline, linestyle=:dash, label="N∗", linecolor=:red)
end

function test_μ_find(;seed::Int64=12345, ω₀=1., λ=√2, μ=0.0, m_reg=0.4, nburn=10000,
					  dt=0.8, nsteps=4, nfaststeps=8, N_target=1.0, nsearch=500000)
	β = 2
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
    avg_func = data -> forgetful_avg(data, forgetful_c)
    (μ_traj, N_measured, κ_measured) = find_μ(model, dyn, N_target, nsearch,
    										  nburn=nburn, weight_fn=avg_func)

    plot_μ_finding(μ_traj, N_measured, κ_measured, μ, N_target)

    return (μ_traj, N_measured, κ_measured)
end