using Plots
using Parameters

function plot_energy_conservation_hmc(model, dyn; use_fa=true)
    randomize!(model)
    update_Bτ!(model)

    if use_fa
	    dyn = HamiltonianFADynamics(; N=N, steps=nsteps, dt=dt)
	    @unpack dt, steps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, Qp, R₁, R₂, temp = dyn
	else
	    dyn = HamiltonianDynamics(; N=N, steps=nsteps, dt=dt)
	    @unpack dt, steps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, R₁, R₂, temp = dyn
	end


    randn!(model.rng, R₁)
    randn!(model.rng, R₂)
    apply_hubbard_matrix(model, R₁, ϕ₁; transpose=true)
    apply_hubbard_matrix(model, R₂, ϕ₂; transpose=true)

    # Perform a burn-in first
    burnin_samples = 10000
    for step in 1:burnin_samples
        sample!(model, dyn)
    end

    randn!(model.rng, dyn.p)
    if use_fa
        fourier_accelerate!(model, p, -0.5)
    end

    calc_F!(model, dyn)
    if use_fa
        copy!(Qp, p)
        fourier_accelerate!(model, Qp, 1.0)
    end

    Es = Vector{Float64}()
    Eboses = Vector{Float64}()
    EHMC_kins = Vector{Float64}()
    EHMC_ints = Vector{Float64}()

    push!(Es, S_total(model, dyn))
    push!(Eboses, S_bose(model))
    push!(EHMC_kins, S_kinetic(dyn))
    push!(EHMC_ints, S_inter(dyn))

    for step in 1:integrator_nsteps
        @. p += F * dt / 2

        if dyn.use_fa
            copy!(Qp, p)
            fourier_accelerate!(model, Qp, 1.0)
            @. model.x += Qp * dt
        else
            @. model.x += p * dt
        end

        calc_F!(model, dyn)
        @. p += F * dt / 2

        if dyn.use_fa
            copy!(Qp, p)
            fourier_accelerate!(model, Qp, 1.0)
        end

	    push!(Es, S_total(model, dyn))
	    push!(Eboses, S_bose(model))
	    push!(EHMC_kins, S_kinetic(dyn))
	    push!(EHMC_ints, S_inter(dyn))
	end

    plot(Es, xlabel="Step", ylabel="E", label="Total")
    plot!(Eboses, xlabel="Step", ylabel="E", label="Bose")
    plot!(EHMC_kins, xlabel="Step", ylabel="E", label="HMC Kin.")
    plot!(EHMC_ints, xlabel="Step", ylabel="E", label="HMC Int.")
end

function animate_phonon_configurations(model, dyn; use_fa=false)
    randomize!(model)
    update_Bτ!(model)

    @unpack dt, steps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, Qp, R₁, R₂, temp = dyn

    # Perform a burn-in first
    burnin_samples = 10000
    for step in 1:burnin_samples
        sample!(model, dyn)
    end

    anim_dt = 0.001
    anim_nsteps = 4000

    # Observe a Hamiltonian trajectory
    randn!(model.rng, R₁)
    randn!(model.rng, R₂)
    apply_hubbard_matrix(model, R₁, ϕ₁; transpose=true)
    apply_hubbard_matrix(model, R₂, ϕ₂; transpose=true)

    randn!(model.rng, dyn.p)
    if use_fa
        fourier_accelerate!(model, p, -0.5)
    end

    xs = Vector{Vector{Float64}}()

    calc_F!(model, dyn)
    if use_fa
        copy!(Qp, p)
        fourier_accelerate!(model, Qp, 1.0)
    end

    for step in 1:anim_nsteps
        @. p += F * dt / 2

        if dyn.use_fa
            copy!(Qp, p)
            fourier_accelerate!(model, Qp, 1.0)
            @. model.x += Qp * dt
        else
            @. model.x += p * dt
        end

        calc_F!(model, dyn)
        @. p += F * dt / 2

        if dyn.use_fa
            copy!(Qp, p)
            fourier_accelerate!(model, Qp, 1.0)
        end

        push!(xs, deepcopy(model.x))
    end

    pyplot()
    τrange = range(0.0, step=0.05, stop=2.0)

    max_x = 3.22923
    min_x = -3.22923

    anim = @animate for i in 1:6:length(xs)
        plot(τrange, xs[i], xlabel="\$\\tau\$", ylabel="\$x_\\tau\$", label="", title="Step = $i", ylims=(min_x, max_x))
    end
    mp4(anim, "WithFA.mp4", fps=30)
    nothing
end