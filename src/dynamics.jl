using Parameters
using LinearAlgebra

abstract type AbstractDynamics end
abstract type AbstractFADynamics <: AbstractDynamics end

struct HamiltonianDynamics <: AbstractDynamics
    dt     ::Float64
    steps  ::Int64
    ϕ₁     ::Vector{Float64}
    ϕ₂     ::Vector{Float64}
    ψ₁     ::Vector{Float64} # Will store (MᵀM)⁻¹ ϕ₁
    ψ₂     ::Vector{Float64} # Will store (MᵀM)⁻¹ ϕ₂
    F      ::Vector{Float64}
    x      ::Vector{Float64}
    p      ::Vector{Float64}
    R₁     ::Vector{Float64}
    R₂     ::Vector{Float64}
    temp   ::Vector{Float64} # Array for intermediate computations
    function HamiltonianDynamics(;N::Int, steps::Int64, dt::Float64)
        """ Hamiltonian Monte Carlo. Introduces an auxiliary field ϕ, with associated
             canonical momentum p.
        """
        ϕ₁   = zeros(Float64, N)
        ϕ₂   = zeros(Float64, N)
        ψ₁   = zeros(Float64, N)
        ψ₂   = zeros(Float64, N)
        F    = zeros(Float64, N)
        x    = zeros(Float64, N)
        p    = zeros(Float64, N)
        R₁   = zeros(Float64, N)
        R₂   = zeros(Float64, N)
        temp = zeros(Float64, N)
        return new(dt, steps, use_fa, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, R₁, R₂, temp)
    end
end

struct HamiltonianFADynamics <: AbstractFADynamics
    dt     ::Float64
    steps  ::Int64
    ϕ₁     ::Vector{Float64}
    ϕ₂     ::Vector{Float64}
    ψ₁     ::Vector{Float64} # Will store (MᵀM)⁻¹ ϕ₁
    ψ₂     ::Vector{Float64} # Will store (MᵀM)⁻¹ ϕ₂
    F      ::Vector{Float64}
    x      ::Vector{Float64}
    p      ::Vector{Float64}
    Qp     ::Vector{Float64}
    R₁     ::Vector{Float64}
    R₂     ::Vector{Float64}
    temp   ::Vector{Float64} # Array for intermediate computations
    function HamiltonianFADynamics(;N::Int, steps::Int64, dt::Float64)
        """ Hamiltonian Monte Carlo. Introduces an auxiliary field ϕ, with associated
             canonical momentum p. This version uses Fourier acceleration.
        """
        ϕ₁   = zeros(Float64, N)
        ϕ₂   = zeros(Float64, N)
        ψ₁   = zeros(Float64, N)
        ψ₂   = zeros(Float64, N)
        F    = zeros(Float64, N)
        x    = zeros(Float64, N)
        p    = zeros(Float64, N)
        Qp   = zeros(Float64, N)
        R₁   = zeros(Float64, N)
        R₂   = zeros(Float64, N)
        temp = zeros(Float64, N)
        return new(dt, steps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, Qp, R₁, R₂, temp)
    end
end

struct MultiStepHamiltonianDynamics <: AbstractDynamics
    dt         ::Float64
    steps      ::Int64
    faststeps  ::Float64
    ϕ₁         ::Vector{Float64}
    ϕ₂         ::Vector{Float64}
    ψ₁         ::Vector{Float64} # Will store (MᵀM)⁻¹ ϕ₁
    ψ₂         ::Vector{Float64} # Will store (MᵀM)⁻¹ ϕ₂
    F          ::Vector{Float64}
    x          ::Vector{Float64}
    p          ::Vector{Float64}
    R₁         ::Vector{Float64}
    R₂         ::Vector{Float64}
    temp       ::Vector{Float64} # Array for intermediate computations
    function MultiStepHamiltonianDynamics(;N::Int, steps::Int64, faststeps::Int64, dt::Float64)
        """ Hamiltonian Monte Carlo. Introduces an auxiliary field ϕ, with associated
             canonical momentum p. This version uses multi-timestepping on the boson action.
        """
        ϕ₁   = zeros(Float64, N)
        ϕ₂   = zeros(Float64, N)
        ψ₁   = zeros(Float64, N)
        ψ₂   = zeros(Float64, N)
        F    = zeros(Float64, N)
        x    = zeros(Float64, N)
        p    = zeros(Float64, N)
        R₁   = zeros(Float64, N)
        R₂   = zeros(Float64, N)
        temp = zeros(Float64, N)
        return new(dt, steps, faststeps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, R₁, R₂, temp)
    end
end

struct MultiStepFAHamiltonianDynamics <: AbstractFADynamics
    dt         ::Float64
    steps      ::Int64
    faststeps  ::Float64
    ϕ₁         ::Vector{Float64}
    ϕ₂         ::Vector{Float64}
    ψ₁         ::Vector{Float64} # Will store (MᵀM)⁻¹ ϕ₁
    ψ₂         ::Vector{Float64} # Will store (MᵀM)⁻¹ ϕ₂
    F          ::Vector{Float64}
    x          ::Vector{Float64}
    p          ::Vector{Float64}
    Qp         ::Vector{Float64}
    R₁         ::Vector{Float64}
    R₂         ::Vector{Float64}
    temp       ::Vector{Float64} # Array for intermediate computations
    function MultiStepFAHamiltonianDynamics(;N::Int, steps::Int64, faststeps::Int64, dt::Float64)
        """ Hamiltonian Monte Carlo. Introduces an auxiliary field ϕ, with associated
             canonical momentum p. This version uses multi-timestepping on the boson action,
             and Fourier acceleration.
        """
        ϕ₁   = zeros(Float64, N)
        ϕ₂   = zeros(Float64, N)
        ψ₁   = zeros(Float64, N)
        ψ₂   = zeros(Float64, N)
        F    = zeros(Float64, N)
        x    = zeros(Float64, N)
        p    = zeros(Float64, N)
        Qp   = zeros(Float64, N)
        R₁   = zeros(Float64, N)
        R₂   = zeros(Float64, N)
        temp = zeros(Float64, N)
        return new(dt, steps, faststeps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, Qp, R₁, R₂, temp)
    end
end

# S_bose = Δτ/2 Σ_τ [ω₀² x_τ² + (x_{τ+1} - x_τ)² / Δτ²]
function S_bose(m::SSModel)
    S = 0.0
    for τ = 1:m.N
        x  = m.x[τ]
        x′ = m.x[mod1(τ+1, m.N)]
        S += (m.Δτ/2) * (m.ω₀^2 * x^2 + (x′ - x)^2 / m.Δτ^2)
    end
    return S
end

function S_inter(dyn::AbstractDynamics)
    """ Calculates the portion of the action associated to auxiliary variables introduced in HMC.
        Important: dyn.ψ₁, dyn.ψ₂ must be updated outside of this routine.
    """
    (dyn.ϕ₁' * dyn.ψ₁ + dyn.ϕ₂' * dyn.ψ₂) / 2
end

function S_kinetic(dyn::AbstractFADynamics)
    (dyn.p' * dyn.Qp) / 2
end

function S_kinetic(dyn::AbstractDynamics)
    (dyn.p' * dyn.p) / 2
end

function S_total(m::SSModel, dyn::AbstractDynamics)
    S_bose(m) + S_inter(dyn) + S_kinetic(dyn)
end

function fourier_accelerate!(m::SSModel, x::Vector{Float64}, pow::Float64)
    @. m.fft_in = Complex(x)
    mul!(m.fft_out, m.pfft, m.fft_in)
    for ω = 1:m.N
        numer = m.m_reg^2 + m.ω₀^2
        denom = m.Δτ * (m.m_reg^2 + m.ω₀^2 + (2 - 2cos(2π * (ω-1)/m.N)) / m.Δτ^2)
        m.fft_out[ω] *= (numer / denom)^pow
    end
    mul!(m.fft_in, m.pifft, m.fft_out)
    @. x = real(m.fft_in)
    return
end

function calc_F!(model::SSModel, dyn::AbstractDynamics)
    """ Calculates F_HMC, also updating Bτ's in model, and ψ's in dyn
    """
    @unpack ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, temp = dyn

    # Update B's in model
    update_Bτ!(model)

    ### Calculate part of force from auxiliary fields
    # ψ₁ = (MᵀM)⁻¹ ϕ₁
    apply_hubbard_inverse(model, ϕ₁, temp; transpose=true)
    apply_hubbard_inverse(model, temp, ψ₁)
    # ψ₂ = (MᵀM)⁻¹ ϕ₂
    apply_hubbard_inverse(model, ϕ₂, temp; transpose=true)
    apply_hubbard_inverse(model, temp, ψ₂)

    # Zero out forces
    fill!(F, 0.0)

    ### Compute auxiliary field force from ψ's
    for τ = 1:model.N
        if τ == 1
            F[τ] += -model.Δτ * model.λ * (ψ₁[model.N] * model.B[1] + ψ₁[1]) * model.B[1] * ψ₁[model.N]
            F[τ] += -model.Δτ * model.λ * (ψ₂[model.N] * model.B[1] + ψ₂[1]) * model.B[1] * ψ₂[model.N]
        else
            F[τ] += -model.Δτ * model.λ * (ψ₁[τ-1] * model.B[τ] - ψ₁[τ]) * model.B[τ] * ψ₁[τ-1]
            F[τ] += -model.Δτ * model.λ * (ψ₂[τ-1] * model.B[τ] - ψ₂[τ]) * model.B[τ] * ψ₂[τ-1]
        end
    end

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
end

function calc_F_aux_fields!(model::SSModel, dyn::AbstractDynamics)
    """ Calculates the component of F_HMC from the auxiliary fields
    """
    @unpack ϕ₁, ϕ₂, ψ₁, ψ₂, F, temp = dyn

    fill!(dyn.F, 0)

    # Update B's in model
    update_Bτ!(model)

    ### Calculate part of force from auxiliary fields
    # ψ₁ = (MᵀM)⁻¹ ϕ₁
    apply_hubbard_inverse(model, ϕ₁, temp; transpose=true)
    apply_hubbard_inverse(model, temp, ψ₁)
    # ψ₂ = (MᵀM)⁻¹ ϕ₂
    apply_hubbard_inverse(model, ϕ₂, temp; transpose=true)
    apply_hubbard_inverse(model, temp, ψ₂)

    ### Compute auxiliary field force from ψ's
    for τ = 1:model.N
        if τ == 1
            dyn.F[τ] += -model.Δτ * model.λ * (ψ₁[model.N] * model.B[1] + ψ₁[1]) * model.B[1] * ψ₁[model.N]
            dyn.F[τ] += -model.Δτ * model.λ * (ψ₂[model.N] * model.B[1] + ψ₂[1]) * model.B[1] * ψ₂[model.N]
        else
            dyn.F[τ] += -model.Δτ * model.λ * (ψ₁[τ-1] * model.B[τ] - ψ₁[τ]) * model.B[τ] * ψ₁[τ-1]
            dyn.F[τ] += -model.Δτ * model.λ * (ψ₂[τ-1] * model.B[τ] - ψ₂[τ]) * model.B[τ] * ψ₂[τ-1]
        end
    end
end

function calc_F_phonon!(model::SSModel, dyn::AbstractDynamics)
    """ Calculuates the component of F_HMC from the phonons
    """
    fill!(dyn.F, 0)
    for τ = 1:model.N
        x  = model.x[τ]
        x′ = model.x[mod1(τ+1, model.N)]
        x″ = model.x[mod1(τ-1, model.N)]
        dyn.F[τ] += -model.Δτ * (
            model.ω₀^2 * x                # phonon potential
            - (x′ - 2x + x″) / model.Δτ^2  # phonon kinetic 
        )
    end
end

# TODO: Any better way to abstract this rather than having four sample functions?

function sample!(model::SSModel, dyn::MultiStepFAHamiltonianDynamics)
    """ Produces a single sample of a field configuration using HMC, by evolving a Hamiltonian
         by multiple timesteps with a symplectic integrator. This version includes FA, and
         multi-timestepping.
    """
    @unpack dt, steps, faststeps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, Qp, R₁, R₂ = dyn

    # Copy phonon field configuration to restore if rejected
    copy!(x, model.x)

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

    # Calculate initial energy
    H = S_total(model, dyn)

    # Step p by a half-step in the interaction Hamiltonian to begin the cycle
    @. p += F * dt / 2

    fastdt = dt / faststeps

    # Evolve x, p under Hamiltonian dynamics using the leapfrog integrator
    for step in 1:steps

        # Perform a sub-leapfrog integration on the phonon action for faststeps # of steps
        calc_F_phonon!(model, dyn)
        # Half-step in Bose Hamiltonian to start cycle
        @. p += F * fastdt / 2

        for faststep in 1:faststeps
            copy!(Qp, p)
            fourier_accelerate!(model, Qp, 1.0)
            @. model.x += Qp * fastdt

            calc_F_phonon!(model, dyn)

            @. p += F * fastdt
        end

        # Back-step by a half-step to re-align times for Bose Hamiltonian
        @. p -= F * fastdt / 2

        # Step interaction hamiltonian
        calc_F_aux_fields!(model, dyn)
        @. p += F * dt
    end

    # Back-step p by a half-step to re-align times for interaction Hamiltonian
    @. p -= F * dt / 2

    # Re-update Qp for Hamiltonian evaluation
    copy!(Qp, p)
    fourier_accelerate!(model, Qp, 1.0)

    # Try to accept new configuration with Metropolis probability
    H′ = S_total(model, dyn) 

    ΔH = H′ - H
    prob = min(1, exp(-ΔH))

    if rand(model.rng) < prob
        return true           # Acceptance
    else
        # Copy original field configuration back
        copy!(model.x, dyn.x)
        return false          # Rejection
    end
end

function sample!(model::SSModel, dyn::AbstractFADynamics)
    """ Produces a single sample of a field configuration using HMC, by evolving a Hamiltonian
         by multiple timesteps with a symplectic integrator. This version includes FA.
    """
    @unpack dt, steps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, Qp, R₁, R₂ = dyn

    # Copy phonon field configuration to restore if rejected
    copy!(x, model.x)

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
    calc_F!(model, dyn)
    copy!(Qp, p)
    fourier_accelerate!(model, Qp, 1.0)

    # Calculate initial energy
    H = S_total(model, dyn)

    # Step p by a half-step to begin the cycle
    @. p += F * dt / 2

    # Evolve x, p under Hamiltonian dynamics using the leapfrog integrator
    for step in 1:steps
        copy!(Qp, p)
        fourier_accelerate!(model, Qp, 1.0)
        @. model.x += Qp * dt
        
        calc_F!(model, dyn)
        @. p += F * dt
    end

    # Back-step p by a half-step to re-align times
    @. p -= F * dt / 2

    # Re-update Qp for Hamiltonian evaluation
    copy!(Qp, p)
    fourier_accelerate!(model, Qp, 1.0)

    # Try to accept new configuration with Metropolis probability
    H′ = S_total(model, dyn) 

    ΔH = H′ - H
    prob = min(1, exp(-ΔH))

    if rand(model.rng) < prob
        return true           # Acceptance
    else
        # Copy original field configuration back
        copy!(model.x, dyn.x)
        return false          # Rejection
    end
end

function sample!(model::SSModel, dyn::AbstractDynamics)
    """ Produces a single sample of a field configuration using HMC, by evolving a Hamiltonian
         by multiple timesteps with a symplectic integrator.
    """
    @unpack dt, steps, ϕ₁, ϕ₂, ψ₁, ψ₂, F, x, p, R₁, R₂ = dyn

    # Copy phonon field configuration to restore if rejected
    copy!(x, model.x)

    update_Bτ!(model)

    # Sample an auxiliary field configurations ϕ₁, ϕ₂
    randn!(model.rng, R₁)
    apply_hubbard_matrix(model, R₁, ϕ₁; transpose=true)
    randn!(model.rng, R₂)
    apply_hubbard_matrix(model, R₂, ϕ₂; transpose=true)

    # Sample an initial momentum configuration
    randn!(model.rng, p)

    # Calculate forces, and update B's, ψ's
    calc_F!(model, dyn)

    # Calculate initial energy
    H = S_total(model, dyn)

    # Step p by a half-step to begin the cycle
    @. p += F * dt / 2

    # Evolve x, p under Hamiltonian dynamics using the leapfrog integrator
    for step in 1:steps
        @. model.x += p * dt
        
        calc_F!(model, dyn)
        @. p += F * dt
    end

    # Back-step p by a half-step to re-align times
    @. p -= F * dt / 2

    # Try to accept new configuration with Metropolis probability
    H′ = S_total(model, dyn) 

    ΔH = H′ - H
    prob = min(1, exp(-ΔH))

    if rand(model.rng) < prob
        return true           # Acceptance
    else
        # Copy original field configuration back
        copy!(model.x, dyn.x)
        return false          # Rejection
    end
end