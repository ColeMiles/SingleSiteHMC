using Statistics
using Parameters

#=  Exact Formula =#

function mean_potential(m::SSModel)
    """ Exact formula for ⟨PEₚₕ⟩
    """
    β = m.N * m.Δτ
    μ₀ = - (m.λ / m.ω₀)^2
    exp1 = exp(β * (m.μ - μ₀/2))
    exp2 = exp(2β * (m.μ - μ₀))
    return (
        (m.ω₀/4) / tanh(β*m.ω₀/2) # phonon
        - (μ₀/2) * ((2exp1 + 4exp2) / (1 + 2exp1 + exp2)) # electron
    )
end

function mean_kinetic(m::SSModel)
    """ Exact formula for ⟨KEₚₕ⟩
    """
    β = m.N * m.Δτ
    return (m.ω₀ / 2) * (1/2 + 1.0 / (exp(β * m.ω₀) - 1))
end

function mean_occupancy(m::SSModel)
    """ Exact formula for ⟨n⟩
    """
    β = m.N * m.Δτ
    μ₀ = -m.λ^2 / m.ω₀^2
    exp1 = exp(β * (m.μ - μ₀/2))
    exp2 = exp(2 * β * (m.μ - μ₀))
    numer = 2 * exp1 + 2 * exp2
    denom = 1 + 2 * exp1 + exp2
    return numer / denom
end

function mean_occupancy_sq(m::SSModel)
    """ Exact formula for ⟨n²⟩
    """
    β = m.N * m.Δτ
    exp1 = exp(β * (m.λ^2 / (2 * m.ω₀^2) + m.μ))
    exp2 = exp(2 * β * (m.λ^2 / m.ω₀^2 + m.μ))
    numer = 2 * exp1 + 4 * exp2
    denom = 1 + 2 * exp1 + exp2
    return numer / denom
end

function mean_double_occupancy(m::SSModel)
    """ Exact formula for ⟨n↑ n↓⟩
    """
    β = m.N * m.Δτ
    exp1 = exp(β * (m.λ^2 / (2 * m.ω₀^2) + m.μ))
    exp2 = exp(2 * β * (m.λ^2 / m.ω₀^2 + m.μ))
    denom = 1 + 2 * exp1 + exp2
    return exp2 / denom
end

function mean_kappa(m::SSModel)
    β = m.N * m.Δτ
    exp1 = exp(β * (m.λ^2 / (2 * m.ω₀^2) + m.μ))
    exp2 = exp(2 * β * (m.λ^2 / m.ω₀^2 + m.μ))
    numer = 2β * ((1 + 2exp1 + exp2) * (exp1 + 2exp2) - (2exp1 + 2exp2) * (exp1 + exp2))
    denom = (1 + 2exp1 + exp2)^2
    return numer / denom
end

function exact_greens(m::SSModel, τ::Float64)
    @unpack μ, λ, ω₀, Δτ, N = m
    β = N * Δτ

    numer_fact1 = exp(τ * μ + λ^2 * τ^2 / (2 * β * ω₀^2))
    numer_fact2 = exp((β + τ) * μ + (β/2) * (λ/ω₀ * (1 + τ/β))^2)
    denom = 1 + 2exp(β * (λ^2/(2*ω₀^2) + μ)) + exp(2 * β * (λ^2/ω₀^2 + μ))
    return (numer_fact1 + numer_fact2) / denom
end

#= Measurement Routines =#

function measure_potential_energy(m::SSModel)
    """ Measures ⟨Vₚₕ⟩
    """
    ret = 0.0
    for τ = 1:m.N
        ret += m.x[τ]^2
    end
    return m.ω₀^2 * ret / (2 * m.N) 
end

function measure_kinetic_energy(m::SSModel)
    """ Measures ⟨Kₚₕ⟩
    """
    acc_diff = 0.0
    for τ = 1:m.N
        acc_diff += (m.x[τ%m.N+1] - m.x[τ])^2
    end
    return 1 / (2 * m.Δτ) - 1 / (2 * m.Δτ^2 * m.N) * acc_diff
end

function measure_mean_occupation(m::SSModel)
    """ Measures ⟨n⟩. Assumes Bτ's are updated.
    """
    return 2 * (1 - 1 / (1 + prod(m.B)))
end

function measure_mean_sq_occupation(m::SSModel)
    """ Measures ⟨n²⟩. Assumes Bτ's are updated.
    Can compute:
        ⟨n²⟩ = ⟨(n↑ + n↓)²⟩ = ⟨n↑²⟩ + ⟨n↓²⟩ + 2 ⟨n↓n↑⟩
    Since there can only be zero or one spin up or down particle,
        ⟨n↑²⟩ = ⟨n↑⟩, ⟨n↓²⟩ = ⟨n↓⟩
    """
    return measure_mean_occupation(m) + 2 * measure_double_occupation(m)
end

function measure_double_occupation(m::SSModel)
    """ Measures ⟨n↑ n↓⟩. Assumes Bτ's are updated
    Since spins are independent, this is simply:
        ⟨n↑ n↓⟩ = ⟨n↑⟩⟨n↓⟩
    """
    avg_n = 1 - 1 / (1 + prod(m.B))
    return avg_n^2
end

# Hack to emulate a "static" variable G only visible within this functions's scope
# Seems this is the best existing option?
# https://discourse.julialang.org/t/const-static-function-parameters/1803/32
let
    G = Vector{Float64}()
    global function measure_electron_greens(m::SSModel,)
        """ Measures Gₑ(τ, τ+Δ) ≡ M⁻¹_{τ, τ+Δ} for all values of Δ.
            Returns [G(0), G(Δτ), G(2Δτ), ..., G((L-1)Δτ].

            The way this functions computes this relies on the commutation of B's
            in the single-site limit.
        """
        @unpack N, B = m 

        W = 1 + prod(B)

        resize!(G, N)
        G .= 0

        # G(0) = W⁻¹, so put N here to cancel out with final normalization
        G[1] = N

        for τ in 1:N
            acc = 1.0
            for Δ in 1:N-1
                τ′ = mod1(τ+Δ, N)
                acc *= B[τ′]
                G[Δ+1] += acc
            end
        end

        ## Uncomment below to only use lower triangle for measurements
        # for τ in 1:N
        #     acc = 1.0
        #     for Δ in 1:N-τ
        #         τ′ = mod1(τ+Δ, N)
        #         acc *= B[τ′]
        #         G[Δ+1] += acc
        #     end
        # end

        # for Δ in 0:N-1
        #     G[Δ+1] /= (W * (N - Δ))
        # end

        G ./= (W * N)
        return G
    end
end

let
    G = Vector{Float64}()
    global function measure_electron_greens_heavy(m::SSModel,)
        """ Measures Gₑ(τ, τ+Δ) ≡ M⁻¹_{τ, τ+Δ} for all values of Δ.
            Returns [G(0), G(Δτ), G(2Δτ), ..., G((L-1)Δτ].

            This version must be correct, but is slow.
        """
        @unpack N, B = m 

        # Build the entire M⁻¹ matrix!
        Minv = zeros(N, N)

        temp = zeros(N)
        for i in 1:N
            temp2 = zeros(N)
            temp2[i] = 1.0
            apply_hubbard_inverse(m, temp2, temp)
            Minv[1:end, i] .= temp
        end

        resize!(G, N)
        G .= 0

        # Now, make the measurements

        for τ in 1:N
            for Δ in 0:N-1
                τ′ = mod1(τ + Δ, N)
                G[Δ+1] += Minv[τ′, τ] * ifelse(τ′ >= τ, 1, -1)
            end
        end

        G ./= N
        return G, Minv
    end
end

let
    G = Vector{Float64}()
    global function measure_phonon_greens(m::SSModel)
        """ Measures Gₚ(τ, τ+Δ) ≡ ⟨X_{τ+Δ} X_{τ}⟩ for all values of Δ.
            Returns [G(0), G(Δτ), G(2Δτ), ..., G((L-1)Δτ].
        """
        @unpack N, B = m 

        resize!(G, m.N)
        G .= 0

        for τ in 1:N
            for Δ in 0:N-1
                τ′ = mod1(τ+Δ, N)
                G[Δ+1] += m.x[τ] * m.x[τ′]
            end
        end

        G ./= N
        return G
    end
end

#= Stochastic Measurement Routines =#

function sto_measure_occupations(m::SSModel; n_avg=1)
    """ Measures both ⟨N⟩, ⟨N²⟩ using a stochastic estimator
    """
    @unpack x, R, α₁, α₂ = m
    L = length(x)

    N_measure_acc = 0.0
    N²_measure_acc = 0.0

    for _ in 1:n_avg
        randn!(m.sto_rng, R)
        apply_hubbard_inverse(m, R, α₁)
        apply_hubbard_inverse(m, R, α₂; transpose=true)

        N_measure = (2. / L) * (R' * (R - α₁))
        N²_measure = N_measure ^ 2 + (2. / L^2) * (-sum((2R - α₁ - α₂).^2) + α₂' * (R - α₁))

        N_measure_acc += N_measure
        N²_measure_acc += N²_measure
    end

    return (N_measure_acc / n_avg, N²_measure_acc / n_avg)
end

#= Binned Statistics Routines =#

"""
Calculates the average and binned standard deviation of a set of data.
The number of bins used is equal to the length of the preallocated `bins` vector
passed to the function.
"""
function binned_statistics(data::AbstractVector{T}, nbins::Int=10)::Tuple{T,T} where {T<:Number}
    
    bins = zeros(T, nbins)
    avg, stdev = binned_statistics(data, bins)
    return avg, stdev
end

function binned_statistics(data::AbstractVector{T}, bins::Vector{T})::Tuple{T,T} where {T<:Number}
    
    N = length(data)
    n = length(bins)
    @assert length(data)%length(bins) == 0
    binsize = div(N, n)
    bins .= 0
    for bin in 1:n
        for i in 1:binsize
            bins[bin] += data[i + (bin-1)*binsize]
        end
        bins[bin] /= binsize
    end
    avg = mean(bins)
    return avg, std(bins, corrected=true, mean=avg) / sqrt(n)
end
