using Random
using FFTW

# Temporarily made this mutable for find_μ process
# TODO: Think of better solution
mutable struct SSModel
    rng::AbstractRNG
    N::Int
    Δτ::Float64
    ω₀::Float64
    λ::Float64
    μ::Float64
    m_reg::Float64
    x::Vector{Float64}
    B::Vector{Float64}

    # Temporary vectors used in stochastic measurements
    R ::Vector{Float64}
    α₁::Vector{Float64}
    α₂::Vector{Float64}

    fft_in ::Vector{ComplexF64}
    fft_out::Vector{ComplexF64}
    pfft::FFTW.cFFTWPlan{Complex{Float64},-1,false,1}
    @static if Sys.iswindows() 
        pifft::AbstractFFTs.ScaledPlan{Complex{Float64},FFTW.cFFTWPlan{Complex{Float64},1,false,1,UnitRange{Int64}},Float64}
    else
        pifft::AbstractFFTs.ScaledPlan{Complex{Float64},FFTW.cFFTWPlan{Complex{Float64},1,false,1},Float64}
    end

    function SSModel(;  seed::Int, N::Int, Δτ::Float64, ω₀::Float64,
                        λ::Float64, μ::Float64, m_reg::Float64=0.,)
        rng = MersenneTwister(seed)
        x = zeros(Float64, N)
        B = zeros(Float64, N)
        fft_in = zeros(ComplexF64, N)
        fft_out = zeros(ComplexF64, N)
        pfft = plan_fft(fft_in)
        pifft = plan_ifft(fft_in)
        R = zeros(Float64, N)
        α₁ = zeros(Float64, N)
        α₂ = zeros(Float64, N)

        return new(rng, N, Δτ, ω₀, λ, μ, m_reg, x, B, R, α₁, α₂, fft_in, fft_out, pfft, pifft)
    end
end

function randomize!(m::SSModel)
    randn!(m.rng, m.x)
    return
end

function update_Bτ!(m::SSModel)
    for τ = 1:m.N
        m.B[τ] = exp(-m.Δτ * (m.λ * m.x[τ] - m.μ))
    end
end

# Applys the M matrix (constructed from m.B's) to a vector
function apply_hubbard_matrix(m::SSModel, v::Vector{Float64}, res::Vector{Float64}; transpose=false)
    if !transpose
        res[1] = 1 * v[1] + m.B[1] * v[m.N]
        for τ = 2:m.N
            res[τ] = -m.B[τ] * v[τ-1] + v[τ]
        end
    else
        for τ = 1:(m.N-1)
            res[τ] = -m.B[τ+1] * v[τ+1] + v[τ]
        end
        res[m.N] = m.B[1] * v[1] + v[m.N]
    end
    return
end

# Applys the M⁻¹ matrix (constructed from m.B's) to a vector
function apply_hubbard_inverse(m::SSModel, v::Vector{Float64}, res::Vector{Float64}; transpose=false)
    denom = 1 + prod(m.B) # needs to be generalized for multi-site

    for τ = 1:m.N
        acc1 = 1.          # chain of B products
        acc2 = acc1 * v[τ] # vector component result

        for j = 1:(m.N-1)
            if !transpose
                τ′ = mod1(τ-j, m.N)
                τ″ = mod1(τ-j+1, m.N)
            else
                τ′ = τ″ = mod1(τ+j, m.N)
            end

            if τ″ == 1
                acc1 *= -1
            end
            acc1 = acc1 * m.B[τ″]
            acc2 += acc1 * v[τ′]
        end
        res[τ] = acc2 / denom
    end
    return
end

function apply_hubbard_squared(m::SSModel, v::Vector{Float64}, res::Vector{Float64})
    """ Applies MᵀM to a vector v, storing the result in res.
    """
    B = zeros(Float64, m.N)
    for τ = 1:m.N
        B[τ] = exp(-m.Δτ * (m.λ * m.x[τ] - m.μ))
        if transpose
            B[τ] = B[τ]' # no-op for single site case
        end
    end

    res[1] = (1 + B[2]^2) * v[1] - B[2] * v[2] + B[1] * v[m.N]
    for τ in 2:m.N-1
        res[τ] = -B[τ] * v[τ-1] + (1 + B[τ+1]^2) * v[τ] - B[τ+1] * v[τ+1]
    end
    res[m.N] = B[1] * v[1] - B[m.N] * v[m.N-1] + (1 + B[1]^2) * v[m.N]
end
