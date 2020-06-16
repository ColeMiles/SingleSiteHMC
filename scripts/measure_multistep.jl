using SingleSiteHMC
using Serialization
using Distributed
using SharedArrays
import Base.Iterators.product

dts = [0.4,]
nfasts = [1, 4, 16, 64,]
m_regs = [0.0, 1.0, 10.0, 100.0,]
t_trajs = [6.4,]

# Need to use SharedArrays for parallel write to same memory
accept_probs = SharedArray{Float64}(length(dts), length(nfasts), length(m_regs)+1, length(t_trajs))
phonon_pots = SharedArray{Float64}(length(dts), length(nfasts), length(m_regs)+1, length(t_trajs), 2)
phonon_kins = SharedArray{Float64}(length(dts), length(nfasts), length(m_regs)+1, length(t_trajs), 2)
occupancies = SharedArray{Float64}(length(dts), length(nfasts), length(m_regs)+1, length(t_trajs), 2)

@sync @distributed for ((idt, dt), (inf, nf), (im, m), (it, t_traj)) in collect(product(
    enumerate(dts), enumerate(nfasts), enumerate(m_regs), enumerate(t_trajs)
    ))
    nsteps = round(Int64, t_traj/dt)
    accept_prob, measures = SingleSiteHMC.multistep_simulate(;seed=54321, dt=dt, nsteps=nsteps,
                                                              nfaststeps=nf, m_reg=m, use_fa=true)
    accept_probs[idt, inf, im, it] = accept_prob
    phonon_pots[idt, inf, im, it, 1] = measures["Phonon Potential"][1]
    phonon_pots[idt, inf, im, it, 2] = measures["Phonon Potential"][2]
    phonon_kins[idt, inf, im, it, 1] = measures["Phonon Kinetic"][1]
    phonon_kins[idt, inf, im, it, 2] = measures["Phonon Kinetic"][2]
    occupancies[idt, inf, im, it, 1] = measures["Occupancy"][1]
    occupancies[idt, inf, im, it, 2] = measures["Occupancy"][2]
end

# Zero Fourier Acceleration runs
@sync @distributed for ((idt, dt), (inf, nf), (it, t_traj)) in collect(product(enumerate(dts), enumerate(nfasts), enumerate(t_trajs)))
    nsteps = round(Int64, t_traj/dt)
    accept_prob, measures = SingleSiteHMC.multistep_simulate(;seed=54321, dt=dt, nsteps=nsteps,
                                                              nfaststeps=nf, m_reg=0.0, use_fa=false)
    accept_probs[idt, inf, end, it] = accept_prob
    phonon_pots[idt, inf, end, it, 1] = measures["Phonon Potential"][1]
    phonon_pots[idt, inf, end, it, 2] = measures["Phonon Potential"][2]
    phonon_kins[idt, inf, end, it, 1] = measures["Phonon Kinetic"][1]
    phonon_kins[idt, inf, end, it, 2] = measures["Phonon Kinetic"][2]
    occupancies[idt, inf, end, it, 1] = measures["Occupancy"][1]
    occupancies[idt, inf, end, it, 2] = measures["Occupancy"][2]
end

# SharedArrays don't serialize well, so copy into this to serialize everything
struct Results
    dts :: Vector{Float64}
    nfasts :: Vector{Float64}
    m_regs :: Vector{Float64}
    t_trajs :: Vector{Float64}
    accept_probs :: Array{Float64, 4}
    phonon_pots :: Array{Float64, 5}
    phonon_kins :: Array{Float64, 5}
    occupancies :: Array{Float64, 5}
end

results = Results(
    dts, nfasts, m_regs, t_trajs,
    zeros(length(dts), length(nfasts), length(m_regs)+1, length(t_trajs)),
    zeros(length(dts), length(nfasts), length(m_regs)+1, length(t_trajs), 2),
    zeros(length(dts), length(nfasts), length(m_regs)+1, length(t_trajs), 2),
    zeros(length(dts), length(nfasts), length(m_regs)+1, length(t_trajs), 2)
)

copy!(results.accept_probs, accept_probs)
copy!(results.phonon_pots, phonon_pots)
copy!(results.phonon_kins, phonon_kins)
copy!(results.occupancies, occupancies)

serialize("multistep_results_nofa.ser", results)
