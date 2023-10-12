# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    TransformedMCMCTransformTuning <: TransformedMCMCTuningAlgorithm

No-op tuning, marks MCMC chains as tuned without performing any other changes
on them. Useful if chains are pre-tuned or tuning is an internal part of the
MCMC sampler implementation.
"""
struct TransformedMCMCTransformTuning <: TransformedMCMCTuningAlgorithm end
export TransformedMCMCTransformTuning



struct TransformedMCMCTransformTuner <: TransformedAbstractMCMCTunerInstance end
export TransformedMCMCTransformTuner

(tuning::TransformedMCMCTransformTuning)(chain::MCMCIterator) = TransformedMCMCTransformTuner()
get_tuner(tuning::TransformedMCMCTransformTuning, chain::MCMCIterator) = TransformedMCMCTransformTuner() 


function TransformedMCMCTransformTuning(tuning::TransformedMCMCTransformTuning, chain::MCMCIterator)
    TransformedMCMCTransformTuner()
end


function tuning_init!(tuner::TransformedMCMCTransformTuner, chain::MCMCIterator, max_nsteps::Integer)
    chain.info = TransformedMCMCIteratorInfo(chain.info, tuned = true)
    nothing
end



function tune_mcmc_transform!!(
    tuner::TransformedMCMCTransformTuner, 
    transform,
    p_accept::Real,
    z_proposed::Vector{<:Float64}, #TODO: use DensitySamples instead
    z_current::Vector{<:Float64},
    stepno::Int,
    context::BATContext
)
    return (tuner, transform)

end

tuning_postinit!(tuner::TransformedMCMCTransformTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::TransformedMCMCTransformTuner, chain::MCMCIterator, max_nsteps::Integer) = nothing

tuning_update!(tuner::TransformedMCMCTransformTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_finalize!(tuner::TransformedMCMCTransformTuner, chain::MCMCIterator) = nothing

tuning_callback(::TransformedMCMCTransformTuning) = nop_func
