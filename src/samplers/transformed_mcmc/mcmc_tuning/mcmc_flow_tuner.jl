# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using AdaptiveFlows


"""
    MCMCFlowTuning <: TransformedMCMCTuningAlgorithm

Train the Normalizing Flow that is used in conjunction with a MCMC sampler.
"""
struct MCMCFlowTuning <: TransformedMCMCTuningAlgorithm end
export MCMCFlowTuning



struct MCMCFlowTuner <: TransformedAbstractMCMCTunerInstance 
    optimizer
    n_batches::Integer
    n_epochs::Integer
end
export MCMCFlowTuner

(tuning::MCMCFlowTuning)(chain::MCMCIterator) = MCMCFlowTuner(AdaptiveFlows.Adam(1f-3), 10, 1)
get_tuner(tuning::MCMCFlowTuning, chain::MCMCIterator) = MCMCFlowTuner(AdaptiveFlows.Adam(1f-3), 10, 1) 


function MCMCFlowTuning(tuning::MCMCFlowTuning, chain::MCMCIterator)
    MCMCFlowTuner(Adam(1f-3), 10, 1)
end


function tuning_init!(tuner::MCMCFlowTuner, chain::MCMCIterator, max_nsteps::Integer)
    chain.info = TransformedMCMCIteratorInfo(chain.info, tuned = true) # Add a counter for training epochs of a flow instead? 
    nothing
end

function tune_mcmc_transform!!(
    tuner::MCMCFlowTuner, 
    flow::AdaptiveFlows.AbstractFlow,
    x::Vector,
    context::BATContext
)   
    # TODO find better way to handle ElasticMatrices
    # flow_new = AdaptiveFlows.optimize_flow_sequentially(nestedview(Matrix(flatview(x))), flow, tuner.optimizer, nbatches = tuner.n_batches, nepochs = tuner.n_epochs, shuffle_samples = false)

    flow_new = AdaptiveFlows.optimize_flow_sequentially(unshaped.(x), flow, tuner.optimizer, nbatches = tuner.n_batches, nepochs = tuner.n_epochs, shuffle_samples = false)
    tuner_new = tuner # might want to update the training parameters 

    return tuner_new, flow_new
end

tuning_postinit!(tuner::MCMCFlowTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::MCMCFlowTuner, chain::MCMCIterator, max_nsteps::Integer) = nothing

tuning_update!(tuner::MCMCFlowTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_finalize!(tuner::MCMCFlowTuner, chain::MCMCIterator) = nothing

tuning_callback(::MCMCFlowTuning) = nop_func
