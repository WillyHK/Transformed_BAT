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

(tuning::MCMCFlowTuning)(chain::MCMCIterator) = MCMCFlowTuner(AdaptiveFlows.Adam(0.005), 2, 20)
get_tuner(tuning::MCMCFlowTuning, chain::MCMCIterator) = MCMCFlowTuner(AdaptiveFlows.Adam(0.005), 2, 20)


function MCMCFlowTuning(tuning::MCMCFlowTuning, chain::MCMCIterator)
    MCMCFlowTuner(AdaptiveFlows.Adam(0.005), 2, 20)
end


function tuning_init!(tuner::MCMCFlowTuner, chain::MCMCIterator, max_nsteps::Integer)
    chain.info = TransformedMCMCIteratorInfo(chain.info, tuned = true) # Add a counter for training epochs of a flow instead? 
    nothing
end

g_state_flow_optimization = (;)

function tune_mcmc_transform!!(
    tuner::MCMCFlowTuner, 
    flow::AdaptiveFlows.AbstractFlow,
    x::AbstractArray,
    target,
    context::BATContext
)   
    # TODO find better way to handle ElasticMatrices
    global g_state_flow_optimization = (x, flow, tuner)

    target_logpdf = x -> BAT.checked_logdensityof(target).(x)
    flow_new = AdaptiveFlows.optimize_flow(nestedview(Matrix(flatview(x))), flow, tuner.optimizer, loss=AdaptiveFlows.negll_flow, nbatches = tuner.n_batches, 
                                                        nepochs = tuner.n_epochs, shuffle_samples = true, logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf))

    eta = tuner.optimizer.eta*0.98
    if (eta < 5f-5)
        eta = 5f-5
    end
    tuner_new = MCMCFlowTuner(AdaptiveFlows.Adam(eta), tuner.n_batches,tuner.n_epochs) # might want to update the training parameters 

    return tuner_new, flow_new
end

tuning_postinit!(tuner::MCMCFlowTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::MCMCFlowTuner, chain::MCMCIterator, max_nsteps::Integer) = nothing

tuning_update!(tuner::MCMCFlowTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_finalize!(tuner::MCMCFlowTuner, chain::MCMCIterator) = nothing

tuning_callback(::MCMCFlowTuning) = nop_func
