abstract type TransformedMCMCProposal end
"""
    BAT.TransformedMHProposal

*BAT-internal, not part of stable public API.*
"""
struct TransformedMHProposal{
    D<:Union{Distribution, AbstractMeasure}
}<: TransformedMCMCProposal
    proposal_dist::D
end      


# TODO AC: find a better solution for this. Problem is that in the with_kw constructor below, we need to dispatch on this type.
struct TransformedMCMCDispatch end

@with_kw struct TransformedMCMCSampling{
    TR<:AbstractTransformTarget,
    IN<:TransformedMCMCInitAlgorithm,
    BI<:TransformedMCMCBurninAlgorithm,
    CT<:ConvergenceTest,
    CB<:Function
} <: AbstractSamplingAlgorithm
    pre_transform::TR = bat_default(TransformedMCMCDispatch, Val(:pre_transform))
    tuning_alg::TransformedMCMCTuningAlgorithm = TransformedRAMTuner() # TODO: use bat_defaults
    adaptive_transform::AdaptiveTransformSpec = default_adaptive_transform(tuning_alg)
    proposal::TransformedMCMCProposal = TransformedMHProposal(Normal()) #TODO: use bat_defaults
    tempering = TransformedNoTransformedMCMCTempering() # TODO: use bat_defaults
    nchains::Int = 4
    nsteps::Int = 3*10#^2#^4 # Why this eskalation?
    #TODO: max_time ?
    init::IN =  bat_default(TransformedMCMCDispatch, Val(:init), pre_transform, nchains, nsteps) #TransformedMCMCChainPoolInit()#TODO AC: use bat_defaults bat_default(MCMCSampling, Val(:init), MetropolisHastings(), pre_transform, nchains, nsteps) #TODO
    burnin::BI = bat_default(TransformedMCMCDispatch, Val(:burnin), pre_transform, nchains, nsteps)
    convergence::CT = TransformedBrooksGelmanConvergence()
    strict::Bool = true
    store_burnin::Bool = false
    nonzero_weights::Bool = true
    callback::CB = nop_func
end

bat_default(::Type{TransformedMCMCDispatch}, ::Val{:pre_transform}) = PriorToGaussian()

bat_default(::Type{TransformedMCMCDispatch}, ::Val{:nsteps}, trafo::AbstractTransformTarget, nchains::Integer) = 10^5

bat_default(::Type{TransformedMCMCDispatch}, ::Val{:init}, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    TransformedMCMCChainPoolInit(nsteps_init = max(div(nsteps, 100), 250))

bat_default(::Type{TransformedMCMCDispatch}, ::Val{:burnin}, trafo::AbstractTransformTarget, nchains::Integer, nsteps::Integer) =
    TransformedMCMCMultiCycleBurnin(nsteps_per_cycle = max(div(nsteps, 10), 2500))



function bat_sample_impl(
    target::AnyMeasureOrDensity,
    algorithm::TransformedMCMCSampling,
    context::BATContext
)
    if(algorithm.adaptive_transform isa CustomTransform)
        println("Using ensembles to sample with flows!")
        return bat_sample_impl_ensemble(target,algorithm,context)
    end

    density, trafo = transform_and_unshape(algorithm.pre_transform, target, context)

    init = mcmc_init!(
        algorithm,
        density,
        algorithm.nchains,
        apply_trafo_to_init(trafo, algorithm.init),
        algorithm.tuning_alg,
        algorithm.nonzero_weights,
        algorithm.store_burnin ? algorithm.callback : nop_func,
        context
    )

    @unpack chains, tuners, temperers = init

    burnin_outputs_coll = if algorithm.store_burnin
        DensitySampleVector(first(chains))
    else
        nothing
    end

    mcmc_burnin!(
       burnin_outputs_coll,
       chains,
       tuners,
       temperers,
       algorithm.burnin,
       algorithm.convergence,
       algorithm.strict,
       algorithm.nonzero_weights,
       algorithm.store_burnin ? algorithm.callback : nop_func
    )

    # sampling
    run_sampling  = _run_sample_impl(
        density,
        algorithm,
        chains,
    )
    samples_trafo, generator = run_sampling.result_trafo, run_sampling.generator

    # prepend burnin samples to output
    if algorithm.store_burnin
        burnin_samples_trafo = varshape(density).(burnin_outputs_coll)
        append!(burnin_samples_trafo, samples_trafo)
        samples_trafo = burnin_samples_trafo
    end

    samples_notrafo = inverse(trafo).(samples_trafo)
    

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end

g_state_post_algorithm = (;)

function bat_sample_impl_ensemble(
    target::AnyMeasureOrDensity,
    algorithm::TransformedMCMCSampling,
    context::BATContext
)  

    density, pre_trafo = transform_and_unshape(algorithm.pre_transform, convert(AbstractMeasureOrDensity, target), context)
    vs = varshape(target)

    init = mcmc_init!(
        algorithm,
        density,
        algorithm.nchains,
        apply_trafo_to_init(pre_trafo, algorithm.init),
        algorithm.tuning_alg,
        algorithm.nonzero_weights,
        algorithm.store_burnin ? algorithm.callback : nop_func,
        context
    )
    

    @unpack chains, tuners, temperers = init

    burnin_outputs_coll = if algorithm.store_burnin
        DensitySampleVector(first(chains))
    else
        nothing
    end

    # burnin and tuning  # @TODO: Hier wird noch kein ensemble BurnIn gemacht !!!!!!!!!!!!!!!
   # mcmc_burnin!(
   #     burnin_outputs_coll,
   #     chains,
   #     tuners,
   #     temperers,
   #     algorithm.burnin,
   #     algorithm.convergence,
   #     algorithm.strict,
   #     algorithm.nonzero_weights,
   #     algorithm.store_burnin ? algorithm.callback : nop_func
   # )


    # sampling
    run_sampling = _run_sample_impl_ensemble(
        density,
        algorithm,
        chains,
        tuners,
        temperers
    )

    samples_trafo, generator = run_sampling.result_trafo, run_sampling.generator

    # prepend burnin samples to output
    if algorithm.store_burnin
        burnin_samples_trafo = burnin_outputs_coll
        prepend!(samples_trafo, burnin_samples_trafo)
    end

    global g_state_post_algorithm = (pre_trafo, samples_trafo, vs, density, algorithm, chains, tuners, temperers, target, context)

    samples_notrafo = inverse(pre_trafo).(samples_trafo)
    #samples_notrafo = vs.(inverse(pre_trafo).(samples_trafo))

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = pre_trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end


#=
function _bat_sample_continue(
    target::AnyMeasureOrDensity,
    generator::TransformedMCMCSampleGenerator,
    ;description::AbstractString = "MCMC iterate"
)
    @unpack algorithm, chains = generator
    density_notrafo = convert(AbstractMeasureOrDensity, target)
    density, trafo = transform_and_unshape(algorithm.pre_transform, density_notrafo)

    run_sampling = _run_sample_impl(density, algorithm, chains, description=description)

    samples_trafo, generator = run_sampling.result_trafo, run_sampling.generator

    samples_notrafo = inverse(trafo).(samples_trafo)

    (result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end
=#

function _run_sample_impl(
    density::AnyMeasureOrDensity,
    algorithm::TransformedMCMCSampling,
    chains::AbstractVector{<:MCMCIterator},
    ;description::AbstractString = "MCMC iterate"
)

    next_cycle!.(chains) 

    progress_meter = ProgressMeter.Progress(algorithm.nchains*algorithm.nsteps, desc=description, barlen=80-length(description), dt=0.1)

    # tuners are set to 'NoOpTuner' for the sampling phase
    for i in 1:algorithm.nsteps
        transformed_mcmc_iterate!(
            chains,
            get_tuner.(Ref(TransformedMCMCNoOpTuning()),chains),
            get_temperer.(Ref(TransformedNoTransformedMCMCTempering()), chains),
            max_nsteps = algorithm.nsteps, #TODO: maxtime
            nonzero_weights = algorithm.nonzero_weights,
            callback = (kwargs...) -> let pm=progress_meter; ProgressMeter.next!(pm) ; end,
        )
    end

    ProgressMeter.finish!(progress_meter)


    output = reduce(vcat, getproperty.(chains, :samples))
    samples_trafo = varshape(density).(output)


    (result_trafo = samples_trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end

function _run_sample_impl_ensemble(
    density::AnyMeasureOrDensity,
    algorithm::TransformedMCMCSampling,
    chains::AbstractVector{<:MCMCIterator},
    tuner,
    temperer,
    ;description::AbstractString = "MCMC iterate"
)

    next_cycle!.(chains) 

    progress_meter = ProgressMeter.Progress(algorithm.nchains*algorithm.nsteps, desc=description, barlen=80-length(description), dt=0.1)

    # tuners are set to 'NoOpTuner' for the sampling phase
    for i in 1:algorithm.nsteps
        transformed_mcmc_iterate!(
            chains,
            tuner,#get_tuner.(Ref(TransformedMCMCNoOpTuning()),chains),
            temperer, #get_temperer.(Ref(TransformedNoTransformedMCMCTempering()), chains),
            max_nsteps = algorithm.nsteps, #TODO: maxtime
            nonzero_weights = algorithm.nonzero_weights,
            callback = (kwargs...) -> let pm=progress_meter; ProgressMeter.next!(pm) ; end,
        )
    end
    ProgressMeter.finish!(progress_meter)

    output = copy(chains[1].states_x[1])
    for chain in chains
        for i in 1:length(chain.states_x)
            append!(output,chain.states_x[i])
        end
    end

    samples_trafo = varshape(density).(output[1:end])

    (result_trafo = samples_trafo, generator = TransformedMCMCSampleGenerator(chains, algorithm))
end
#
#    output = reduce(vcat, getfield.(chains, :states_x))
#
#    (result_trafo = output, generator = TransformedMCMCSampleGenerator(chains, algorithm))
#end