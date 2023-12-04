mutable struct TransformedMCMCEnsembleIterator{
    PR<:RNGPartition,
    D<:BATMeasure,
    F<:Function,
    Q<:TransformedMCMCProposal,
    SV,#<:Vector{DensitySampleVector},
    S<:DensitySampleVector,
    CTX<:BATContext,
} <: MCMCIterator
    rngpart_cycle::PR
    μ::D
    f::F
    proposal::Q
    states_x::SV
    state_z::S
    stepno::Int
    n_walker::Int
    n_accepted::Vector{Int}
    info::TransformedMCMCIteratorInfo
    context::CTX
end

getmeasure(ensemble::TransformedMCMCEnsembleIterator) = ensemble.μ

get_context(ensemble::TransformedMCMCEnsembleIterator) = ensemble.context

mcmc_info(ensemble::TransformedMCMCEnsembleIterator) = ensemble.info 

nsteps(ensemble::TransformedMCMCEnsembleIterator) = ensemble.stepno

nsamples(ensemble::TransformedMCMCEnsembleIterator) = length(ensemble.states_x) # @TODO

current_ensemble(ensemble::TransformedMCMCEnsembleIterator) = last(ensemble.states_x)

sample_type(ensemble::TransformedMCMCEnsembleIterator) = eltype(ensemble.state_z)

samples_available(ensemble::TransformedMCMCEnsembleIterator) = size(ensemble.states_x,1) > 0

isvalidchain(ensemble::TransformedMCMCEnsembleIterator) = min(current_sample(ensemble).logd) > -Inf

isviablechain(ensemble::TransformedMCMCEnsembleIterator) = nsamples(ensemble) >= 2

eff_acceptance_ratio(ensemble::TransformedMCMCEnsembleIterator) = nsamples(ensemble) / ensemble.stepno


#ctor
function TransformedMCMCEnsembleIterator(
    algorithm::TransformedMCMCSampling,
    target,
    id::Integer,
    v_init::AbstractVector{<:Real},
    context::BATContext
) 
    TransformedMCMCEnsembleIterator(algorithm, target, Int32(id), v_init, context)
end


export TransformedMCMCEnsembleIterator
function TransformedMCMCEnsembleIterator(
    algorithm::TransformedMCMCSampling,
    target,
    id::Int32,
    v_init::AbstractVector{},
    context::BATContext,
)
    rngpart_cycle = RNGPartition(get_rng(context), 0:(typemax(Int16) - 2))

    μ = target
    proposal = algorithm.proposal
    stepno = 1
    n_walker=100
    cycle = 1
    n_accepted = zeros(Int64,length(v_init))

    adaptive_transform_spec = algorithm.adaptive_transform
    f = init_adaptive_transform(adaptive_transform_spec, μ, context)

    logd_x = logdensityof(μ).(v_init)
    states = DensitySampleVector.([(v_init, logd_x, ones(length(logd_x)), fill(TransformedMCMCTransformedSampleID(id, 1, 0),length(logd_x)), fill(nothing,length(logd_x)))])

    state_z = f(states[end])
    
    iter = TransformedMCMCEnsembleIterator(
        rngpart_cycle,
        target,
        f,
        proposal,
        states,
        state_z,
        stepno,
        n_walker,
        n_accepted,
        TransformedMCMCIteratorInfo(id, cycle, false, false),
        context
    )
end


function _rebuild_density_sample_vector(s::DensitySampleVector, x, logd, weight=ones(length(x)))
    @unpack info, aux = s
    DensitySampleVector((x, logd, weight, info, aux))
end


function propose_mcmc(
    iter::TransformedMCMCEnsembleIterator{<:Any, <:Any, <:Any, <:TransformedMHProposal}
)
    @unpack μ, f, proposal, states_x, state_z, stepno, context = iter

    rng = get_rng(context)
    states_x = last(states_x)
    x, logd_x = states_x.v, states_x.logd
    z, logd_z = flatview(state_z.v), state_z.logd
    f_inv = inverse(f)
    n, m = size(z)

    z_proposed = z + rand(rng, proposal.proposal_dist, (n,m)) #TODO: check if proposal is symmetric? otherwise need additional factor?
        
    x_proposed, ladj = with_logabsdet_jacobian(f_inv, z_proposed)

    z_proposed = nestedview(z_proposed)
    x_proposed = nestedview(x_proposed)

    logd_x_proposed = BAT.checked_logdensityof(μ).(x_proposed)
    logd_z_proposed = logd_x_proposed + vec(ladj)

    @assert logd_z_proposed ≈ logdensityof(MeasureBase.pullback(f_inv, μ)).(z_proposed) #TODO: remove

    p_accept = clamp.(exp.(logd_z_proposed-logd_z), 0, 1)

    state_z_proposed = _rebuild_density_sample_vector(state_z, z_proposed, logd_z_proposed)
    state_x_proposed = _rebuild_density_sample_vector(states_x, x_proposed, logd_x_proposed)

    return state_x_proposed, state_z_proposed, p_accept
end


function propose_mala(
    iter::TransformedMCMCEnsembleIterator{<:Any, <:Any, <:Any, <:TransformedMHProposal}
)
    @unpack μ, f, proposal, states_x, state_z, stepno, context = iter
    rng = get_rng(context)
    AD_sel = context.ad
    f_inv = inverse(f)

    z = state_z.v
    logd_x = last(states_x).logd

    z_proposed = similar(z)

    tau = 0.01
   
    ν = Transformed(μ, f, TDLADJCorr())
    log_ν = BAT.checked_logdensityof(ν)
    ∇log_ν = gradient_func(log_ν, AD_sel)

    for i in 1:length(z)
        if (rand() < (1/length(z_proposed)))
            z_proposed[i] = rand(MvNormal(zeros(length(z[i])),I(length(z[i]))))
        else
            #z_proposed[i] = z[i] + sqrt(2*tau) .* rand(rng, proposal.proposal_dist, n) + tau .* ∇log_ν(z[i])
            z_proposed[i] = z[i] + sqrt(2*tau) * rand(MvNormal(zeros(length(z[i])),ones(length(z[i])))) + tau .* ∇log_ν(z[i])
        end
    end  

    logd_z_proposed = log_ν.(z_proposed)

    x_proposed = f_inv(z_proposed)

    for i in 1:length(x_proposed)
        if any(isnan.(x_proposed[i]))
            x_proposed[i] = last(states_x).v[i]
            println("Fucking NaN!")
        end
    end

    logd_x_proposed = BAT.checked_logdensityof(μ).(x_proposed)

    p_accept = clamp.(exp.(logd_x_proposed-logd_x), 0, 1)

    state_x_proposed = _rebuild_density_sample_vector(last(states_x), x_proposed, logd_x_proposed)
    state_z_proposed = _rebuild_density_sample_vector(state_z, z_proposed, logd_z_proposed)

    return state_x_proposed, state_z_proposed, p_accept
end


function transformed_mcmc_step!!(
    iter::TransformedMCMCEnsembleIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance,
)
    @unpack  μ, f, proposal, states_x, state_z, stepno, n_accepted, context = iter
    rng = get_rng(context)
    state_x = last(states_x)

    state_x_proposed, state_z_proposed, p_accept = propose_mala(iter)

    z_proposed, logd_z_proposed = state_z_proposed.v, state_z_proposed.logd
    x_proposed, logd_x_proposed = state_x_proposed.v, state_x_proposed.logd
    
    accepted = rand(rng, length(p_accept)) .<= p_accept

    x_new, logd_x_new = state_x.v, state_x.logd
    z_new, logd_z_new = state_z.v, state_z.logd

    x_new[accepted], z_new[accepted], logd_x_new[accepted], logd_z_new[accepted] = x_proposed[accepted], z_proposed[accepted], logd_x_proposed[accepted], logd_z_proposed[accepted]

    state_x_new = DensitySampleVector((x_new, logd_x_new,ones(length(x_new)), fill(TransformedMCMCTransformedSampleID(iter.info.id, iter.info.cycle, iter.stepno), length(x_new)), fill(nothing,length(x_new))))

    push!(states_x, state_x_new) 

    state_z_new = _rebuild_density_sample_vector(state_z, z_new, logd_z_new)

    if (tuner isa MCMCFlowTuner)
        tuner_new, f_new = tune_mcmc_transform!!(tuner, f, state_x_new.v, context)
        f=f_new.result
    else
        tuner_new, f = tune_mcmc_transform!!(tuner, f, p_accept, z_proposed,[2,3,4] , stepno, context)
    end
    tempering_new, μ_new = temper_mcmc_target!!(tempering, μ, stepno)

    f_new = f

    iter.μ, iter.f, iter.states_x, iter.state_z = μ_new, f_new, states_x, state_z_new
    iter.n_accepted += Int.(accepted)
    iter.stepno += 1
    @assert iter.context === context

    return (iter, tuner_new, tempering_new)
end


function transformed_mcmc_trafo_proposal_step!!(
    ensemble::TransformedMCMCEnsembleIterator,
    tempering::TransformedMCMCTemperingInstance
)
    @unpack  μ, f, proposal, states_x, state_z, stepno, context = ensemble
    rng = get_rng(context)
    proposal_dist = proposal.proposal_dist
    f_inv = inverse(f)

    state_x = last(states_x)

    state_z_proposed = nestedview(rand(MvNormal(zeros(length(x[1])),I(length(x[1]))),length(x)))#bat_sample_impl(proposal_dist, BAT.IIDSampling(length(state_x)), context).result

    x_proposed = f_inv(state_z_proposed)

    logd_x_proposed = logdensityof(μ).(x_proposed)

    p_accept = clamp.(exp.(logd_x_proposed - logd_x), 0, 1)

    accepted = rand(rng, length(p_accept)) .<= p_accept
        
    x_new, logd_x_new = state_x.v, state_x.logd
    z_new, logd_z_new = state_z.v, state_z.logd

    x_new[accepted], z_new[accepted], logd_x_new[accepted], logd_z_new[accepted] = x_proposed[accepted], z_proposed[accepted], logd_x_proposed[accepted], logd_z_proposed[accepted]

    state_x_new = DensitySampleVector((x_new, logd_x_new, ones(length(x_new)), fill(TransformedMCMCTransformedSampleID(ensemble.info.id, ensemble.info.cycle, ensemble.stepno), length(x_new)), fill(nothing, length(x_new))))
    push!(states_x, state_x_new) 

    state_z_new = _rebuild_density_sample_vector(state_z, z_new, logd_z_new)

    tempering_new, μ_new = temper_mcmc_target!!(tempering, μ, stepno)

    ensemble.μ, ensemble.state_z = μ_new, state_z_new
    ensemble.n_accepted += Int.(accepted)
    ensemble.stepno += 1
    @assert ensemble.context === context

    return (ensemble, tempering_new)
end

#function transformed_mcmc_iterate!(
#    ensemble::TransformedMCMCEnsembleIterator,
#    tuner::MCMCFlowTuner,
#    tempering::TransformedMCMCTemperingInstance;
#    max_nsteps::Integer = 1,
#    max_time::Real = Inf,
#    nonzero_weights::Bool = true,
#    callback::Function = nop_func,
#    transformed_proposal_idx::Integer = 10
#)
#    @debug "Starting iteration over MCMC ensemble $(mcmc_info(ensemble).id) with $max_nsteps steps in max. $(@sprintf "%.1f seconds." max_time)"
#    
#    start_time = time()
#    last_progress_message_time = start_time
#    start_nsteps = nsteps(ensemble)
#    start_nsteps = nsteps(ensemble)
#
#    while (
#        (nsteps(ensemble) - start_nsteps) < max_nsteps &&
#        (time() - start_time) < max_time
#    )
#        if nsteps(ensemble)%transformed_proposal_idx == 0
#            transformed_mcmc_trafo_proposal_step!!(ensemble, tempering)
#        else
#            transformed_mcmc_step!!(ensemble, tuner, tempering)
#        end
#
#        callback(Val(:mcmc_step), ensemble)
#
#        current_time = time()
#        elapsed_time = current_time - start_time
#        logging_interval = 5 * round(log2(elapsed_time/60 + 1) + 1)
#        if current_time - last_progress_message_time > logging_interval
#            last_progress_message_time = current_time
#            @debug "Iterating over MCMC ensemble $(mcmc_info(ensemble).id), completed $(nsteps(ensemble) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsteps(ensemble) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time) so far."
#        end
#    end
#
#    current_time = time()
#    elapsed_time = current_time - start_time
#    @debug "Finished iteration over MCMC ensemble $(mcmc_info(ensemble).id), completed $(nsteps(ensemble) - start_nsteps) steps and produced $(nsteps(ensemble) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time)."
#    println("Finished iteration over MCMC ensemble $(mcmc_info(ensemble).id), completed $(nsteps(ensemble) - start_nsteps) steps and produced $(nsteps(ensemble) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time).")
#    return nothing
#end
#
#
#function transformed_mcmc_iterate!(
#    ensemble::MCMCIterator,
#    tuner::TransformedAbstractMCMCTunerInstance,
#    tempering::TransformedMCMCTemperingInstance;
#    # tuner::TransformedAbstractMCMCTunerInstance;
#    max_nsteps::Integer = 1,
#    max_time::Real = Inf,
#    nonzero_weights::Bool = true,
#    callback::Function = nop_func
#)
#    cb = callback# combine_callbacks(tuning_callback(tuner), callback) #TODO CA: tuning_callback
#    
#    transformed_mcmc_iterate!(
#        ensemble, tuner, tempering,
#        max_nsteps = max_nsteps, max_time = max_time, nonzero_weights = nonzero_weights, callback = cb
#    )
#
#    return nothing
#end
##
##
##function transformed_mcmc_iterate!(
#    chains::AbstractVector{<:MCMCIterator},
#    tuners::AbstractVector{<:TransformedAbstractMCMCTunerInstance},
#    temperers::AbstractVector{<:TransformedMCMCTemperingInstance};
#    kwargs...
#)
#    if isempty(chains)
#        @debug "No MCMC ensemble(s) to iterate over."
#        return chains
#    else
#        @debug "Starting iteration over $(length(chains)) MCMC ensemble(s)"
#    end
#
#    @sync for i in eachindex(chains, tuners, temperers)
#        Base.Threads.@spawn transformed_mcmc_iterate!(chains[i], tuners[i], temperers[i]#= , tnrs[i] =#; kwargs...)
#    end
#
#    return nothing
#end


function transformed_mcmc_iterate!( 
    ensembles::AbstractVector{<:TransformedMCMCEnsembleIterator},
    tuners::AbstractVector{<:TransformedAbstractMCMCTunerInstance},
    temperers::AbstractVector{<:TransformedMCMCTemperingInstance};
    kwargs...
)

    if isempty(ensembles)
        @debug "No MCMC ensemble(s) to iterate over."
        return ensembles
    else
        @debug "Starting iteration over $(length(ensembles)) MCMC ensemble(s)"
    end
    
    for i in 1:length(ensembles)
        transformed_mcmc_step!!(ensembles[i], tuners[i],temperers[i])
    end

    return nothing
end


function reset_rng_counters!(ensemble::TransformedMCMCEnsembleIterator)
    rng = get_rng(get_context(ensemble))
    set_rng!(rng, ensemble.rngpart_cycle, ensemble.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, ensemble.stepno)
    nothing
end


function next_cycle!(
    ensemble::TransformedMCMCEnsembleIterator,

)
    ensemble.info = TransformedMCMCIteratorInfo(ensemble.info, cycle = ensemble.info.cycle + 1)
    ensemble.stepno = 0

    reset_rng_counters!(ensemble)

    ensemble.states_x[1] = last(ensemble.states_x)
    resize!(ensemble.states_x, 1)

    ensemble.states_x[1].weight[1] = 1
    ensemble.states_x[1].info[1] = TransformedMCMCTransformedSampleID(ensemble.info.id, ensemble.info.cycle, ensemble.stepno)
    
    ensemble
end