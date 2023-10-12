mutable struct TransformedMCMCEnsembleIterator{
    PR<:RNGPartition,
    D<:BATMeasure,
    F,
    Q<:TransformedMCMCProposal,
    SV<:Vector{DensitySampleVector},
    S<:DensitySampleVector,
    CTX<:BATContext,
} <: MCMCIterator
    rngpart_cycle::PR
    μ::D
    f_transform::F
    proposal::Q
    ensembles::SV
    samples_z::S
    stepno::Int
    n_accepted::Vector{Int}
    info::TransformedMCMCIteratorInfo
    context::CTX
end

getmeasure(chain::TransformedMCMCEnsembleIterator) = chain.μ

get_context(chain::TransformedMCMCEnsembleIterator) = chain.context

mcmc_info(chain::TransformedMCMCEnsembleIterator) = chain.info 

nsteps(chain::TransformedMCMCEnsembleIterator) = chain.stepno

nsamples(chain::TransformedMCMCEnsembleIterator) = length(chain.ensembles) # @TODO

current_ensemble(chain::TransformedMCMCEnsembleIterator) = last(chain.ensembles)

sample_type(chain::TransformedMCMCEnsembleIterator) = eltype(chain.samples_z)

samples_available(chain::TransformedMCMCEnsembleIterator) = size(chain.ensembles,1) > 0

isvalidchain(chain::TransformedMCMCEnsembleIterator) = min(current_sample(chain).logd) > -Inf

isviablechain(chain::TransformedMCMCEnsembleIterator) = nsamples(chain) >= 2

eff_acceptance_ratio(chain::TransformedMCMCEnsembleIterator) = nsamples(chain) / chain.stepno



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


#ctor
g_state = (;)
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
    cycle = 1
    n_accepted = zeros(Int64,length(v_init))

    adaptive_transform_spec = algorithm.adaptive_transform
    g = init_adaptive_transform(adaptive_transform_spec, μ, context)

    logd_x = logdensityof(μ).(v_init)
    state_x = DensitySampleVector((v_init, logd_x, ones(length(logd_x)), fill(TransformedMCMCTransformedSampleID(id, 1, 0),length(logd_x)), fill(nothing,length(logd_x))))
    inverse_g = inverse(g)
    z = vec(inverse_g.(v_init)) # sample_x.v 
    global g_state = (state_x)

    logd_z = logdensityof(MeasureBase.pullback(g, μ)).(z)
    state_z = _rebuild_density_sample_vector(state_x, z, logd_z) 
    
    states = Vector{DensitySampleVector}(undef,1)
    states[1] = state_x
    iter = TransformedMCMCEnsembleIterator(
        rngpart_cycle,
        target,
        g,
        proposal,
        states,
        state_z,
        stepno,
        n_accepted,
        TransformedMCMCIteratorInfo(id, cycle, false, false),
        context
    )
end


function _rebuild_density_sample_vector(s::DensitySampleVector, x, logd, weight=ones(length(x)))
    @unpack info, aux = s
    DensitySampleVector((x, logd, weight, info, aux))
end

g_state=(;)

function propose_mcmc(
    iter::TransformedMCMCEnsembleIterator{<:Any, <:Any, <:Any, <:TransformedMHProposal}
)
    @unpack μ, f_transform, proposal, ensembles, samples_z, stepno, context = iter
    rng = get_rng(context)
    samples_x = last(ensembles)
    x, logd_x = samples_x.v, samples_x.logd
    z, logd_z = flatview(unshaped.(samples_z.v)), samples_z.logd

    global g_state = (z)
    n, m = size(z)

    z_proposed = z + rand(rng, proposal.proposal_dist, (n,m)) #TODO: check if proposal is symmetric? otherwise need additional factor?
        
    x_proposed, ladj = with_logabsdet_jacobian(f_transform, z_proposed)

    z_proposed = nestedview(z_proposed)
    x_proposed = nestedview(x_proposed)

    logd_x_proposed = BAT.checked_logdensityof(μ).(x_proposed)
    logd_z_proposed = logd_x_proposed + vec(ladj)

    @assert logd_z_proposed ≈ logdensityof(MeasureBase.pullback(f_transform, μ)).(z_proposed) #TODO: remove

    # TODO AC: do we need to check symmetry of proposal distribution?
    # T = typeof(logd_z)
    # p_accept = if logd_z_proposed > -Inf
    #     # log of ratio of forward/reverse transition probability
    #     log_tpr = if issymmetric(proposal.proposal_dist)
    #         T(0)
    #     else
    #         log_tp_fwd = proposaldist_logpdf(proposaldist, proposed_params, current_params)
    #         log_tp_rev = proposaldist_logpdf(proposaldist, current_params, proposed_params)
    #         T(log_tp_fwd - log_tp_rev)
    #     end

    #     p_accept_unclamped = exp(proposed_log_posterior - current_log_posterior - log_tpr)
    #     T(clamp(p_accept_unclamped, 0, 1))
    # else
    #     zero(T)
    # end

    p_accept = clamp.(exp.(logd_z_proposed-logd_z), 0, 1)

    sample_z_proposed = _rebuild_density_sample_vector(samples_z, z_proposed, logd_z_proposed)
    sample_x_proposed = _rebuild_density_sample_vector(samples_x, x_proposed, logd_x_proposed)

    return sample_x_proposed, sample_z_proposed, p_accept
end


using Flux
import Flux: gradient
function propose_mala(
    iter::TransformedMCMCEnsembleIterator{<:Any, <:Any, <:Any, <:TransformedMHProposal}
)
    @unpack μ, f_transform, proposal, ensembles, samples_z, stepno, context = iter
    rng = get_rng(context)
    samples_x = last(ensembles)
    x, logd_x = samples_x.v, samples_x.logd
    z, logd_z = samples_z.v, samples_z.logd

    n = size(z[1], 1)
    #z_proposed = z + rand(rng, proposal.proposal_dist, n) #TODO: check if proposal is symmetric? otherwise need additional factor?
    tau = 0.1

    global g_state=(z)
    z_proposed = []
    for i in 1:length(z)
        z_proposed[i] = z[i] + sqrt(2*tau)*rand(rng, proposal.proposal_dist, n) #+ tau*Flux.gradient(x -> logdensityof(μ,x),z)[1]
    end  

    return z_proposed

    x_proposed, ladj = with_logabsdet_jacobian(f_transform, z_proposed)
    logd_x_proposed = BAT.checked_logdensityof(μ, x_proposed)
    logd_z_proposed = logd_x_proposed + ladj
    @assert logd_z_proposed ≈ logdensityof(MeasureBase.pullback(f_transform, μ), z_proposed) #TODO: remove

    
    # TODO AC: do we need to check symmetry of proposal distribution?
    # T = typeof(logd_z)
    # p_accept = if logd_z_proposed > -Inf
    #     # log of ratio of forward/reverse transition probability
    #     log_tpr = if issymmetric(proposal.proposal_dist)
    #         T(0)
    #     else
    #         log_tp_fwd = proposaldist_logpdf(proposaldist, proposed_params, current_params)
    #         log_tp_rev = proposaldist_logpdf(proposaldist, current_params, proposed_params)
    #         T(log_tp_fwd - log_tp_rev)
    #     end

    #     p_accept_unclamped = exp(proposed_log_posterior - current_log_posterior - log_tpr)
    #     T(clamp(p_accept_unclamped, 0, 1))
    # else
    #     zero(T)
    # end

    p_accept = clamp(exp(logd_z_proposed-logd_z), 0, 1)

    sample_z_proposed = _rebuild_density_sample(samples_z, z_proposed, logd_z_proposed)
    sample_x_proposed = _rebuild_density_sample(samples_x, x_proposed, logd_x_proposed)

    return sample_x_proposed, sample_z_proposed, p_accept
end


function transformed_mcmc_step!!(
    iter::TransformedMCMCEnsembleIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance,
)
    @unpack  μ, f_transform, proposal, ensembles, samples_z, stepno, context = iter
    rng = get_rng(context)
    samples_x = last(ensembles)
    x, logd_x = samples_x.v, samples_x.logd
    z, logd_z = samples_z.v, samples_z.logd
    @unpack n_accepted, stepno = iter

    sample_x_proposed, sample_z_proposed, p_accept = propose_mcmc(iter)

    z_proposed, logd_z_proposed = sample_z_proposed.v, sample_z_proposed.logd
    x_proposed, logd_x_proposed = sample_x_proposed.v, sample_x_proposed.logd

    global g_state = (p_accept, z_proposed, z)
    tuner_new, f_transform = tune_mcmc_transform!!(tuner, f_transform, p_accept, z_proposed, z, stepno, context)
    
    accepted = rand(rng, length(p_accept)) .<= p_accept

    # f_transform may have changed
    # x_new, z_new, logd_x_new, logd_z_new = if accepted
        #     x_proposed, inverse_f(x_proposed), logd_x_proposed, logd_z_proposed
        # else
        #     x, inverse_f(x), logd_x, logd_z
        # end
        
    inverse_f = inverse(f_transform)
    n_walkers = length(z_proposed)
    x_new = Vector{Vector}(undef, n_walkers)
    z_new = Vector{Vector{Float64}}(undef, n_walkers)
    logd_x_new = Vector{Float64}(undef, n_walkers)
    logd_z_new = Vector{Float64}(undef, n_walkers)

    z_new_temp = nestedview(inverse_f(flatview(unshaped.(x_proposed))))
    g_state = (x)
    x_inv_temp = nestedview(inverse_f(hcat(unshaped.(x)...)))

    accepted_neg = .! accepted 
    x_new[accepted], z_new[accepted], logd_x_new[accepted], logd_z_new[accepted] = x_proposed[accepted], z_new_temp[accepted], logd_x_proposed[accepted], logd_z_proposed[accepted]
    x_new[accepted_neg], z_new[accepted_neg], logd_x_new[accepted_neg], logd_z_new[accepted_neg] = x[accepted_neg], x_inv_temp[accepted_neg], logd_x[accepted_neg], logd_z[accepted_neg]

    state_x_new = DensitySampleVector((x_new,logd_x_new,ones(length(x_new)), fill(TransformedMCMCTransformedSampleID(iter.info.id, iter.info.cycle, iter.stepno),length(x_new)), fill(nothing,length(x_new))))

    global g_state=(ensembles,state_x_new)
    push!(ensembles, state_x_new) 

    samples_z_new = _rebuild_density_sample_vector(samples_z, z_new, logd_z_new)


    tempering_new, μ_new = temper_mcmc_target!!(tempering, μ, stepno)

    f_new = f_transform

    # iter_new = TransformedMCMCEnsembleIterator(μ_new, f_new, proposal, samples_new, sample_z_new, stepno, n_accepted+Int(accepted), context)

    global g_state = (samples_z_new,iter.samples_z)
    iter.μ, iter.f_transform, iter.ensembles, iter.samples_z = μ_new, f_new, ensembles, samples_z_new
    iter.n_accepted += Int.(accepted)
    iter.stepno += 1
    @assert iter.context === context

    return (iter, tuner_new, tempering_new)
end



function transformed_mcmc_iterate!(
    chain::TransformedMCMCEnsembleIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func,
)
    @debug "Starting iteration over MCMC chain $(mcmc_info(chain).id) with $max_nsteps steps in max. $(@sprintf "%.1f seconds." max_time)"

    start_time = time()
    last_progress_message_time = start_time
    start_nsteps = nsteps(chain)
    start_nsteps = nsteps(chain)

    while (
        (nsteps(chain) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        transformed_mcmc_step!!(chain, tuner, tempering)
        callback(Val(:mcmc_step), chain)
  
        #TODO: output schemes

        current_time = time()
        elapsed_time = current_time - start_time
        logging_interval = 5 * round(log2(elapsed_time/60 + 1) + 1)
        if current_time - last_progress_message_time > logging_interval
            last_progress_message_time = current_time
            @debug "Iterating over MCMC chain $(mcmc_info(chain).id), completed $(nsteps(chain) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsteps(chain) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time) so far."
        end
    end

    current_time = time()
    elapsed_time = current_time - start_time
    @debug "Finished iteration over MCMC chain $(mcmc_info(chain).id), completed $(nsteps(chain) - start_nsteps) steps and produced $(nsteps(chain) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time)."
    println("Finished iteration over MCMC chain $(mcmc_info(chain).id), completed $(nsteps(chain) - start_nsteps) steps and produced $(nsteps(chain) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time).")
    return nothing
end


function transformed_mcmc_iterate!(
    chain::MCMCIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance;
    # tuner::TransformedAbstractMCMCTunerInstance;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
)
    cb = callback# combine_callbacks(tuning_callback(tuner), callback) #TODO CA: tuning_callback
    
    transformed_mcmc_iterate!(
        chain, tuner, tempering,
        max_nsteps = max_nsteps, max_time = max_time, nonzero_weights = nonzero_weights, callback = cb
    )

    return nothing
end

t = []
function transformed_mcmc_iterate!(
    chains::AbstractVector{<:MCMCIterator},
    tuners::AbstractVector{<:TransformedAbstractMCMCTunerInstance},
    temperers::AbstractVector{<:TransformedMCMCTemperingInstance};
    kwargs...
)
    global t = tuners
    Hallo2
    if isempty(chains)
        @debug "No MCMC chain(s) to iterate over."
        return chains
    else
        @debug "Starting iteration over $(length(chains)) MCMC chain(s)"
    end

    @sync for i in eachindex(chains, tuners, temperers)
        Base.Threads.@spawn transformed_mcmc_iterate!(chains[i], tuners[i], temperers[i]#= , tnrs[i] =#; kwargs...)
    end

    return nothing
end

function transformed_mcmc_iterate!(
    chains::AbstractVector{<:MCMCIterator},
    tuners::AbstractVector{<:TransformedAbstractMCMCTunerInstance},
    temperers::AbstractVector{<:TransformedMCMCTemperingInstance};
    kwargs...
)
    if isempty(chains)
        @debug "No MCMC chain(s) to iterate over."
        return chains
    else
        @debug "Starting iteration over $(length(chains)) MCMC chain(s)"
    end
    
    transformed_mcmc_step!!(chains, tuners,temperers)
    return nothing
end
#=
# Unused?
function reset_chain(
    rng::AbstractRNG,
    chain::TransformedMCMCEnsembleIterator,
)
    rngpart_cycle = RNGPartition(rng, 0:(typemax(Int16) - 2))
    #TODO reset cycle count?
    chain.rngpart_cycle = rngpart_cycle
    chain.info = TransformedMCMCEnsembleIteratorInfo(chain.info, cycle=0)
    chain.context = set_rng(chain.context, rng)
    # wants a next_cycle!
    # reset_rng_counters!(chain)
end
=#


function reset_rng_counters!(chain::TransformedMCMCEnsembleIterator)
    rng = get_rng(get_context(chain))
    set_rng!(rng, chain.rngpart_cycle, chain.info.cycle)
    rngpart_step = RNGPartition(rng, 0:(typemax(Int32) - 2))
    set_rng!(rng, rngpart_step, chain.stepno)
    nothing
end


function next_cycle!(
    chain::TransformedMCMCEnsembleIterator,

)
    chain.info = TransformedMCMCEnsembleIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    chain.stepno = 0

    reset_rng_counters!(chain)

    chain.samples[1] = last(chain.samples)
    resize!(chain.samples, 1)

    chain.samples.weight[1] = 1
    chain.samples.info[1] = TransformedMCMCTransformedSampleID(chain.info.id, chain.info.cycle, chain.stepno)
    
    chain
end

