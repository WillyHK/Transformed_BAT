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
    f::F
    proposal::Q
    states_x::SV
    state_z::S
    stepno::Int
    n_accepted::Vector{Int}
    info::TransformedMCMCIteratorInfo
    context::CTX
end

getmeasure(chain::TransformedMCMCEnsembleIterator) = chain.μ

get_context(chain::TransformedMCMCEnsembleIterator) = chain.context

mcmc_info(chain::TransformedMCMCEnsembleIterator) = chain.info 

nsteps(chain::TransformedMCMCEnsembleIterator) = chain.stepno

nsamples(chain::TransformedMCMCEnsembleIterator) = length(chain.states_x) # @TODO

current_ensemble(chain::TransformedMCMCEnsembleIterator) = last(chain.states_x)

sample_type(chain::TransformedMCMCEnsembleIterator) = eltype(chain.state_z)

samples_available(chain::TransformedMCMCEnsembleIterator) = size(chain.states_x, 1) > 0

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
    f = init_adaptive_transform(adaptive_transform_spec, μ, context)

    logd_x = logdensityof(μ).(v_init)
    state_x = DensitySampleVector((v_init, logd_x, ones(length(logd_x)), fill(TransformedMCMCTransformedSampleID(id, 1, 0),length(logd_x)), fill(nothing,length(logd_x))))
    f_inv = inverse(f)
    
    state_z = f_inv(state_x)
    
    states = Vector{DensitySampleVector}(undef,1)
    states[1] = state_x
    iter = TransformedMCMCEnsembleIterator(
        rngpart_cycle,
        target,
        f,
        proposal,
        states,
        state_z,
        stepno,
        n_accepted,
        TransformedMCMCIteratorInfo(id, cycle, false, false),
        context
    )
end


function propose_mcmc(
    iter::TransformedMCMCEnsembleIterator{<:Any, <:Any, <:Any, <:TransformedMHProposal}
)
    @unpack μ, f, proposal, states_x, state_z, stepno, context = iter
    rng = get_rng(context)
    states_x = last(states_x)
    x, logd_x = states_x.v, states_x.logd
    z, logd_z = flatview(unshaped.(state_z.v)), state_z.logd

    n, m = size(z)

    z_proposed = z + rand(rng, proposal.proposal_dist, (n,m)) #TODO: check if proposal is symmetric? otherwise need additional factor?
        
    x_proposed, ladj = with_logabsdet_jacobian(f, z_proposed)

    z_proposed = nestedview(z_proposed)
    x_proposed = nestedview(x_proposed)

    logd_x_proposed = BAT.checked_logdensityof(μ).(x_proposed)
    logd_z_proposed = logd_x_proposed + vec(ladj)

    @assert logd_z_proposed ≈ logdensityof(MeasureBase.pullback(f, μ)).(z_proposed) #TODO: remove

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

    state_z_proposed = _rebuild_density_sample_vector(state_z, z_proposed, logd_z_proposed)
    state_x_proposed = _rebuild_density_sample_vector(states_x, x_proposed, logd_x_proposed)

    return state_x_proposed, state_z_proposed, p_accept
end

g_state_1 = (;)
g_state_2 = (;)


function propose_mala(
    iter::TransformedMCMCEnsembleIterator{<:Any, <:Any, <:Any, <:TransformedMHProposal}
)
    @unpack μ, f, proposal, states_x, state_z, stepno, context = iter
    rng = get_rng(context)
    AD_sel = context.ad

    z, logd_z = unshaped.(state_z.v), state_z.logd
    z_proposed = similar(z)

    n = size(z[1], 1)
    tau = 0.001

    μ_flat = unshaped(μ)    
    ν = Transformed(μ_flat, inverse(f), TDLADJCorr())
    log_ν = BAT.checked_logdensityof(ν)
    ∇log_ν = gradient_func(log_ν, AD_sel)
    
    global g_state_1 = (ν, log_ν, ∇log_ν, z, f, proposal, n, rng)

    for i in 1:length(z)
        z_proposed[i] = z[i] + sqrt(2*tau) .* rand(rng, proposal.proposal_dist, n) + tau .* ∇log_ν(z[i])
    end  

    x_proposed = f(z_proposed)

    global g_state_2 = (x_proposed)

    logd_x_proposed = BAT.checked_logdensityof(unshaped(μ)).(x_proposed)
    logd_z_proposed = log_ν.(z_proposed)

    p_accept = clamp.(exp.(logd_z_proposed-logd_z), 0, 1)

    state_x_proposed = _rebuild_density_sample_vector(last(states_x), x_proposed, logd_x_proposed)

    println("Completed mala proposal $stepno")

    return state_x_proposed, p_accept
end

function transformed_mcmc_step!!(
    iter::TransformedMCMCEnsembleIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance,
)
    @unpack  μ, f, proposal, states_x, state_z, stepno, n_accepted, context = iter
    rng = get_rng(context)
    state_x = last(states_x)
    x, logd_x = state_x.v, state_x.logd
    z, logd_z = state_z.v, state_z.logd

    state_x_proposed, state_z_proposed, p_accept = propose_mcmc(iter)

    z_proposed, logd_z_proposed = state_z_proposed.v, state_z_proposed.logd
    x_proposed, logd_x_proposed = state_x_proposed.v, state_x_proposed.logd

    tuner_new, f = tune_mcmc_transform!!(tuner, f, p_accept, z_proposed, z, stepno, context)
    
    accepted = rand(rng, length(p_accept)) .<= p_accept

    f_inv = inverse(f)
    n_walkers = length(z_proposed)
    x_new = Vector{Vector}(undef, n_walkers)
    z_new = Vector{Vector{Float64}}(undef, n_walkers)
    logd_x_new = Vector{Float64}(undef, n_walkers)
    logd_z_new = Vector{Float64}(undef, n_walkers)

    z_new_temp = nestedview(f_inv(flatview(unshaped.(x_proposed))))
    x_inv_temp = nestedview(f_inv(hcat(unshaped.(x)...)))

    accepted_neg = .! accepted 
    x_new[accepted], z_new[accepted], logd_x_new[accepted], logd_z_new[accepted] = x_proposed[accepted], z_new_temp[accepted], logd_x_proposed[accepted], logd_z_proposed[accepted]
    x_new[accepted_neg], z_new[accepted_neg], logd_x_new[accepted_neg], logd_z_new[accepted_neg] = x[accepted_neg], x_inv_temp[accepted_neg], logd_x[accepted_neg], logd_z[accepted_neg]

    state_x_new = DensitySampleVector((x_new,logd_x_new,ones(length(x_new)), fill(TransformedMCMCTransformedSampleID(iter.info.id, iter.info.cycle, iter.stepno),length(x_new)), fill(nothing,length(x_new))))

    push!(states_x, state_x_new) 

    state_z_new = _rebuild_density_sample_vector(state_z, z_new, logd_z_new)

    tempering_new, μ_new = temper_mcmc_target!!(tempering, μ, stepno)

    f_new = f

    iter.μ, iter.f, iter.states_x, iter.state_z = μ_new, f_new, states_x, state_z_new
    iter.n_accepted += Int.(accepted)
    iter.stepno += 1
    @assert iter.context === context

    return (iter, tuner_new, tempering_new)
end

g_state_3 = (;)

function transformed_mcmc_step!!(
    iter::TransformedMCMCEnsembleIterator,
    tuner::MCMCFlowTuner,
    tempering::TransformedMCMCTemperingInstance,
)
    @unpack  μ, f, states_x, stepno, context = iter
    rng = get_rng(context)
    state_x = last(states_x)
    x, logd_x = unshaped.(state_x.v), state_x.logd

    state_x_proposed, p_accept = propose_mala(iter)

    x_proposed, logd_x_proposed = state_x_proposed.v, state_x_proposed.logd

    accepted = rand(rng, length(p_accept)) .<= p_accept
        
    x_new, logd_x_new = x, logd_x
    x_new[accepted], logd_x_new[accepted] = x_proposed[accepted], logd_x_proposed[accepted]

    state_x_new = DensitySampleVector((x_new, logd_x_new, ones(length(x_new)), fill(TransformedMCMCTransformedSampleID(iter.info.id, iter.info.cycle, iter.stepno), length(x_new)), fill(nothing, length(x_new))))
    push!(states_x, state_x_new) 

    global g_state_3 = (x_new)

    tuner_new, f_new = tune_mcmc_transform!!(tuner, f, x_new, context)
    
    state_z_new = inverse(f_new)(state_x_new)

    #z_new, ladj = with_logabsdet_jacobian(inverse(flow_new), x_new)
    #state_z_new = DensitySampleVector((z_new, logd_x_new - ladj, ones(length(z_new)), fill(TransformedMCMCTransformedSampleID(iter.info.id, iter.info.cycle, iter.stepno), length(z_new)), fill(nothing, length(z_new))))

    tempering_new, μ_new = temper_mcmc_target!!(tempering, μ, stepno)

    iter.μ, iter.f, iter.state_z = μ_new, f_new, state_z_new
    iter.n_accepted += Int.(accepted)
    iter.stepno += 1
    @assert iter.context === context

    return (iter, tuner_new, tempering_new)
end


function transformed_mcmc_iterate!(
    ensemble::TransformedMCMCEnsembleIterator,
    tuner::TransformedAbstractMCMCTunerInstance,
    tempering::TransformedMCMCTemperingInstance;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func,
)
    #update with correct message
    @debug "Starting iteration over MCMC chain $(mcmc_info(ensemble).id) with $max_nsteps steps in max. $(@sprintf "%.1f seconds." max_time)"

    start_time = time()
    last_progress_message_time = start_time
    start_nsteps = nsteps(ensemble)
    start_nsteps = nsteps(ensemble)

    while (
        (nsteps(ensemble) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        transformed_mcmc_step!!(ensemble, tuner, tempering)
        callback(Val(:mcmc_step), ensemble)
  
        #TODO: output schemes

        current_time = time()
        elapsed_time = current_time - start_time
        logging_interval = 5 * round(log2(elapsed_time/60 + 1) + 1)
        if current_time - last_progress_message_time > logging_interval
            last_progress_message_time = current_time
            @debug "Iterating over MCMC chain $(mcmc_info(ensemble).id), completed $(nsteps(ensemble) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsteps(ensemble) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time) so far."
        end
    end

    current_time = time()
    elapsed_time = current_time - start_time
    @debug "Finished iteration over MCMC chain $(mcmc_info(ensemble).id), completed $(nsteps(ensemble) - start_nsteps) steps and produced $(nsteps(ensemble) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time)."
    println("Finished iteration over MCMC chain $(mcmc_info(ensemble).id), completed $(nsteps(ensemble) - start_nsteps) steps and produced $(nsteps(ensemble) - start_nsteps) samples in $(@sprintf "%.1f s" elapsed_time).")
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

function _rebuild_density_sample_vector(s::DensitySampleVector, x, logd, weight=ones(length(x)))
    @unpack info, aux = s
    DensitySampleVector((x, logd, weight, info, aux))
end
