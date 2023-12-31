# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct TransformedMCMCChainPoolInit <: TransformedMCMCInitAlgorithm

MCMC chain pool initialization strategy.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TransformedMCMCChainPoolInit <: TransformedMCMCInitAlgorithm
    init_tries_per_chain::ClosedInterval{Int64} = ClosedInterval(8, 128)
    nsteps_init::Int64 = 1000
    initval_alg::InitvalAlgorithm = InitFromTarget()
end

export TransformedMCMCChainPoolInit


function apply_trafo_to_init(trafo::Function, initalg::TransformedMCMCChainPoolInit)
    TransformedMCMCChainPoolInit(
    initalg.init_tries_per_chain,
    initalg.nsteps_init,
    apply_trafo_to_init(trafo, initalg.initval_alg)
    )
end



function _construct_chain(
    rngpart::RNGPartition,
    id::Integer,
    algorithm::TransformedMCMCSampling,
    density::AbstractMeasureOrDensity,
    initval_alg::InitvalAlgorithm,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = bat_initval(density, initval_alg, new_context).result
    return TransformedMCMCIterator(algorithm, density, id, v_init, new_context)
end

_gen_chains(
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    algorithm::TransformedMCMCSampling,
    density::AbstractMeasureOrDensity,
    initval_alg::InitvalAlgorithm,
    context::BATContext
) = [_construct_chain(rngpart, id, algorithm, density, initval_alg, context) for id in ids]

function mcmc_init!(
    algorithm::TransformedMCMCSampling,
    density::AbstractMeasureOrDensity,
    nchains::Integer,
    init_alg::TransformedMCMCChainPoolInit,
    tuning_alg::TransformedMCMCTuningAlgorithm, # TODO: part of algorithm? # MCMCTuner
    nonzero_weights::Bool,
    callback::Function,
    context::BATContext
)
    @info "TransformedMCMCChainPoolInit: trying to generate $nchains viable MCMC chain(s)."

    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_alg.init_tries_per_chain) * nchains

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC chain to determine chain, output and tuner types." #TODO: remove!

    dummy_context = deepcopy(context)
    dummy_initval = unshaped(bat_initval(density, InitFromTarget(), dummy_context).result, varshape(density))
    dummy_chain = TransformedMCMCIterator(algorithm, density, 1, dummy_initval, dummy_context) 
    dummy_tuner = get_tuner(tuning_alg, dummy_chain)
    dummy_temperer = get_temperer(algorithm.tempering, density)

    chains = similar([dummy_chain], 0)
    tuners = similar([dummy_tuner], 0)
    temperers = similar([dummy_temperer], 0)

    init_tries::Int = 1

    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        viable_tuners = similar(tuners, 0)
        viable_chains = similar(chains, 0)
        viable_temperers = similar(temperers, 0)
        viable_outputs = [] #similar(outputs, 0) #TODO

        # as the iteration after viable check is more costly, fill up to be at least capable to skip a complete reiteration.
        while length(viable_tuners) < min_nviable-length(tuners) && ncandidates < max_ncandidates
            n = min(min_nviable, max_ncandidates - ncandidates)
            @debug "Generating $n $(init_tries > 1 ? "additional " : "")candidate MCMC chain(s)."

            new_chains = _gen_chains(rngpart, ncandidates .+ (one(Int64):n), algorithm, density, initval_alg, context)

            filter!(isvalidchain, new_chains)

            new_tuners = get_tuner.(Ref(tuning_alg), new_chains)
            new_temperers = fill(get_temperer(algorithm.tempering, density), size(new_tuners,1))

            next_cycle!.(new_chains)

            tuning_init!.(new_tuners, new_chains, init_alg.nsteps_init)
            ncandidates += n

            @debug "Testing $(length(new_chains)) candidate MCMC chain(s)."

            transformed_mcmc_iterate!(
                new_chains, new_tuners, new_temperers#,
               # max_nsteps = clamp(div(init_alg.nsteps_init, 5), 10, 50),
                #callback = callback,
               # nonzero_weights = nonzero_weights
            )

            # testing if chains are viable:
            viable_idxs = findall(isviablechain.(new_chains))

            new_outputs = getproperty.(new_chains, :samples) #TODO ?

            append!(viable_tuners, new_tuners[viable_idxs])
            append!(viable_chains, new_chains[viable_idxs])
            append!(viable_outputs, new_outputs[viable_idxs])
            append!(viable_temperers, new_temperers[viable_idxs])

        end

        @debug "Found $(length(viable_tuners)) viable MCMC chain(s)."

        if !isempty(viable_chains)
            desc_string = string("Init try ", init_tries, " for nvalid=", length(viable_tuners), " of min_nviable=", length(tuners), "/", min_nviable )
            progress_meter = ProgressMeter.Progress(length(viable_tuners) * init_alg.nsteps_init, desc=desc_string, barlen=80-length(desc_string), dt=0.1)
            transformed_mcmc_iterate!(
                viable_chains, viable_tuners, viable_temperers;
                max_nsteps = init_alg.nsteps_init,
                callback = (kwargs...)-> let pm=progress_meter; ProgressMeter.next!(pm) ; end,
                nonzero_weights = nonzero_weights
            )
            ProgressMeter.finish!(progress_meter)
            nsamples_thresh = floor(Int, 0.8 * median([nsamples(chain) for chain in viable_chains]))
            good_idxs = findall(chain -> nsamples(chain) >= nsamples_thresh, viable_chains)
            @debug "Found $(length(viable_chains)) MCMC chain(s) with at least $(nsamples_thresh) unique accepted samples."

            append!(chains, view(viable_chains, good_idxs))
            append!(tuners, view(viable_tuners, good_idxs))
            append!(temperers, view(viable_temperers, good_idxs))
        end

        init_tries += 1
    end

    outputs = getproperty.(chains, :samples)

    length(chains) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")

    m = nchains
    tidxs = LinearIndices(chains)
    n = length(tidxs)

    modes = hcat(broadcast(samples -> Array(bat_findmode(samples, MaxDensitySearch(), context).result), outputs)...)

    final_chains = similar(chains, 0)
    final_tuners = similar(tuners, 0)
    final_temperers = similar(temperers, 0)
    final_outputs = similar(outputs, 0)

    # TODO: should we put this into a function?
    if 2 <= m < size(modes, 2)
        clusters = kmeans(modes, m, init = KmCentralityAlg())
        clusters.converged || error("k-means clustering of MCMC chains did not converge")

        mincosts = fill(Inf, m)
        chain_sel_idxs = fill(0, m)

        for i in tidxs
            j = clusters.assignments[i]
            if clusters.costs[i] < mincosts[j]
                mincosts[j] = clusters.costs[i]
                chain_sel_idxs[j] = i
            end
        end

        @assert all(j -> j in tidxs, chain_sel_idxs)

        for i in sort(chain_sel_idxs)
            push!(final_chains, chains[i])
            push!(final_tuners, tuners[i])
            push!(final_temperers, temperers[i])
            push!(final_outputs, outputs[i])
        end
    elseif m == 1
        i = findmax(nsamples.(chains))[2]
        push!(final_chains, chains[i])
        push!(final_tuners, tuners[i])
        push!(final_temperers, temperers[i])
        push!(final_outputs, outputs[i])
    else
        @assert length(chains) == nchains
        resize!(final_chains, nchains)
        copyto!(final_chains, chains)

        @assert length(tuners) == nchains
        resize!(final_tuners, nchains)
        copyto!(final_tuners, tuners)

        @assert length(temperers) == nchains
        resize!(final_temperers, nchains)
        copyto!(final_temperers, temperers)

        @assert length(outputs) == nchains
        resize!(final_outputs, nchains)
        copyto!(final_outputs, outputs)
    end

    @info "Selected $(length(final_chains)) MCMC chain(s)."
    #tuning_postinit!.(final_tuners, final_chains, final_outputs) #TODO: implement

    (chains = final_chains, tuners = final_tuners, temperers = final_temperers, outputs = final_outputs)
end
