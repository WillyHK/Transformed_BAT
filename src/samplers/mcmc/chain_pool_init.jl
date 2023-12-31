# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MCMCChainPoolInit <: MCMCInitAlgorithm

MCMC chain pool initialization strategy.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCChainPoolInit <: MCMCInitAlgorithm
    init_tries_per_chain::ClosedInterval{Int64} = ClosedInterval(8, 128)
    nsteps_init::Int64 = 1000
    initval_alg::InitvalAlgorithm = InitFromTarget()
end

export MCMCChainPoolInit


function apply_trafo_to_init(trafo::Function, initalg::MCMCChainPoolInit)
    MCMCChainPoolInit(
    initalg.init_tries_per_chain,
    initalg.nsteps_init,
    apply_trafo_to_init(trafo, initalg.initval_alg)
    )
end



function _construct_chain(
    rngpart::RNGPartition,
    id::Integer,
    algorithm::MCMCAlgorithm,
    density::AbstractMeasureOrDensity,
    initval_alg::InitvalAlgorithm,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = bat_initval(density, initval_alg, new_context).result
    return MCMCIterator(algorithm, density, id, v_init, new_context)
end

_gen_chains(
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    algorithm::MCMCAlgorithm,
    density::AbstractMeasureOrDensity,
    initval_alg::InitvalAlgorithm,
    context::BATContext
) = [_construct_chain(rngpart, id, algorithm, density, initval_alg, context) for id in ids]

# TODO AC discuss
function _cluster_selection(
    chains::AbstractVector{<:MCMCIterator},
    tuners,
    outputs::AbstractVector{<:DensitySampleVector},
    scale::Real=3,
    decision_range_skip::Real=0.9,
)
    logds_by_chain = [view(s.logd,(floor(Int,decision_range_skip*length(s))):length(s)) for s in outputs]
    medians = [median(x) for x in logds_by_chain]
    stddevs = [std(x) for x in logds_by_chain]

    # yet uncategoriesed
    uncat = eachindex(chains, tuners, outputs, logds_by_chain, stddevs, medians)

    # clustered indices
    cidxs = Vector{Vector{eltype(uncat)}}()
    # categories all to clusters
    while length(uncat) > 0
        idxmin = findmin(view(stddevs,uncat))[2]

        cidx_sel = map(means_remaining_uncat -> abs(means_remaining_uncat-medians[uncat[idxmin]]) < scale*stddevs[uncat[idxmin]], view(medians,uncat))

        push!(cidxs, uncat[cidx_sel])
        uncat = uncat[.!cidx_sel]
    end
    medians_c = [ median(reduce(vcat, view(logds_by_chain, ids))) for ids in cidxs]
    idx_order = sortperm(medians_c, rev=true)

    chains_by_cluster = [ view(chains, ids) for ids in cidxs[idx_order]]
    tuners_by_cluster = [ view(tuners, ids) for ids in cidxs[idx_order]]
    outputs_by_cluster = [ view(outputs, ids) for ids in cidxs[idx_order]]
    ( chains = chains_by_cluster[1], tuners = tuners_by_cluster[1], outputs = outputs_by_cluster[1], )
end


function mcmc_init!(
    algorithm::MCMCAlgorithm,
    density::AbstractMeasureOrDensity,
    nchains::Integer,
    init_alg::MCMCChainPoolInit,
    tuning_alg::MCMCTuningAlgorithm,
    nonzero_weights::Bool,
    callback::Function,
    context::BATContext
)
    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_alg.init_tries_per_chain) * nchains

    @info "MCMCChainPoolInit: trying to generate $(min_nviable) viable MCMC chain(s)."

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC chain to determine chain, output and tuner types."

    dummy_context = deepcopy(context)
    dummy_initval = unshaped(bat_initval(density, InitFromTarget(), dummy_context).result, varshape(density))
    dummy_chain = MCMCIterator(algorithm, density, 1, dummy_initval, dummy_context)
    dummy_tuner = tuning_alg(dummy_chain)

    chains = similar([dummy_chain], 0)
    tuners = similar([dummy_tuner], 0)
    outputs = similar([DensitySampleVector(dummy_chain)], 0)
    init_tries::Int = 1

    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        viable_tuners = similar(tuners, 0)
        viable_chains = similar(chains, 0)
        viable_outputs = similar(outputs, 0)

        # as the iteration after viable check is more costly, fill up to be at least capable to skip a complete reiteration.
        while length(viable_tuners) < min_nviable-length(tuners) && ncandidates < max_ncandidates
            n = min(min_nviable, max_ncandidates - ncandidates)
            @debug "Generating $n $(init_tries > 1 ? "additional " : "")candidate MCMC chain(s)."

            new_chains = _gen_chains(rngpart, ncandidates .+ (one(Int64):n), algorithm, density, initval_alg, context)

            filter!(isvalidchain, new_chains)

            new_tuners = tuning_alg.(new_chains)
            new_outputs = DensitySampleVector.(new_chains)
            next_cycle!.(new_chains)
            tuning_init!.(new_tuners, new_chains, init_alg.nsteps_init)
            ncandidates += n

            @debug "Testing $(length(new_tuners)) candidate MCMC chain(s)."

            mcmc_iterate!(
                new_outputs, new_chains, new_tuners;
                max_nsteps = clamp(div(init_alg.nsteps_init, 5), 10, 50),
                callback = callback,
                nonzero_weights = nonzero_weights
            )
            @info length.(new_outputs)
            
            viable_idxs = findall(isviablechain.(new_chains))

            append!(viable_tuners, new_tuners[viable_idxs])
            append!(viable_chains, new_chains[viable_idxs])
            append!(viable_outputs, new_outputs[viable_idxs])
        end

        @debug "Found $(length(viable_tuners)) viable MCMC chain(s)."

        if !isempty(viable_tuners)
            desc_string = string("Init try ", init_tries, " for nvalid=", length(viable_tuners), " of min_nviable=", length(tuners), "/", min_nviable )
            progress_meter = ProgressMeter.Progress(length(viable_tuners) * init_alg.nsteps_init, desc=desc_string, barlen=80-length(desc_string), dt=0.1)

            mcmc_iterate!(
                viable_outputs, viable_chains, viable_tuners;
                max_nsteps = init_alg.nsteps_init,
                callback = (kwargs...)-> let pm=progress_meter, callback=callback ; callback(kwargs) ; ProgressMeter.next!(pm) ; end,
                nonzero_weights = nonzero_weights
            )

            ProgressMeter.finish!(progress_meter)

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(chain) for chain in viable_chains]))
            good_idxs = findall(chain -> nsamples(chain) >= nsamples_thresh, viable_chains)
            @debug "Found $(length(viable_tuners)) MCMC chain(s) with at least $(nsamples_thresh) unique accepted samples."

            append!(chains, view(viable_chains, good_idxs))
            append!(tuners, view(viable_tuners, good_idxs))
            append!(outputs, view(viable_outputs, good_idxs))
        end

        init_tries += 1
    end

    # TODO AC
    if true
        @unpack chains, tuners, outputs = _cluster_selection(chains, tuners, outputs)
        length(tuners) < nchains && error("Failed to generate $nchains viable MCMC chains")
    else
        length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")
    end


    m = nchains
    tidxs = LinearIndices(tuners)
    n = length(tidxs)

    modes = hcat(broadcast(samples -> Array(bat_findmode(samples, MaxDensitySearch(), context).result), outputs)...)

    final_chains = similar(chains, 0)
    final_tuners = similar(tuners, 0)
    final_outputs = similar(outputs, 0)

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
            push!(final_outputs, outputs[i])
        end
    elseif m == 1
        i = findmax(nsamples.(chains))[2]
        push!(final_chains, chains[i])
        push!(final_tuners, tuners[i])
        push!(final_outputs, outputs[i])
    else
        @assert length(chains) == nchains
        resize!(final_chains, nchains)
        copyto!(final_chains, chains)

        @assert length(tuners) == nchains
        resize!(final_tuners, nchains)
        copyto!(final_tuners, tuners)

        @assert length(outputs) == nchains
        resize!(final_outputs, nchains)
        copyto!(final_outputs, outputs)
    end

    @info "Selected $(length(final_tuners)) MCMC chain(s)."
    tuning_postinit!.(final_tuners, final_chains, final_outputs)

    (chains = final_chains, tuners = final_tuners, outputs = final_outputs)
end
