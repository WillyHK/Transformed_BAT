# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct TransformedMCMCEnsemblePoolInit <: TransformedMCMCInitAlgorithm

MCMC ensemble pool initialization strategy.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TransformedMCMCEnsemblePoolInit <: TransformedMCMCInitAlgorithm
    init_tries_per_ensemble::ClosedInterval{Int64} = ClosedInterval(1, 128) # vorher beginn bei 8 whyever
    nsteps_init::Int64 = 1000
    initval_alg::InitvalAlgorithm = InitFromTarget()
end

export TransformedMCMCEnsemblePoolInit


function apply_trafo_to_init(trafo::Function, initalg::TransformedMCMCEnsemblePoolInit)
    TransformedMCMCEnsemblePoolInit(
    initalg.init_tries_per_ensemble,
    initalg.nsteps_init,
    apply_trafo_to_init(trafo, initalg.initval_alg)
    )
end


function _construct_ensemble(
    rngpart::RNGPartition,
    id::Integer,
    algorithm::TransformedMCMCSampling,
    density::AbstractMeasureOrDensity,
    initval_alg::InitvalAlgorithm,
    nwalker::Integer,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = nestedview(ElasticArray{Float64}(undef, totalndof(density), 0))
    
    for i in 1:nwalker
        push!(v_init, bat_initval(density, initval_alg, new_context).result)
    end

    return TransformedMCMCEnsembleIterator(algorithm, density, Int32(id), v_init, new_context)
end

_gen_ensembles(
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    algorithm::TransformedMCMCSampling,
    density::AbstractMeasureOrDensity,
    initval_alg::InitvalAlgorithm,
    nwalker::Int,
    context::BATContext
) = [_construct_ensemble(rngpart, id, algorithm, density, initval_alg, nwalker,context) for id in ids]

g_state = (;)
function mcmc_init!(
    algorithm::TransformedMCMCSampling,
    density::AbstractMeasureOrDensity,
    nensembles::Integer,
    init_alg::TransformedMCMCEnsemblePoolInit,
    tuning_alg::TransformedMCMCTuningAlgorithm, # TODO: part of algorithm? # MCMCTuner
    nonzero_weights::Bool,
    callback::Function,
    context::BATContext
)
    @info "TransformedMCMCEnsemblePoolInit: trying to generate $nensembles viable MCMC states_x(s)."

    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_ensemble) * nensembles
    max_ncandidates::Int = maximum(init_alg.init_tries_per_ensemble) * nensembles

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC ensemble to determine ensemble, output and tuner types." #TODO: remove!

    dummy_context = deepcopy(context)
    dummy_init_state = nestedview(ElasticArray{Float64}(undef, totalndof(density), 0))
    push!(dummy_init_state, bat_initval(density, InitFromTarget(), dummy_context).result)

    dummy_ensemble = TransformedMCMCEnsembleIterator(algorithm, density, Int32(1), dummy_init_state, dummy_context) 
    dummy_tuner = get_tuner(tuning_alg, dummy_ensemble)
    dummy_temperer = get_temperer(algorithm.tempering, density)

    states_x = similar([dummy_ensemble], 0)
    tuners = similar([dummy_tuner], 0)
    temperers = similar([dummy_temperer], 0)


    transformed_mcmc_iterate!(
        states_x, tuners, temperers
    )

    outputs = []


    init_tries::Int = 1
    nwalker::Int = 100

    #while length(tuners) < min_nviable && ncandidates < max_ncandidates
        #viable_tuners = similar(tuners, 0)
        #viable_ensembles = []#similar(states_x, 0) #TODO
        #viable_temperers = similar(temperers, 0)
        #viable_outputs = [] #similar(outputs, 0) #TODO

        # as the iteration after viable check is more costly, fill up to be at least capable to skip a complete reiteration.
        #while length(viable_tuners) < min_nviable-length(tuners) && ncandidates < max_ncandidates
            n = min(min_nviable, max_ncandidates - ncandidates)
            @debug "Generating $n $(init_tries > 1 ? "additional " : "")candidate MCMC ensemble(s)."

            new_ensembles = _gen_ensembles(rngpart, ncandidates .+ (one(Int64):n), algorithm, density, initval_alg,nwalker*10,context)
            #filter!(isvalidensemble, new_ensembles)

            new_tuners = get_tuner.(Ref(TransformedMCMCNoOpTuning()),new_ensembles) # NoOpTuner for BurnIn
            new_temperers = fill(get_temperer(algorithm.tempering, density), size(new_tuners,1))

            next_cycle!.(new_ensembles)

            tuning_init!.(new_tuners, new_ensembles, init_alg.nsteps_init)
            ncandidates += n

            @debug "Testing $(length(new_ensembles)) candidate MCMC ensemble(s)."

            nteststeps = 100
            for i in 1:nteststeps
                transformed_mcmc_iterate!(
                    new_ensembles, new_tuners, new_temperers
                )
            end

            for ensemble in new_ensembles
                mask = (ensemble.n_accepted .!= 0)
                viable_walker = ensemble.states_x[end][mask]
                choosen = rand(1:length(viable_walker),nwalker)

                ensemble.states_x = [viable_walker[choosen]]
                ensemble.state_z = ensemble.state_z[mask][choosen]
                ensemble.n_accepted = zeros(Int64,nwalker)
                if(length(viable_walker)<nwalker)
                    error("Found not enough good walker!")
                end
            end

            new_tuners = get_tuner.(Ref(tuning_alg), new_ensembles)
            new_outputs = getproperty.(new_ensembles, :states_x)             

        return (chains = new_ensembles, tuners = new_tuners, temperers = new_temperers, outputs = new_outputs)
        
#        # testing if states_x are viable:
#            viable_idxs = findall(isviablechain.(new_ensembles))
#
#            new_outputs = getproperty.(new_ensembles, :states_x) #TODO ?
#            println(typeof(new_outputs))
#
#            append!(viable_tuners, new_tuners[viable_idxs])
#            #append!(viable_ensembles, new_ensembles[viable_idxs])
#            viable_ensembles = new_ensembles[viable_idxs]
#            append!(viable_outputs, new_outputs[viable_idxs])
#            append!(viable_temperers, new_temperers[viable_idxs])
#        #end
#
#
#
#
#        @debug "Found $(length(viable_tuners)) viable MCMC ensemble(s)."
#
#        if !isempty(viable_ensembles)
#            desc_string = string("Init try ", init_tries, " for nvalid=", length(viable_tuners), " of min_nviable=", length(tuners), "/", min_nviable )
#            progress_meter = ProgressMeter.Progress(length(viable_tuners) * init_alg.nsteps_init, desc=desc_string, barlen=80-length(desc_string), dt=0.1)
#            global g_state = (viable_ensembles, viable_tuners, viable_temperers)
#
#            transformed_mcmc_iterate!(
#                viable_ensembles, viable_tuners, viable_temperers;
#                max_nsteps = init_alg.nsteps_init,
#                callback = (kwargs...)-> let pm=progress_meter; ProgressMeter.next!(pm) ; end,
#                nonzero_weights = nonzero_weights
#            )
#            ProgressMeter.finish!(progress_meter)
#            nsamples_thresh = floor(Int, 0.8 * median([nsamples(ensemble) for ensemble in viable_ensembles]))
#            good_idxs = findall(ensemble -> nsamples(ensemble) >= nsamples_thresh, viable_ensembles)
#            @debug "Found $(length(viable_ensembles)) MCMC ensemble(s) with at least $(nsamples_thresh) unique accepted samples."
#
#            global g_state = (states_x,view(viable_ensembles, good_idxs))
#            #append!(states_x, view(viable_ensembles, good_idxs))
#            states_x = viable_ensembles
#            append!(tuners, view(viable_tuners, good_idxs))
#            append!(temperers, view(viable_temperers, good_idxs))
#        end
#
#        init_tries += 1
#    #end
#
#    outputs = getproperty.(states_x, :states_x)
#
#    length(states_x) < min_nviable && error("Failed to generate $min_nviable viable MCMC states_x")
#
#    m = nensembles
#    tidxs = LinearIndices(states_x)
#    n = length(tidxs)
#
#    # modes = hcat(broadcast(samples -> Array(bat_findmode(samples, MaxDensitySearch(), context).result), outputs)...)
#
#    final_ensembles = states_x
#    final_tuners =tuners
#    final_temperers =temperers
#
#    out::Vector{DensitySampleVector} = []
#    for i in 1:length(outputs)
#        x = outputs[i]
#        samples = x[1]
#        #for j in 2:length(x)
#        #    append!(samples,x[j])
#        #end
#        push!(out,samples)
#    end
#    final_outputs=out
    #final_ensembles = similar(states_x, 0)
    #final_tuners = similar(tuners, 0)
    #final_temperers = similar(temperers, 0)
    #final_outputs = similar(outputs, 0)

#    # TODO: should we put this into a function?
#    if 2 <= m < size(modes, 2)
#        clusters = kmeans(modes, m, init = KmCentralityAlg())
#        clusters.converged || error("k-means clustering of MCMC states_x did not converge")
#
#        mincosts = fill(Inf, m)
#        ensemble_sel_idxs = fill(0, m)
#
#        for i in tidxs
#            j = clusters.assignments[i]
#            if clusters.costs[i] < mincosts[j]
#                mincosts[j] = clusters.costs[i]
#                ensemble_sel_idxs[j] = i
#            end
#        end
#
#        @assert all(j -> j in tidxs, ensemble_sel_idxs)
#
#        for i in sort(ensemble_sel_idxs)
#            push!(final_ensembles, states_x[i])
#            push!(final_tuners, tuners[i])
#            push!(final_temperers, temperers[i])
#            push!(final_outputs, outputs[i])
#        end
#    elseif m == 1
#        i = findmax(nsamples.(states_x))[2]
#        push!(final_ensembles, states_x[i])
#        push!(final_tuners, tuners[i])
#        push!(final_temperers, temperers[i])
#        push!(final_outputs, outputs[i])
#    else
#        @assert length(states_x) == nensembles
#        resize!(final_ensembles, nensembles)
#        copyto!(final_ensembles, states_x)
#
#        @assert length(tuners) == nensembles
#        resize!(final_tuners, nensembles)
#        copyto!(final_tuners, tuners)
#
#        @assert length(temperers) == nensembles
#        resize!(final_temperers, nensembles)
#        copyto!(final_temperers, temperers)
#
#        @assert length(outputs) == nensembles
#        resize!(final_outputs, nensembles)
#        copyto!(final_outputs, outputs)
#    end

    #@info "Selected $(length(final_ensembles)) MCMC ensemble(s)."
    #tuning_postinit!.(final_tuners, final_ensembles, final_outputs) #TODO: implement

    #(chains = final_ensembles, tuners = final_tuners, temperers = final_temperers, outputs = final_outputs)
end
