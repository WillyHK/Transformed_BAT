# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function transformed_check_convergence!(
    chains::AbstractVector{<:MCMCIterator},
    samples::AbstractVector{<:DensitySampleVector},
    algorithm::ConvergenceTest,
    context::BATContext
)
    result = convert(Bool, bat_convergence(samples, algorithm, context).result)
    for chain in chains
        chain.info = TransformedMCMCIteratorInfo(chain.info, converged = result)
    end
    result
end



"""
    gr_Rsqr(stats::AbstractVector{<:TransformedMCMCBasicStats})
    gr_Rsqr(samples::AbstractVector{<:DensitySampleVector})

*BAT-internal, not part of stable public API.*

Gelman-Rubin ``\$R^2\$`` for all DOF.
"""
# TODO AC: reactivate
# function gr_Rsqr end

function gr_Rsqr(stats::AbstractVector{<:TransformedMCMCBasicStats})
    m = totalndof(first(stats))
    W = mean([cs.param_stats.cov[i,i] for cs in stats, i in 1:m], dims=1)[:]
    B = var([cs.param_stats.mean[i] for cs in stats, i in 1:m], dims=1)[:]
    (W .+ B) ./ W
end

#TODO AC: reactivate
# function gr_Rsqr(samples::AbstractVector{<:DensitySampleVector})
#     gr_Rsqr(TransformedMCMCBasicStats.(samples))
# end



"""
    struct TransformedGelmanRubinConvergence <: ConvergenceTest

Gelman-Rubin maximum R^2 convergence test.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TransformedGelmanRubinConvergence <: ConvergenceTest
    threshold::Float64 = 1.1
end

export TransformedGelmanRubinConvergence

function bat_convergence_impl(samples::AbstractVector{<:DensitySampleVector}, algorithm::TransformedGelmanRubinConvergence, ::BATContext)
    max_Rsqr = maximum(gr_Rsqr(samples))
    vt = ValueAndThreshold{max_Rsqr}(max_Rsqr, <=, algorithm.threshold)
    converged = convert(Bool, vt)
    @debug begin
        success_str = converged ? "have" : "have *not*"
        "Chains $success_str converged, max(R^2) = $(vt.value), threshold = $(vt.threshold)"
    end
    (result = vt,)
end



@doc doc"""
    bg_R_2sqr(stats::AbstractVector{<:TransformedMCMCBasicStats}; corrected::Bool = false)
    bg_R_2sqr(samples::AbstractVector{<:DensitySampleVector}; corrected::Bool = false)

*BAT-internal, not part of stable public API.*

Brooks-Gelman R_2^2 for all DOF.
If normality is assumed, 'corrected' should be set to true to account for the sampling variability.
"""
function bg_R_2sqr(stats::AbstractVector{<:TransformedMCMCBasicStats}; corrected::Bool = false)
    p = totalndof(first(stats))
    m = length(stats)
    n = mean(Float64.(nsamples.(stats)))

    σ_W = var([cs.param_stats.cov[i,i] for cs in stats, i in 1:p], dims = 1)[:]
    B  = var([cs.param_stats.mean[i] for cs in stats, i in 1:p], dims = 1)[:]
    W = mean([cs.param_stats.cov[i,i] for cs in stats, i in 1:p], dims = 1)[:]

    σ_sq = m * (n - 1) / (m*n - 1) * W + n * (m - 1) / (m*n - 1) * B

    R_unc = σ_sq ./ W

    if corrected == false
        return R_unc
    end

    σ_ij = [cs.param_stats.cov[i,i] for cs in stats, i in 1:p]
    x_ij = [cs.param_stats.mean[i] for cs in stats, i in 1:p]

    cov_σx = [cov(σ_ij[:,j], x_ij[:,j]) for j in 1:p]
    cov_σx_sq = [cov(σ_ij[:,j], x_ij[:,j].^2) for j in 1:p]

    N = (n-1)/n
    M = (m-1)/m
    V = N*σ_sq + M*B

    σ_V = N^2/m*σ_W + 2*M/(m-1)*B.^2 + 2*M*N/m*(cov_σx_sq - 2*B.*cov_σx)
    d = 2 * V.^2 ./ σ_V

    R_unc.*(d.+3)./(d.+1)
end

# TODO AC: reactivate
# function bg_R_2sqr(samples::AbstractVector{<:DensitySampleVector}; corrected::Bool = false)
#     bg_R_2sqr(TransformedMCMCBasicStats.(samples), corrected = corrected)
# end



"""
    struct TransformedBrooksGelmanConvergence <: ConvergenceTest

Brooks-Gelman maximum R^2 convergence test.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TransformedBrooksGelmanConvergence <: ConvergenceTest
    threshold::Float64 = 1.1
    corrected::Bool = false
end

export TransformedBrooksGelmanConvergence

function bat_convergence_impl(samples::AbstractVector{<:DensitySampleVector}, algorithm::TransformedBrooksGelmanConvergence, ::BATContext)
    max_Rsqr = maximum(bg_R_2sqr(samples, corrected = algorithm.corrected))
    vt = ValueAndThreshold{max_Rsqr}(max_Rsqr, <=, algorithm.threshold)
    converged = convert(Bool, vt)
    @debug begin
        success_str = converged ? "have" : "have *not*"
        "Chains $success_str converged, max(R^2) = $(vt.value), threshold = $(vt.threshold)"
    end
    (result = vt,)
end



function bat_convergence_impl(samples::DensitySampleVector, algorithm::Union{TransformedGelmanRubinConvergence, TransformedBrooksGelmanConvergence}, context::BATContext)
    # create a vector of chains
    chains_ind = unique([i.chainid for i in samples.info])
    vector_chains = DensitySampleVector[]
    # ToDo: Improve implementation
    for i in chains_ind
        mask_chain = [j.chainid == i for j in samples.info]
        push!(vector_chains, samples[mask_chain])
    end

    bat_convergence_impl(vector_chains, algorithm, context)
end
