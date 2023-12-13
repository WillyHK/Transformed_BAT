using Pkg
Pkg.activate(".")

using Revise
using Plots
using AdaptiveFlows
using BAT

BAT._Plots_backend(args...; kwargs...) = Plots.backend(args...; kwargs...)
BAT._Plots_cgrad(args...; kwargs...) = Plots.cgrad(args...; kwargs...)
BAT._Plots_grid(args...; kwargs...) = Plots.grid(args...; kwargs...)
BAT._Plots_Shape(args...; kwargs...) = Plots.Shape(args...; kwargs...)
BAT._Plots_Surface(args...; kwargs...) = Plots.Surface(args...; kwargs...)

BAT._Plots_backend_is_pyplot() = Plots.backend() isa Plots.PyPlotBackend

using BAT.MeasureBase
using AffineMaps
using ChangesOfVariables
using BAT.LinearAlgebra
using BAT.Distributions
using BAT.InverseFunctions
import BAT: TransformedMCMCIterator, TransformedAdaptiveMHTuning, TransformedRAMTuner, TransformedMHProposal, TransformedNoTransformedMCMCTempering, transformed_mcmc_step!!, TransformedMCMCTransformedSampleID
using Random123
using PositiveFactorizations
using AutoDiffOperators

import BAT: mcmc_iterate!, transformed_mcmc_iterate!, TransformedMCMCSampling, plot, tuning_init!

using Serialization
using Distributions
using KernelAbstractions
using Flux
import BAT: CustomTransform, TransformedMCMCNoOpTuner
using InverseFunctions
using ArraysOfArrays
using ValueShapes

include("../../examples/ExamplePosterior.jl")
posterior = get_kamm(3)
#posterior = BAT.example_posterior()

####################################################################
# Sampling without ensembles and flow
####################################################################
context = BATContext(ad = ADModule(:ForwardDiff))
x = @time BAT.bat_sample(posterior, 
        TransformedMCMCSampling(
            pre_transform=PriorToGaussian(), 
            init=TransformedMCMCChainPoolInit(),
            tuning_alg=TransformedMCMCNoOpTuning(), 
            nchains=4, nsteps=500),
        context).result; # @TODO: Why are there so many samples?
plot(x,bins=200)
println(sum(x.weight))
println(length(x.v)/sum(x.weight))

samples::Matrix{Float32} = flatview(unshaped.(x.v))

####################################################################
# Ensembles without flow
####################################################################
function EnsembleSampling(posterior, f,mala=true)
    y = @time BAT.bat_sample_impl(posterior, 
        TransformedMCMCSampling(
            pre_transform=PriorToGaussian(), 
            init=TransformedMCMCEnsemblePoolInit(),
            tuning_alg=TransformedMCMCNoOpTuning(), 
            adaptive_transform=f, use_mala=mala,
            nchains=4, nsteps=2500),
        context).result;
    println(length(y.v))
    println(length(unique(y.v))/length(y.v))
    plot(y,bins=200)
end

density_notrafo = convert(BAT.AbstractMeasureOrDensity, posterior)
densit, trafo = BAT.transform_and_unshape(PriorToGaussian(), density_notrafo, context)

s = cholesky(Positive, BAT._approx_cov(densit)).L
mul = BAT.CustomTransform(Mul(s))

EnsembleSampling(posterior,mul,false)

####################################################################
# Flow trained to identity
####################################################################

d = length(x.v[1])
n = bat_sample(MvNormal(zeros(d),I(d))).result
normal::Matrix{Float32} = flatview(unshaped.(n.v))
flow_n = build_flow(normal)
flow_n = AdaptiveFlows.optimize_flow_sequentially(normal, flow_n, Adam(1f-3)).result

function flat2batsamples(smpls_flat)
    n = length(smpls_flat[:,1])
    smpls = [smpls_flat[i,1:end] for i in 1:n]
    weights = ones(length(smpls))
    logvals = zeros(length(smpls))
    return BAT.DensitySampleVector(smpls, logvals, weight = weights)
end

plot(flat2batsamples(normal'))
plot(flat2batsamples(flow_n(normal)'))

f = BAT.CustomTransform(flow_n)

####################################################################
# Test the MALA-Step without tuning
####################################################################
EnsembleSampling(posterior,f,false) # MC prop.
EnsembleSampling(posterior,f,true)

####################################################################
# Test the FlowTuner
####################################################################
zx = @time BAT.bat_sample_impl(posterior, 
        TransformedMCMCSampling(
            pre_transform=PriorToGaussian(), 
            init=TransformedMCMCEnsemblePoolInit(),
            tuning_alg=MCMCFlowTuning(), 
            adaptive_transform=f,
            nchains=4, nsteps=2500),
        context)
z=zx.result 
plot(z)
println(length(unique(z.v))/length(z.v))

####################################################################
# Well trained flow
####################################################################
flow=build_flow(normal) #samples)  # ATTENTION: There are big Problems if there is some ScaleShifting in the Flow!
samples = samples ./ std(samples)
flow = AdaptiveFlows.optimize_flow_sequentially(samples, flow, Adam(1f-3)).result

plot(flat2batsamples(samples'))
plot(flat2batsamples(flow(samples)'))

f = BAT.CustomTransform(flow)

###############################
# Old code below this
###############################

samp = inverse(flow)(normal)
plot(flat2batsamples(samp'))

using BenchmarkTools
logd_z = @btime logdensityof(MeasureBase.pullback(g, μ),z)





g= BAT.CustomTransform(Mul(s))
x = @btime MeasureBase.pullback(g, posterior)
l = @btime logdensityof(x);
logd_z = @btime l(z);

z2 = rand(posterior.prior);
logd_z = logdensityof(posterior)(z2)

l = logdensityof(MeasureBase.pullback(g, μ));

function myFunction(l, z)
    for i in 1:100
        l(z)
    end
end

@profview myFunction(l, z)

μ(z)

gg = inverse(g)



context = BATContext(ad = ADModule(:ForwardDiff))

#posterior = BAT.example_posterior()

my_result = @time BAT.bat_sample_impl(posterior, TransformedMCMCSampling(pre_transform=PriorToGaussian(), nchains=4, nsteps=4*100000), context)



my_result = @time BAT.bat_sample_impl(posterior, TransformedMCMCSampling(pre_transform=PriorToGaussian(), tuning_alg=TransformedAdaptiveMHTuning(), nchains=4, nsteps=4*100000, adaptive_transform=f), context)

my_samples = my_result.result



using Plots
plot(my_samples)

r_mh = @time BAT.bat_sample_impl(posterior, MCMCSampling( nchains=4, nsteps=4*100000, store_burnin=true), context)

r_hmc = @time BAT.bat_sample_impl(posterior, MCMCSampling(mcalg=HamiltonianMC(), nchains=4, nsteps=4*20000), context)
 
plot(bat_sample(posterior).result)

using BAT.Distributions
using BAT.ValueShapes
prior2 = NamedTupleDist(ShapedAsNT,
    b = [4.2, 3.3],
    a = Exponential(1.0),
    c = Normal(1.0,3.0),
    d = product_distribution(Weibull.(ones(2),1)),
    e = Beta(1.0, 1.0),
    f = MvNormal([0.3,-2.9],Matrix([1.7 0.5;0.5 2.3]))
    )

posterior.likelihood.density._log_f(rand(posterior.prior))

posterior.likelihood.density._log_f(rand(prior2))

posterior2 = PosteriorDensity(BAT.logfuncdensity(posterior.likelihood.density._log_f), prior2)


@profview r_ram2 = @time BAT.bat_sample_impl(posterior2, TransformedMCMCSampling(pre_transform=PriorToGaussian(), nchains=4, nsteps=4*100000), context)

@profview r_mh2 = @time BAT.bat_sample_impl(posterior2, MCMCSampling( nchains=4, nsteps=4*100000, store_burnin=true), context)

r_hmc2 = @time BAT.bat_sample_impl(posterior2, MCMCSampling(mcalg=HamiltonianMC(), nchains=4, nsteps=4*20000), context)
