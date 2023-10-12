using Pkg
Pkg.activate(".")

using Revise
using BAT

#using Plots
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
using AdaptiveFlows

#include("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/wweber/Documents/eNormalizingFlows/FlowShowCase/Posterior.jl")
#posterior = get_mix(2)
#posterior = MvNormal(zeros(2),I(2))
posterior = BAT.example_posterior()
x = BAT.bat_sample(posterior,MCMCSampling()).result 
samples::Matrix{Float32} = flatview(unshaped.(x.v))
flow=build_flow(samples)
flow(samples)
#flow = optimize_flow_sequentially(samples,flow,Adam(1f-3)).result

context = BATContext(ad = ADModule(:ForwardDiff))
density_notrafo = convert(BAT.AbstractMeasureOrDensity, posterior)
density, trafo = BAT.transform_and_unshape(PriorToGaussian(), density_notrafo, context)

s = cholesky(Positive, BAT._approx_cov(density)).L
f = BAT.CustomTransform(flow.flow.fs[2])
#f = BAT.CustomTransform(flow.flow.fs[2].flow.fs[1])
#f = BAT.CustomTransform(x->2x)
#f = BAT.CustomTransform(Mul(s))
(f::RQSplineCouplingBlock)(x::AbstractVector) = vec(f(reshape(x, :, 1)))

function ChangesOfVariables.with_logabsdet_jacobian(f::RQSplineCouplingBlock, x::AbstractVector)
    #println("x",x)
    #println("type, flow", f.flow)
    y, ladj = ChangesOfVariables.with_logabsdet_jacobian(f, reshape(x, :, 1))
    return vec(y), ladj[1]
end

(f::InverseRQSplineCouplingBlock)(x::AbstractVector) = vec(f(reshape(x, :, 1)))

function ChangesOfVariables.with_logabsdet_jacobian(f::InverseRQSplineCouplingBlock, x::AbstractVector)
    #println("x",x)
    #println("type, flow", f.flow)
    y, ladj = ChangesOfVariables.with_logabsdet_jacobian(f, reshape(x, :, 1))
    return vec(y), ladj[1]
end
my_result = @time BAT.bat_sample_impl(posterior, 
                TransformedMCMCSampling(pre_transform=PriorToGaussian(), tuning_alg=TransformedMCMCNoOpTuning(), adaptive_transform=f, nchains=4, nsteps=100),
                 context).result 

samples::Matrix{Float32} = flatview(unshaped.(my_result.v))

function flat2batsamples(smpls_flat)
    n = length(smpls_flat[:,1])
    smpls = [smpls_flat[i,1:end] for i in 1:n]
    weights = ones(length(smpls))
    logvals = zeros(length(smpls))
    return BAT.DensitySampleVector(smpls, logvals, weight = weights)
end

function trainsample(posterior, flow)
    f = BAT.CustomTransform(flow.flow.fs[2])
    my_result = @time BAT.bat_sample_impl(posterior, TransformedMCMCSampling(pre_transform=PriorToGaussian(), tuning_alg=TransformedMCMCNoOpTuning(), adaptive_transform=f, nchains=4, nsteps=1000),context).result 
    samples::Matrix{Float32} = flatview(unshaped.(my_result.v))
    flow = optimize_flow_sequentially(samples,flow,Adam(1f-3),nbatches=10,nepochs=10,shuffle_samples=true).result
    flow(samples)

    return flow
end

n = bat_sample(MvNormal(zeros(7),I(7))).result
normal::Matrix{Float32} = flatview(unshaped.(n.v))

@time flow = trainsample(posterior,flow)
samp = inverse(flow)(normal)
plot(flat2batsamples(samp'))



plot(my_result)
#a, b, c, d = BAT.g_state
g, z, μ, target, v_init = BAT.g_state

using BenchmarkTools
logd_z = @btime logdensityof(MeasureBase.pullback(g, μ),z)

#g2 = BAT.CustomTransform(Mul(s))
x = @btime MeasureBase.pullback(g, μ);
l = @btime logdensityof(x);
logd_z = @btime l(z);

z2 = rand(posterior.prior);
logd_z = @btime logdensityof(posterior)(z2);

l = logdensityof(MeasureBase.pullback(g, μ));

function myFunction(l, z)
    for i in 1:100
        l(z)
    end
end

@profview myFunction(l, z)

μ(z)

gg = inverse(g)



try
    flow = get_flow_musketeer(7,CPU(),10);#,10,10);
    #flow = open(deserialize,"/ceph/groups/e4/users/wweber/private/Master/Plots/mix_dims_mySampler/2D_mix/flow.jls");
    println(flow(samples)[:,1:3])
    flow=inverse(flow)
    println(flow(samples)[:,1:3])
    f=  BAT.CustomTransform(flow)
    #println(f.f)
    my_result = @time BAT.bat_sample_impl(posterior, TransformedMCMCSampling(adaptive_transform=f), context).result
    #plot(my_result.result)
    #savefig("/ceph/groups/e4/users/wweber/private/Master/Code/BAT_trafo/test.pdf")
    #flow(samples)
  #  @time flow, hist = train(flow, samples, 1, 4f-3, 10,100);#, l = 4f-3, nbatches = 10, nepochs = 500, wanna_use_GPU = false)
  #  samples2 = flow(samples)
catch e
    println(e)
end

#ENV["JULIA_DEBUG"] = "BAT"


context = BATContext(ad = ADModule(:ForwardDiff))

#posterior = BAT.example_posterior()

my_result = @time BAT.bat_sample_impl(posterior, TransformedMCMCSampling(pre_transform=PriorToGaussian(), nchains=4, nsteps=4*100000), context)


density_notrafo = convert(BAT.AbstractMeasureOrDensity, posterior)
density, trafo = BAT.transform_and_unshape(PriorToGaussian(), density_notrafo, context)

s = cholesky(Positive, BAT._approx_cov(density)).L
f = BAT.CustomTransform(Mul(s))

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
https://github.com/bat/AdaptiveFlows.jl/tree/dev