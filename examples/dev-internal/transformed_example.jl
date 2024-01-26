using Pkg

@time begin
    Pkg.activate("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/Project.toml")
    Pkg.instantiate()
end

#using Revise
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


function make_Path(testname::String,dir::String="/ceph/groups/e4/users/wweber/private/Master/Plots")
    path = "$dir/$testname"
    while (isdir(path))
        println("WARNUNG: testname existiert bereits, mache Trick.")
        path = "$path/Trick"
    end
    mkpath(path)
    return path
end


function flat2batsamples(smpls_flat)
    n = length(smpls_flat[:,1])
    smpls = [smpls_flat[i,1:end] for i in 1:n]
    weights = ones(length(smpls))
    logvals = zeros(length(smpls))
    return BAT.DensitySampleVector(smpls, logvals, weight = weights)
end


function test_MCMC(posterior,path::String)
    ####################################################################
    # Sampling without ensembles and flow
    ####################################################################
    x = @time BAT.bat_sample(posterior,
            TransformedMCMCSampling(
                pre_transform=PriorToGaussian(), 
                init=TransformedMCMCChainPoolInit(),
                tuning_alg=TransformedMCMCNoOpTuning(), 
                nchains=4, nsteps=354),
            context).result; # @TODO: Why are there so many samples?
    plot(x,bins=200)#,xlims=xl)
    savefig("$path/truth.pdf")
    print("Number of samples: ")
    println(sum(x.weight))
    print("Acceptance rate: ")
    println(length(x.v)/sum(x.weight))
    
    samples2 = []
    for i in 1:length(x.v)
        for j in 1:x.weight[i]
            push!(samples2,x.v[i])
        end
    end
    samples = BAT2Matrix(samples2)
    return samples
end


####################################################################
# Ensembles without flow
####################################################################
function EnsembleSampling(posterior, f;use_mala=true, tuning=MCMCFlowTuning(), 
                            tau=0.01, nchains=1, nsteps=100,nwalker=100)
    context = BATContext(ad = ADModule(:ForwardDiff))
    y = @time BAT.bat_sample_impl(posterior, 
        TransformedMCMCSampling(
            pre_transform=PriorToGaussian(), 
            init=TransformedMCMCEnsemblePoolInit(), #@TO-Do:  nsteps angeben, faktor wie viele
            tuning_alg=tuning, tau=tau,
            adaptive_transform=f, use_mala=use_mala,
            nchains=nchains, nsteps=nsteps,nwalker=nwalker),
            context);
    x = y.result
    print("Number of samples: ")
    println(sum(x.weight))
    print("Acceptance rate: ")
    println(length(unique(x.v))/length(x.v))
    return flatview(x.v), y.flow
end


# density_notrafo = convert(BAT.AbstractMeasureOrDensity, posterior)
# densit, trafo = BAT.transform_and_unshape(PriorToGaussian(), density_notrafo, context)
# 
# s = cholesky(Positive, BAT._approx_cov(densit)).L
# mul = BAT.CustomTransform(Mul(s))
# 
# y, flow = EnsembleSampling(posterior,mul,false);
# plot(y,bins=200,xlims=xl)
# EnsembleSampling(posterior,mul,true)

####################################################################
# Flow trained to identity
####################################################################
function get_identity_flow(dim::Int, path::String)
    d = dim
    n = bat_sample(MvNormal(zeros(d),I(d))).result
    normal::Matrix{Float32} = flatview(unshaped.(n.v))
    flow_n = build_flow(normal)
    @time flow_n, opt_state, loss_hist = AdaptiveFlows.optimize_flow_sequentially(normal, flow_n, Adam(1f-2), nbatches=1,nepochs=100, shuffle_samples=true);

    # Plot erstellen
    x = 1:length(loss_hist[2][1])
    plot(x, loss_hist[2][1], label="Loss curve", xlabel="X-Achse", ylabel="Y-Achse")
    savefig("$path/ident1.pdf")
    plot(flat2batsamples(normal'),bins=200)
    savefig("$path/ident2.pdf")
    plot(flat2batsamples(inverse(flow_n)(normal)'),bins=200)
    savefig("$path/ident3.pdf")

    return BAT.CustomTransform(flow_n)
end

####################################################################
# Test the Flow without tuning
####################################################################
# z_mh, flow=EnsembleSampling(posterior,f,false); # MC prop.
# plot(z_mh,bins=200,xlims=xl)
# 
# z_mala, flow2 =EnsembleSampling(posterior,f,true);
# plot(z_mala,bins=200,xlims=xl)

####################################################################
# Test the FlowTuner
####################################################################
#t_mh, flow3=EnsembleSampling(posterior,f,false,MCMCFlowTuning()); # MC prop. # @TODO: Why nan :( # There is NaN because we train on the same samples, because low acptrate
#plot(t_mh,bins=200,xlims=xl) # @TODO: Investigate why it becomes so nice :D
#i=1
function iterative_test(posterior, iters::Int, path::String, testsamples::Matrix; use_mala=true, tuning=MCMCFlowTuning(), 
                            tau=0.01, nchains=1, nsteps=100,nwalker=100)
    datei_val = "$path/data.txt"
    open(datei_val, "w") do file
        write(file, "")
    end

    datei = "/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/src/data.txt"
    open(datei, "w") do file
        write(file, "")
    end

    flow = build_flow(testsamples)
    f= BAT.CustomTransform(flow)
    loss_data_v = []
    push!(loss_data_v,mvnormal_negll_flow2(flow,testsamples))
    for i in 1:iters
        println(i)
        ii = string(i)
        mkpath("$path/$ii/")
        mala, flow = EnsembleSampling(posterior,f,use_mala=use_mala, tuning=tuning, 
                                                    tau=tau, nchains=nchains, nsteps=nsteps,nwalker=nwalker)
        plot(flat2batsamples(mala'),bins=200)                                                                         # @TO-DO. Flow lernt wenig und wenn das falsche
        savefig("$path/$ii/samples.pdf")

        #samples::Matrix{Float32} = flatview(unshaped.(t_mala.v))
        plot(flat2batsamples(flow(testsamples)'))
        savefig("$path/$ii/flowTruth.pdf")

        f = BAT.CustomTransform(flow)

        #Validation:
        push!(loss_data_v,mvnormal_negll_flow2(flow,testsamples))
        #open(datei_val, "a") do file # @ToDo lieber in Array pushen und außerhalb der schleife schreiben
        #    write(file, "$v")
        #    if i != iters
        #        write(file, ", ")
        #    end
        #end
    end

    daten_string = read(datei, String)
    daten_string = replace(daten_string, "][" => ", ")
    daten_string = replace(daten_string, "[" => " ")
    daten_string = replace(daten_string, "]" => " ")
    loss_data = parse.(Float64, split(daten_string, ", "))
    plot(loss_data, linewidth=1,xlabel="Epoch", ylabel="Loss", title="Loss", left_margin = 6Plots.mm)
    for i in 1:iters
        vline!([i*(length(loss_data)/iters)],color="red",label=false)
        savefig("$path/iterateLoss.pdf") 
    end
    
    #loss_data_v = parse.(Float64, split(read(datei_val, String), ", "))
    plot(loss_data_v, linewidth=1,xlabel="Iteration", ylabel="Loss", title="Loss", left_margin = 6Plots.mm)
    savefig("$path/iterateLoss_vali.pdf") 
end


function mvnormal_negll_flow2(flow::F, x::AbstractMatrix{<:Real}) where F<:AbstractFlow
    nsamples = size(x, 2) 
    
    y, ladj = with_logabsdet_jacobian(flow, x)
    ll = (sum(logpdf.(Normal(0,1),flow(x))) - sum(ladj)) / nsamples # Ich rechne hier Minus statt plus !

    return -ll
end


function make_slide(pdfpath1,pdfpath2; slidepath = "/ceph/groups/e4/users/wweber/private/Master/Slides", title="Title")
         
    file = open("$slidepath/plots.tex", "a")
    write(file, "\\begin{frame}{$title}\n")
    write(file, "   \\begin{figure}\n")
    write(file, "       \\centering\n")
    write(file, "       \\begin{subfigure}{0.5\\textwidth}\n")
    write(file, "           \\includegraphics[width=0.5\\textwidth]{$pdfpath1}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "       \\begin{subfigure}{0.5\\textwidth}\n")
    write(file, "           \\includegraphics[width=0.5\\textwidth]{$pdfpath2}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "   \\end{figure}\n")
    write(file, "\\end{frame}\n\n")
    close(file)
end


####################################################################
# Train flow on existing samples
####################################################################
function train_flow(samples::Matrix, path::String, minibatches, epochs, batchsize)
    flow=build_flow(samples) #samples)  # ATTENTION: There are big Problems if there is some ScaleShifting in the Flow!
    loss = []
    vali = []
    train = []
    lr=1f-3
    for i in 1:(round(Int,(size(samples,2)/batchsize))-1)
        #lr=lr*0.995
        flow, opt_state, loss_hist = AdaptiveFlows.optimize_flow_sequentially(samples[1:end,((i-1)*batchsize)+1:i*batchsize],flow, Adam(lr), 
                                                    nbatches=minibatches,nepochs=epochs, shuffle_samples=true);
        #for element in loss_hist[2][1]
        push!(loss, mean(loss_hist[2][1]))#element)
        push!(vali, mvnormal_negll_flow2(flow,samples))
        push!(train, mvnormal_negll_flow2(flow,samples[1:end,((i-1)*batchsize)+1:i*batchsize]))
        #end
    end
    x = 1:length(loss)
    plot(x, loss, label="State loss", xlabel="State", ylabel="Loss", title="Loss", left_margin = 6Plots.mm)
    savefig("$path/Loss.pdf")

    x = 1:length(vali)
    plot!(x, vali, label="Validation Loss")
    plot!(1:length(train), train, label="Trainings Loss")
    savefig("$path/Loss_vali.pdf")

    plot(flat2batsamples(flow(samples)'))
    savefig("$path/ftruth.pdf")
    println(lr)
    make_slide("$path/Loss_vali.pdf","$path/ftruth.pdf",title="bs_$batchsize-ep_$epochs-minib_$minibatches")
    return BAT.CustomTransform(flow)
end

println("ende")
# 
# # Test the Flow without tuning
# z_mala2, flow2 =EnsembleSampling(posterior,f2,true);
# plot(z_mala2,bins=200,xlims=xl)                                                                        # @TO-DO. Mit besseren Flow wird das Sampling schlechter    
# plot(flat2batsamples(flow2(flatview(z_mala2.v))'))
# 
# # Test the FlowTuner
# t_mala2, flow5 = EnsembleSampling(posterior,f2,true,MCMCFlowTuning());
# plot(t_mala2,bins=200,xlims=xl)                                                                         # @TO-DO. Hier bringt Training dann plötzlich gutes Improvement
# plot(flat2batsamples(flow5(flatview(t_mala2.v))'))
# 



# ###############################
# # Old code below this
# ###############################
# 
# samp = inverse(flow)(normal)
# plot(flat2batsamples(samp'))
# 
# using BenchmarkTools
# logd_z = @btime logdensityof(MeasureBase.pullback(g, μ),z)
# 
# 
# g= BAT.CustomTransform(Mul(s))
# x = @btime MeasureBase.pullback(g, posterior)
# l = @btime logdensityof(x);
# logd_z = @btime l(z);
# 
# z2 = rand(posterior.prior);
# logd_z = logdensityof(posterior)(z2)
# 
# l = logdensityof(MeasureBase.pullback(g, μ));
# 
# function myFunction(l, z)
#     for i in 1:100
#         l(z)
#     end
# end
# 
# @profview myFunction(l, z)
# 
# μ(z)
# 
# gg = inverse(g)
# 
# 
# 
# context = BATContext(ad = ADModule(:ForwardDiff))
# 
# #posterior = BAT.example_posterior()
# 
# my_result = @time BAT.bat_sample_impl(posterior, TransformedMCMCSampling(pre_transform=PriorToGaussian(), nchains=4, nsteps=4*100000), context)
# 
# 
# 
# my_result = @time BAT.bat_sample_impl(posterior, TransformedMCMCSampling(pre_transform=PriorToGaussian(), tuning_alg=TransformedAdaptiveMHTuning(), nchains=4, nsteps=4*100000, adaptive_transform=f), context)
# 
# my_samples = my_result.result
# 
# 
# 
# using Plots
# plot(my_samples)
# 
# r_mh = @time BAT.bat_sample_impl(posterior, MCMCSampling( nchains=4, nsteps=4*100000, store_burnin=true), context)
# 
# r_hmc = @time BAT.bat_sample_impl(posterior, MCMCSampling(mcalg=HamiltonianMC(), nchains=4, nsteps=4*20000), context)
#  
# plot(bat_sample(posterior).result)
# 
# using BAT.Distributions
# using BAT.ValueShapes
# prior2 = NamedTupleDist(ShapedAsNT,
#     b = [4.2, 3.3],
#     a = Exponential(1.0),
#     c = Normal(1.0,3.0),
#     d = product_distribution(Weibull.(ones(2),1)),
#     e = Beta(1.0, 1.0),
#     f = MvNormal([0.3,-2.9],Matrix([1.7 0.5;0.5 2.3]))
#     )
# 
# posterior.likelihood.density._log_f(rand(posterior.prior))
# 
# posterior.likelihood.density._log_f(rand(prior2))
# 
# posterior2 = PosteriorDensity(BAT.logfuncdensity(posterior.likelihood.density._log_f), prior2)
# 
# 
# @profview r_ram2 = @time BAT.bat_sample_impl(posterior2, TransformedMCMCSampling(pre_transform=PriorToGaussian(), nchains=4, nsteps=4*100000), context)
# 
# @profview r_mh2 = @time BAT.bat_sample_impl(posterior2, MCMCSampling( nchains=4, nsteps=4*100000, store_burnin=true), context)
# 
# r_hmc2 = @time BAT.bat_sample_impl(posterior2, MCMCSampling(mcalg=HamiltonianMC(), nchains=4, nsteps=4*20000), context)
# 