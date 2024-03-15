using Pkg

ENV["JULIA_NO_PRECOMPILE"] = "1" # Precompiling aus, da nicht immer alle Pakete verwendet werden?
Pkg.activate("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/Project.toml")
if !isfile("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/Manifest.toml")
    Pkg.add("Plots")
    Pkg.instantiate()
else
    println("Start without instantiate")
end
println("start :)")

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
using MonotonicSplines

include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/dev-internal/toolBox.jl")


####################################################################
# Generate MCMC samples which may look like Ensemblesampling
####################################################################
function aufteilen(arr::Vector{T}, block_size::Int) where T
    num_elements = length(arr)
    num_blocks = ceil(Int, num_elements / block_size)

    aufgeteilte_arrays = [T[] for _ in 1:num_blocks]
    for (index, value) in enumerate(arr)
        block_index = mod(index - 1, num_blocks) + 1
        push!(aufgeteilte_arrays[block_index], value)
    end
    return aufgeteilte_arrays
end

function test_MCMC(posterior,nsamp; simulate_walker=true)
    samples2 = []
    while (length(samples2) < nsamp)
        x = @time BAT.bat_sample(posterior,
                TransformedMCMCSampling(
                pre_transform=PriorToGaussian(), 
                init=TransformedMCMCChainPoolInit(),
                burnin=TransformedMCMCMultiCycleBurnin(max_ncycles=100),     
                tuning_alg=TransformedMCMCNoOpTuning(), strict=false, 
                nchains=4, nsteps=500),#354),
                context).result; # @TODO: Why are there so many samples?
        #plot(x,bins=200)#,xlims=xl)
        #savefig("$path/truth.pdf")
        print("Number of samples: ")
        println(sum(x.weight))
        print("Acceptance rate: ")
        println(length(x.v)/sum(x.weight))
    
        for i in 1:length(x.v)
            for j in 1:x.weight[i]
                push!(samples2,x.v[i])
            end
        end
    end
    # reorder samples to simulate an ensemble_sampling_process
    if simulate_walker
        samples = BAT2Matrix(vcat(aufteilen(samples2[1:nsamp],round(Int,nsamp/1000))...))
    else
        samples = BAT2Matrix(samples2[1:nsamp])
    end
    return samples
end


function tempering_gif(path; nsamp=100000, stepsize = 0.05, context = BATContext(ad = ADModule(:ForwardDiff)))
    factor = 0.0
    prior = BAT.NamedTupleDist(a = Uniform(-10,10))
    likelihood = params -> begin      
        return LogDVal(logpdf(MixtureModel(Normal, [(-3,1.0),(0,1.0),(3,1.0)],[1/3,1/3,1/3]), params[1]))
    end
    richtig, t = BAT.transform_and_unshape(BAT.DoNotTransform(), PosteriorDensity(likelihood, prior), context)
    ani= Animation()

    while (factor <= 1)
        likelihood = params -> begin      
            return LogDVal(factor*logpdf(MixtureModel(Normal, [(-3,1.0),(0,1.0),(3,1.0)],[1/3,1/3,1/3]), params[1]))
        end
        posterior, trafo = BAT.transform_and_unshape(BAT.DoNotTransform(), PosteriorDensity(likelihood, prior), context)
        samp = test_MCMC(posterior,nsamp)
        plot(flat2batsamples(samp'), density=true,right_margin=9Plots.mm)
        title!("$(size(samp,2)) Samples of [(flat prior) * ($factor * loglikelihood)]")
        xlims!(-10,10)
        ylims!(0,0.14)
        x_values = Vector(range(minimum(samp), stop=maximum(samp), length=1000))
        y(x) = densityof(richtig,[x])
        y_values = y.(x_values)
        factor2 = posterior.prior.bounds.vol.hi[1]-posterior.prior.bounds.vol.lo[1]
        p=plot!(x_values, y_values*factor2,density=true, linewidth=3.2,legend =:topright, label ="truth", color="black")
        frame(ani,p);
        factor = round(factor+stepsize,digits=3)
    end
    gif(ani, "$path/tempering.gif", fps=2)
end


##############################################################################
# This is the final version of sampling with a combinations of Flows and BAT
##############################################################################
function FlowSampling(path, posterior; dims = length(posterior.likelihood.shape), Knots=20, context =BATContext(ad = ADModule(:ForwardDiff)), 
                    n_samp=500000, tuner =MCMCFlowTuning(), use_mala=false, walker=1000, tau=0.5, nchains=1, 
                    flow = build_flow(rand(MvNormal(zeros(dims),I(dims)),10000), [InvMulAdd, RQSplineCouplingModule(dims, K = Knots)]),
                    marginaldistribution = get_triplemode(1), identystart=false, burnin=100000, pretrafo=BAT.PriorToGaussian())

    if identystart
        #target_logpdf = x -> logpdf(get_normal(dims)).(x)
        flow = AdaptiveFlows.optimize_flow(rand(MvNormal(zeros(dims),ones(dims)),10^5), flow, Adam(5f-3) , loss=AdaptiveFlows.negll_flow, nbatches = 4, 
                                                        nepochs = 100, shuffle_samples = true, logpdf = (AdaptiveFlows.std_normal_logpdf,AdaptiveFlows.std_normal_logpdf)).result
    end
    
    x = @time BAT.bat_sample_impl(posterior, 
                                TransformedMCMCSampling(pre_transform=pretrafo, 
                                                        init=TransformedMCMCEnsemblePoolInit(),
                                                        tuning_alg=tuner, tau=tau, nsteps=Int(n_samp/walker)+burnin-1,
                                                        adaptive_transform=BAT.CustomTransform(flow), use_mala=use_mala,
                                                        nchains=nchains, nwalker=walker),
                                                        context);
    samples = x.result.v[burnin*walker+1:end]
    print("Acceptance rate: ")
    println(length(unique(samples))/length(samples))

    samples_trafo = x.result_trafo.v
    plot_flow_alldimension(path, x.flow, BAT2Matrix(samples_trafo),Knots); # Flow was trained inside the PriorToGaussian()-Room
    plot_samples(path, BAT2Matrix(samples), marginaldistribution)
    return x
end


##############################################################################
# In one dimension is it eZ to investigate the Spline Function
##############################################################################
function SplineWatch_FlowSampling(path, posterior; dims = 1, Knots=20, context =BATContext(ad = ADModule(:ForwardDiff)), 
                        n_samp=500000, tuner =MCMCFlowTuning(), use_mala=false, walker=1000, tau=0.5, nchains=1, 
                        flow = build_flow(rand(MvNormal(zeros(dims),I(dims)),10000), [InvMulAdd, RQSplineCouplingModule(dims, K = Knots)]),
                        marginaldistribution = get_triplemode(1))
    if (length(posterior.likelihood.shape) != dims)
        println("Spline analysis at the moment only for 1D Cases")
        return FlowSampling(path, posterior; dims = length(posterior.likelihood.shape), Knots=Knots, context = context, 
                            n_samp= n_samp, tuner =tuner , use_mala=use_mala, walker=walker, tau=tau, nchains=nchains, 
                            marginaldistribution = marginaldistribution)
    end

    flow = build_flow(rand(MvNormal(zeros(dims),I(dims)),10000), [InvMulAdd, RQSplineCouplingModule(dims, K = Knots)])
    dummy = rand(MvNormal(zeros(dims),ones(dims)),10) # dummy-samples are used to evaluate the NN to get the spline function
    plot_spline(flow,dummy)
    savefig("$path/spline_before_$Knots.jpg")

    x = FlowSampling(path, posterior, dims = 1, flow=flow, context = context, Knots=Knots,
                            n_samp= n_samp, tuner =tuner , use_mala=use_mala, walker=walker, tau=tau, nchains=nchains, 
                            marginaldistribution = marginaldistribution)

    plot_spline(x.flow,dummy)
    savefig("$path/spline_after_$Knots.jpg")
end


####################################################################
# Look at Spline functions with differnt number of knots
####################################################################
function knot_gif(path, samp,target_logpdf;von=1,bis=10)
    path=make_Path("knots",path)
    ani= Animation()
    for i in von:bis
        print(i)
        flow, opt, lost_hist = train(samp,target_logpdf,K=i,epochs=100,batches=5,shuffle=true,opti=Adam(1f-2))
        p=plot_flow(flow,samp)
        savefig("$path/x$(nummer(i))")
        p=plot_spline(flow,samp,sigma=false)
        savefig("$path/$(nummer(i))")
        frame(ani,p);
    end
    gif(ani, "$path/knots.gif", fps=2)
end


####################################################################
# Sample a black-white image, draw samples from black
####################################################################
function makeBild(bild, path; n_samp=10^4)
    samp = image2samples(bild, target_samples=n_samp)
    plot(flat2batsamples(samp'), density=true,right_margin=9Plots.mm)
    savefig("$path/true.pdf")

    target_logpdf = x-> logpdf(MvNormal(zeros(2),I(2)),x)
    #using BenchmarkTools
    flow=build_flow(samp, [InvMulAdd, RQSplineCouplingModule(size(samp,1), K = 40)])
    lr=1f-3
    plot_flow(flow,samp)
    ylims!(-2.5,2.5)
    xlims!(-2.5,2.5)
    savefig("$path/dpg0.pdf")
    for i in 1:9
        @time flow, opt_state, loss_hist = AdaptiveFlows.optimize_flow(samp,flow, Adam(lr),
                               loss=AdaptiveFlows.negll_flow, #loss_history = loss_hist,
                               logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
                               nbatches=5,nepochs=1, shuffle_samples=true)
        plot(flat2batsamples(flow(samp)'))
        ylims!(-2.5,2.5)
        xlims!(-2.5,2.5)
        #plot_flow_alldimension(path,flow,samp,i)
        savefig("$path/dpg$i.pdf")
        #lr=lr*0.95
    end
    for i in 10:15
        @time flow, opt_state, loss_hist = AdaptiveFlows.optimize_flow(samp,flow, Adam(lr),
                               loss=AdaptiveFlows.negll_flow, #loss_history = loss_hist,
                               logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
                               nbatches=10,nepochs=5, shuffle_samples=true)
        plot(flat2batsamples(flow(samp)'))
        ylims!(-2.5,2.5)
        xlims!(-2.5,2.5)
        savefig("$path/dpg9$i.pdf")
        #lr=lr*0.95
    end
    plot_flow_alldimension(make_Path("marginals",path),flow,samp,ß)
end


####################################################################
# Train flow on existing samples
####################################################################
function train_flow(samples::Matrix, path::String, minibatches, epochs; batchsize=1000, eperplot=ceil(Int,(length(samples)/batchsize)/120),
                    lr=5f-2, K = 8, flow = build_flow(samples./(std(samples)/2), [InvMulAdd, RQSplineCouplingModule(size(samples,1), K = K)]), lrf=1, loss = [])
    path = make_Path("algo",path)
    meta = plot_metadaten(path, length(samples),minibatches, epochs, batchsize, lr, K, lrf)
    mkdir("$path/Loss_vali")
    mkdir("$path/ftruth")
    mkdir("$path/spline")
    animation_ft = Animation()
    ani_spline = Animation()
    
    AdaptiveFlows.optimize_flow_sequentially(samples[1:end,1:10],flow, Adam(lr),  # Precompile to be faster
                                                nbatches=1,nepochs=1, shuffle_samples=true);

    vali = [mvnormal_negll_flow(flow.flow.fs[2],flow.flow.fs[1](samples))]
    create_Animation(flow, samples, vali, path, 0, ani_spline, lr, meta, animation_ft=animation_ft)
    lr=round(lr,digits=6)
    make_slide("$path/spline/$(nummer(0)).pdf","$path/metadaten.pdf",title="Algo(batchsize=$batchsize, epochs=0, lr=$lr)")  
    batches=round(Int,(size(samples,2)/batchsize))
    for i in 1:batches
        println("$i von $batches Iterationen")
        flow, opt_state, loss_hist = AdaptiveFlows.optimize_flow_sequentially(samples[1:end,((i-1)*batchsize)+1:i*batchsize],flow, Adam(lr), 
                                                    nbatches=minibatches,nepochs=epochs, shuffle_samples=false);
        #for element in loss_hist[2][1]
        #    push!(loss, element)
        #end
        push!(loss, mean(loss_hist[2][1]))#element)
        push!(vali, mvnormal_negll_flow(flow.flow.fs[2],flow.flow.fs[1](samples)))

        if (i%eperplot) == 0
            create_Animation(flow, samples, vali, path, i*epochs, ani_spline, round(lr,digits=6),meta,animation_ft=animation_ft)
    
           # lr=round(lr,digits=6)
            #make_slide("$path/ftruth/$(nummer(epochs)).pdf","$path/spline/$(nummer(epochs)).pdf","$path/Loss_vali/$(nummer(epochs)).pdf",
            #            title="Algo(batchsize=$batchsize, epochs=$i*$epochs, minib=$minibatches, lr=$lr)")
        end
        if (lr > 5f-5)
            lr=lr*lrf
        else
            lr=5f-5
        end
    end
    epochs=batches
    #gif(animation_ft, "$path/ftruth/transform.gif", fps=min(ceil(Int,(epochs/eperplot)/15),10))
    gif(ani_spline, "$path/spline/spline.gif", fps=10)#min(ceil(Int,(epochs/eperplot)/15),10))
    lr=round(lr,digits=6)
    make_slide("$path/ftruth/$(nummer(epochs)).pdf","$path/spline/$(nummer(epochs)).pdf","$path/Loss_vali/$(nummer(epochs)).pdf",
                title="Training(batchsize=$batchsize, epochs=$epochs, minib=$minibatches, lr=$lr)")

    #return flow, loss, lr
    return BAT.CustomTransform(flow)
end


#####################################################################
# Train batchwise for an FIX AMOUNT OF TIME (and create gif)
#####################################################################
function timed_training(iid::Matrix, samples::Matrix, path2::String, minibatches, epochs, target_logpdf; batchsize=1000, eperplot=ceil(Int,(length(samples)/batchsize)/100),
                    lr=1f-2, K = 8, flow2 = build_flow(samples./(std(samples)/2), [InvMulAdd, RQSplineCouplingModule(size(samples,1), K = K)]), 
                    lrf=1, vali = [], loss_f=AdaptiveFlows.negll_flow, runtime=600)
    path = make_Path("$batchsize-$epochs-$minibatches-$lrf",path2)
    meta = plot_metadaten(path, length(samples),minibatches, epochs, batchsize, lr, K, lrf)
    mkdir("$path/Loss_vali")
    mkdir("$path/ftruth")
    mkdir("$path/spline")
    animation_ft = Animation()
    ani_spline = Animation()
    
    # Initialiate flow:
    flow, opt, lh = train(samples, target_logpdf, K=K, epochs=0)

    loss = [validate_pushfwd_negll(flow,samples)] 
    vali = [validate_pushfwd_negll(flow,iid)] 
    batches=round(Int,(size(samples,2)/batchsize))

    create_Animation(flow, samples, [100.0], path, 0, ani_spline, lr, meta, animation_ft=animation_ft, vali=[100.0])
    lr=round(lr,digits=6)
    make_slide("$path/spline/$(nummer(0)).pdf","$path/metadaten.pdf",title="Algo(batchsize=$batchsize, epochs=0, lr=$lr)") 
    
    i=0
    zeit = time()
    while (time()-zeit) < runtime
        i=i+1
        if i>batches
            i=1
        end
        println("$i von $batches Iterationen")

        flow, opt_state, loss_hist = train(samples[1:end,((i-1)*batchsize)+1:i*batchsize], target_logpdf, flow=flow,
                                      batches=minibatches, epochs=epochs, opti=Adam(lr), shuffle=false, loss=loss_f)
            
            
        #for element in loss_hist[2][1]
        #    push!(loss, element)
        #end
        push!(loss, mean(loss_hist[2][1]))#validate_pushfwd_negll(flow,samples))
        push!(vali, validate_pushfwd_negll(flow,iid))

        if (true)#(i%eperplot) == 0
            create_Animation(flow, samples, loss, path, i*epochs, ani_spline, round(lr,digits=6),meta,animation_ft=animation_ft, vali=vali)
        end
        if (lr > 5f-5)
            lr=lr*lrf
        else
            lr=5f-5
        end
    end
    # Mittelwert letzen 10 % des Trainings berechnen zur Reduzierung von Schwankung bei zu hoher lr
    midloss = mean(vali[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    midloss2 = mean(loss[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    file = open("$path2/2D_$(nummer(runtime)).txt", "a")
        write(file, "$epochs, $lrf, $midloss, $midloss2\n")
    close(file)

    epochs=batches
    #gif(animation_ft, "$path/ftruth/transform.gif", fps=3)
    gif(ani_spline, "$path/spline.gif", fps=3)
    
    #return flow, loss, lr
    return BAT.CustomTransform(flow)
end


#####################################################################
# Train batchwise for n batches
#####################################################################
function batchwise_training(iid::Matrix, samples::Matrix, path2::String, minibatches, epochs, target_logpdf; batchsize=1000, eperplot=1,
        lr=1f-2, K = 8, flow2 = build_flow(samples./(std(samples)/2), [InvMulAdd, RQSplineCouplingModule(size(samples,1), K = K)]), 
        lrf=1, vali = [], loss_f=AdaptiveFlows.negll_flow, batches=round(Int,(size(samples,2)/batchsize)))
    path = make_Path("$batchsize-$epochs-$minibatches-$lrf",path2)
    dims = size(samples,1)
    for i in 1:dims
        mkdir("$path/dim_$i")
    end
    meta = plot_metadaten(path, length(samples),minibatches, epochs, batchsize, lr, K, lrf)

    # Initialiate flow:
    flow, opt, lh = train(samples, target_logpdf, K=K, epochs=0)

    #loss = [validate_pushfwd_negll(flow,samples)] 
    #vali = [validate_pushfwd_negll(flow,iid)] 
    vali = [[100, 100.0, 100]]

    plot_flow_alldimension(path, flow, samples,0);
    #lr=round(lr,digits=6)
    #make_slide("$path/spline/$(nummer(0)).pdf","$path/metadaten.pdf",title="Algo(batchsize=$batchsize, epochs=0, lr=$lr)") 

    for i in 1:batches
        println("$i von $batches Iterationen")

        flow, opt_state, loss_hist = train(samples[1:end,((i-1)*batchsize)+1:i*batchsize], target_logpdf, flow=flow,
                              batches=minibatches, epochs=epochs, opti=Adam(lr), shuffle=false, loss=loss_f)


        #for element in loss_hist[2][1]
        #    push!(loss, element)
        #end
        #push!(loss, mean(loss_hist[2][1]))#validate_pushfwd_negll(flow,samples))
        #push!(vali, validate_pushfwd_negll(flow,iid))
        push!(vali, [mean(loss_hist[2][d]) for d in 1:dims])

        if (i%eperplot) == 0
            plot_flow_alldimension(path, flow, samples,i);
        end

        if (lr > 5f-4)
            lr=lr*lrf
        else
            lr= 5f-4
        end
    end
    plot_loss_alldimension(path,vali[2:end])
    # Mittelwert letzen 10 % des Trainings berechnen zur Reduzierung von Schwankung bei zu hoher lr
    midloss = mean(vali[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    #midloss2 = mean(loss[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    file = open("$path2/2D.txt", "a")
        write(file, "$epochs, $lrf, $(join(string.(midloss), ", ")) \n")# $midloss2\n")
    close(file)

    plot(flat2batsamples(flow(samples)'), density=true,right_margin=9Plots.mm)
    savefig("$path/result.pdf")
    return flow, vali
end


function get_tempered_samples(posterior, tfactor; n_samp=1000, dims=3, context = BATContext(ad = ADModule(:ForwardDiff)))
    posterior, trafo = BAT.transform_and_unshape(BAT.DoNotTransform(), posterior, context)
    target_logpdf = x-> BAT.checked_logdensityof(posterior).(x)
    mcmc=test_MCMC(posterior,n_samp,simulate_walker=false)
    samp=mcmc[1:end,1:n_samp]
    return samp, target_logpdf
end


function tempering_training(posterior, iid::Matrix, path2::String, minibatches, epochs; tempersize=0.05, batchsize=1000, eperplot=2, peaks=3,
    lr=1f-2, K = 20, flow2 = build_flow(iid./(std(iid)/2), [InvMulAdd, RQSplineCouplingModule(size(iid,1), K = K)]), 
    lrf=0.95, vali = [], loss_f=AdaptiveFlows.negll_flow, batches=round(Int,(size(iid,2)/batchsize)))
    path = make_Path("$batchsize-$epochs-$minibatches-$lrf",path2)
    dims = size(iid,1)
    for i in 1:dims
        mkdir("$path/dim_$i")
    end
    meta = plot_metadaten(path, 0,minibatches, epochs, batchsize, lr, K, lrf)

    # Initialiate flow:
    samples, target_logpdf = get_tempered_samples(posterior, 0,n_samp=10)
    flow, opt, lh = train(samples, target_logpdf, K=K, epochs=0, flow=flow2)

    vali = [[100, 100.0, 100]]
    plot_flow_alldimension(path, flow, iid,0);

    tf=0.0
    i=0
    while tf <= 1
        println("tf = $tf, lr = $lr")


        nuridentflow = FlowSampling(path, posterior, use_mala=false, n_samp=10^5,Knots=Knots,flow=flow,
                                    marginaldistribution=marginal, identystart=false, 
                                    tuner=BAT.TransformedMCMCNoOpTuning(),burnin=100)


        minibatches=1
        epochs=10
        samples = BAT2Matrix(nuridentflow.result_trafo.v)
        #samples, target_logpdf = get_tempered_samples(posterior, tf,n_samp=batchsize,dims=size(iid,1))
        flow, opt_state, loss_hist = train(samples, target_logpdf, flow=flow,
                              batches=minibatches, epochs=epochs, opti=Adam(lr), shuffle=true, loss=loss_f)

        push!(vali, [mean(loss_hist[2][d]) for d in 1:dims])

        if (i%eperplot) == 0
            plot_flow_alldimension(path, flow, iid,Int(tf*100));
        end

        if (lr > 5f-5)
            lr=lr*lrf
        else
            lr=5f-5
        end
        #if tf > 0.8
        #    tempersize=0.01
        #end
        tf = round(tf+tempersize,digits=3)
        i +=1
    end

    plot_loss_alldimension(path,vali[2:end])
    # Mittelwert letzen 10 % des Trainings berechnen zur Reduzierung von Schwankung bei zu hoher lr
    midloss = mean(vali[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    #midloss2 = mean(loss[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    file = open("$path2/2D.txt", "a")
        write(file, "$epochs, $lrf, $(join(string.(midloss), ", ")) \n")# $midloss2\n")
    close(file)

    plot(flat2batsamples(flow(iid)'), density=true,right_margin=9Plots.mm)
    savefig("$path/result.pdf")
    return flow, vali
end



function tempering_test(iid::Matrix, path2::String, minibatches, epochs; tempersize=0.05, batchsize=1000, eperplot=2, peaks=3,
    lr=0.05, K = 20, flow2 = build_flow(iid./2, [InvMulAdd, RQSplineCouplingModule(size(iid,1), K = K)]), 
    lrf=1, vali = [], loss_f=AdaptiveFlows.negll_flow, batches=round(Int,(size(iid,2)/batchsize)))
    path = make_Path("$batchsize-$epochs-$minibatches-$lrf",path2)
    dims = size(iid,1)
    for i in 1:dims
        mkdir("$path/dim_$i")
    end
    meta = plot_metadaten(path, 0,minibatches, epochs, batchsize, lr, K, lrf)
    plot(flat2batsamples(iid'))
    savefig("$path/testsamples.pdf")
    
    # init Posterior
    tf=0.0
    posterior= get_bachelor(dims,tf=tf)
    # Initialiate flow:
    samples, target_logpdf = get_tempered_samples(posterior, 0,n_samp=10)
    flow, opt, lh = train(iid, target_logpdf, K=K, epochs=0, flow=flow2)

    vali = [[100, 100.0, 100]]
    plot_flow_alldimension(path, flow, iid,0);

    i=0
    while tf <= 1
        println("tf = $tf, lr = $lr")
        posterior= get_bachelor(dims,tf=tf)

        nuridentflow = FlowSampling(path, posterior, use_mala=false, n_samp=10^5,Knots=Knots,flow=flow,
                                    marginaldistribution=marginal, identystart=false, 
                                    tuner=BAT.TransformedMCMCNoOpTuning(),burnin=100,pretrafo=BAT.DoNotTransform())

        minibatches=1
        epochs=10
        samples = BAT2Matrix(nuridentflow.result_trafo.v)
        plot(flat2batsamples(samples'))
        savefig("$path/samples_trafo_$i.pdf")
        s=BAT2Matrix(nuridentflow.result.v)
        plot(flat2batsamples(s'))
        savefig("$path/samples_$i.pdf")
        #samples, target_logpdf = get_tempered_samples(posterior, tf,n_samp=batchsize,dims=size(iid,1))
        flow, opt_state, loss_hist = train(samples, target_logpdf, flow=flow,
                              batches=minibatches, epochs=epochs, opti=Adam(lr), shuffle=true, loss=loss_f)

        push!(vali, [mean(loss_hist[2][d]) for d in 1:dims])

        if (i%eperplot) == 0
            plot_flow_alldimension(path, flow, iid,Int(tf*100));
        end
        #p=plot_spline(flow,iid)
        #savefig("$path/spline_$i.pdf")

        if (lr > 5f-5)
            lr=lr*lrf
        else
            lr=5f-5
        end
        #if tf > 0.8
        #    tempersize=0.01
        #end
        tf = round(tf+tempersize,digits=3)
        i +=1
    end

    plot_loss_alldimension(path,vali[2:end])
    # Mittelwert letzen 10 % des Trainings berechnen zur Reduzierung von Schwankung bei zu hoher lr
    midloss = mean(vali[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    #midloss2 = mean(loss[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    file = open("$path2/2D.txt", "a")
        write(file, "$epochs, $lrf, $(join(string.(midloss), ", ")) \n")# $midloss2\n")
    close(file)

    plot(flat2batsamples(flow(iid)'), density=true,right_margin=9Plots.mm)
    savefig("$path/result.pdf")
    return flow, vali
end
#####################################################################
# Use Loss functions of Adaptive Flows for validation on testsamples
#####################################################################
function validate_pushfwd_negll(flow,samp)
    dims=size(samp,1)
    f=AdaptiveFlows.PushForwardLogDensity(FlowModule(InvMulAdd(I(dims), 
        zeros(dims)), false), AdaptiveFlows.std_normal_logpdf)
    #return AdaptiveFlows.negll_flow_loss(flow.flow.fs[2],flow.flow.fs[1](samp), f(flow.flow.fs[1](samp)), f)
    return [AdaptiveFlows.negll_flow_loss(flow.flow.fs[i],flow.flow.fs[1](samp), f(flow.flow.fs[1](samp)), f) for i in 2:dims+1] #because first ist ScaleShift
end

function validate_pushfwd_KLDiv(flow,samp)
    f=AdaptiveFlows.PushForwardLogDensity(FlowModule(InvMulAdd(I(size(samp,1)), 
        zeros(size(samp,1))), false), AdaptiveFlows.std_normal_logpdf)
    return AdaptiveFlows.KLDiv_flow_loss(flow.flow.fs[2],flow.flow.fs[1](samp), f(flow.flow.fs[1](samp)), f)
    # Starting here to test KLDiv and finde the error
    #x=flow.flow.fs[1](samp)
    #logd_orig = f(flow.flow.fs[1](samp))
    #nsamples = size(x, 2) 
    #flow_corr = flow.flow.fs[2]#fchain(flow,f.f)
    ##logpdf_y = logpdfs[2].logdensity
    #y, ladj = with_logabsdet_jacobian(flow_corr, x)
    #ll = target_logpdf(x) - vec(ladj)
    #KLDiv = sum(exp.(ll) .* ((ll) - AdaptiveFlows.std_normal_logpdf(y))) / nsamples
end

function negll_flow_loss2(flow::F, x::AbstractMatrix{<:Real}, logpdf::Function) where F<:AbstractFlow
    nsamples = size(x, 2) 
    flow_corr = fchain(flow,logpdf.f)
    y, ladj = with_logabsdet_jacobian(flow_corr, x)
    ll = (sum(logpdf.logdensity(y)) + sum(ladj)) / nsamples
    return -ll
end

function mvnormal_negll_flow2(flow::F, x::AbstractMatrix{<:Real}) where F<:AbstractFlow
    nsamples = size(x, 2) 
    
    y, ladj = with_logabsdet_jacobian(flow, x)
    ll = (sum(logpdf.(Normal(0,1),flow(x))) + sum(ladj)) / nsamples # Ich rechne hier Minus statt plus !

    return -ll
end


#####################################################################
# Method to simulate batchwise Flow training during Ensemblesampling
#####################################################################
function train_flow_batchwise(samples::Matrix, path::String, minibatches, epochs, batchsize; 
            lr=5f-2, K = 8, eperplot=ceil(Int,(length(samples)/batchsize)/120), flow = build_flow(samples, [InvMulAdd, RQSplineCouplingModule(size(samples,1), K = K)]), 
            lrf=1, loss = [mvnormal_negll_flow(flow.flow.fs[2],flow.flow.fs[1](samples))])

    path = make_Path("train",path)
    meta = plot_metadaten(path, length(samples),minibatches, epochs, batchsize, lr, K, lrf)
    mkdir("$path/Loss_vali")
    mkdir("$path/ftruth")
    mkdir("$path/spline")
    animation_ft = Animation()
    ani_spline = Animation()
    
    AdaptiveFlows.optimize_flow_sequentially(samples[1:end,1:10],flow, Adam(lr),  # Precompile to be faster
                                                nbatches=1,nepochs=1, shuffle_samples=true);

    create_Animation(flow, samples, loss, path, 0, ani_spline, lr, meta, animation_ft=animation_ft)
    lr=round(lr,digits=6)
    make_slide("$path/spline/$(nummer(0)).pdf","$path/metadaten.pdf",title="Train(batchsize=$batchsize, epochs=0, lr=$lr)")   

    iters=Int(epochs/eperplot)
    for epoch in 1:iters
        println("$epoch von $iters Iterationen")
        flow, opt_state, loss_hist = AdaptiveFlows.optimize_flow_sequentially(samples[1:end,1:batchsize],#[1:end,((i-1)*batchsize)+1:i*batchsize],
                                        flow, Adam(lr), nbatches=minibatches,nepochs=eperplot, shuffle_samples=true);#false);
        for element in loss_hist[2][1]
            push!(loss, element)
        end

        create_Animation(flow, samples, loss, path, eperplot*epoch, ani_spline, round(lr,digits=6), meta, animation_ft=animation_ft)
        lr=round(lr,digits=6)
        #make_slide("$path/ftruth/$(nummer(epoch)).pdf","$path/spline/$(nummer(epoch)).pdf","$path/Loss_vali/$(nummer(epoch)).pdf",
        #            title="Training(batchsize=$batchsize, epochs=$epoch*$eperplot, minib=$minibatches, lr=$lr)")
        if (lr > 5f-5)
            lr=lr*lrf
        else
            lr=5f-5
        end
    end
    
    # Make gifs  
    #gif(animation_ft, "$path/ftruth/transform.gif", fps=min(ceil(Int,(epochs/eperplot)/15),10))
    gif(ani_spline, "$path/spline/spline.gif", fps=10)#min(ceil(Int,(epochs/eperplot)/15),10))
    lr=round(lr,digits=6)
    make_slide("$path/ftruth/$(nummer(epochs)).pdf","$path/spline/$(nummer(epochs)).pdf","$path/Loss_vali/$(nummer(epochs)).pdf",
                title="Training(batchsize=$batchsize, epochs=$epochs, minib=$minibatches, lr=$lr)")

    return flow, loss, lr
end


####################################################################
# Sample a black-white image, draw samples from black - may old?
####################################################################
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

println("ende")
####################################################################
# Test the FlowTuner
####################################################################
#t_mh, flow3=EnsembleSampling(posterior,f,false,MCMCFlowTuning()); # MC prop. # @TODO: Why nan :( # There is NaN because we train on the same samples, because low acptrate
#plot(t_mh,bins=200,xlims=xl) # @TODO: Investigate why it becomes so nice :D
#i=1
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