using Pkg

@time begin
    Pkg.activate("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/Project.toml")
    Pkg.instantiate()
    #Pkg.add("Plots")
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
using MonotonicSplines


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

function make_slide(pdfpath1; slidepath = "/ceph/groups/e4/users/wweber/private/Master/Slides", title="Title")
         
    title = replace(title, "_" => "\\_")
    file = open("$slidepath/plots.tex", "a")
    write(file, "\\begin{frame}{$title}\n")
    write(file, "   \\begin{figure}\n")
    write(file, "       \\centering\n")
    write(file, "       \\includegraphics[width=0.7\\textwidth]{$pdfpath1}\n")
    write(file, "   \\end{figure}\n")
    write(file, "\\end{frame}\n\n")
    close(file)
end

function make_slide(pdfpath1,pdfpath2; slidepath = "/ceph/groups/e4/users/wweber/private/Master/Slides", title="Title")
         
    title = replace(title, "_" => "\\_")
    file = open("$slidepath/plots.tex", "a")
    write(file, "\\begin{frame}{$title}\n")
    write(file, "   \\begin{figure}\n")
    write(file, "       \\centering\n")
    write(file, "       \\begin{subfigure}{0.49\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath1}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "       \\begin{subfigure}{0.49\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath2}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "   \\end{figure}\n")
    write(file, "\\end{frame}\n\n")
    close(file)
end

function make_slide(pdfpath1,pdfpath2,pdfpath3; slidepath = "/ceph/groups/e4/users/wweber/private/Master/Slides", title="Title")
         
    title = replace(title, "_" => "\\_")
    file = open("$slidepath/plots.tex", "a")
    write(file, "\\begin{frame}{$title}\n")
    write(file, "   \\begin{figure}\n")
    write(file, "       \\centering\n")
    write(file, "       \\begin{subfigure}{0.44\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath1}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "       \\begin{subfigure}{0.44\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath2}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "       \\begin{subfigure}{0.35\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath3}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "   \\end{figure}\n")
    write(file, "\\end{frame}\n\n")
    close(file)
end


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


function test_MCMC(posterior,nsamp)#;path::String)
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
    #plot(x,bins=200)#,xlims=xl)
    #savefig("$path/truth.pdf")
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
    # reorder samples to simulate an ensemble_sampling_process
    samples = BAT2Matrix(vcat(aufteilen(samples2[1:nsamp],round(Int,nsamp/1000))...))
    return samples
end


function nummer(x; digits=5)
    return string(x, pad=digits, base=10)
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

function make_Gif

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

function normalize(data::Matrix)
    mean_value = mean(data)
    std_value = std(data)
    normalized_data = (data .- mean_value) ./ std_value
    return normalized_data
end

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
    ll = (sum(logpdf.(Normal(0,1),flow(x))) + sum(ladj)) / nsamples # Ich rechne hier Minus statt plus !

    return -ll
end

function plot_spline(flow, samples)
    #p=plot(flat2batsamples(samp'),alpha=0.3)
    x1,x2 = quantile(samp[1,:],0.96), quantile(samp[1,:],0.04)
    p=plot(Shape([x1, x2, x2, x1], [-5.5,-5.5,5.5,5.5]), fillalpha=0.25, c=:yellow, linewidth=0.0, label=false, line=false)
    w,h,d = MonotonicSplines.get_params(flow.flow.fs[2].flow.fs[1].nn(flow.flow.fs[1](samples)[[false],:], 
                                        flow.flow.fs[2].flow.fs[1].nn_parameters, 
                                        flow.flow.fs[2].flow.fs[1].nn_state)[1], 1)
    x = inverse(flow.flow.fs[1])(reshape(w[:,1,1],1,length(w[:,1,1])))'
    #println(reshape(w[:,1,1],1,length(w[:,1,1])))
    y = h[:,1,1]                        
    plot!(x,y, seriestype = :scatter, label="Knots", legend =true,xlabel="Input value", ylabel="Output value")
    x=range(minimum(x)-0.5,stop=maximum(x)+0.5,length=10^4)
    y = reshape(flow(Matrix(reshape(x,1,10^4))),10^4,1)
    plot!(x,y, linewidth = 2.5, label="Spline function")
    plot!(x, ones(length(x))*-2, fillrange=ones(length(x))*2, fillalpha=0.25,c=:yellow ,label="2 sigma region of propability")
    ylims!(-5.5,5.5)
    #savefig("$path/spline.pdf")
    return p
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
    create_Plots(flow, samples, vali, path, 0, ani_spline, lr, meta, animation_ft=animation_ft)
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
            create_Plots(flow, samples, vali, path, i*epochs, ani_spline, round(lr,digits=6),meta,animation_ft=animation_ft)
    
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


function timed_training(iid::Matrix, samples::Matrix, path2::String, minibatches, epochs, target_logpdf; batchsize=1000, eperplot=ceil(Int,(length(samples)/batchsize)/100),
                    lr=5f-2, K = 8, flow = build_flow(samples./(std(samples)/2), [InvMulAdd, RQSplineCouplingModule(size(samples,1), K = K)]), lrf=1, vali = [])
    path = make_Path("$batchsize-$epochs-$minibatches-$lrf",path2)
    meta = plot_metadaten(path, length(samples),minibatches, epochs, batchsize, lr, K, lrf)
    mkdir("$path/Loss_vali")
    mkdir("$path/ftruth")
    mkdir("$path/spline")
    animation_ft = Animation()
    ani_spline = Animation()
    
    #AdaptiveFlows.optimize_flow_sequentially(samples[1:end,1:10],flow, Adam(lr),  # Precompile to be faster
    #                                            nbatches=1,nepochs=1, shuffle_samples=true);
    c = AdaptiveFlows.optimize_flow(samples[1:end,1:10],flow, Adam(lr),  # Precompile to be faster
                                                loss=AdaptiveFlows.negll_flow,
                                                logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
                                                nbatches=1,nepochs=1, shuffle_samples=true).loss_hist

    vali = [2.0]#validate_pushfwd_negll(flow,iid)]#[AdaptiveFlows.negll_flow_loss(flow.flow.fs[2],flow.flow.fs[1](iid),target_logpdf(flow.flow.fs[1](iid)),(f=target_logpdf,))]   
    create_Plots(flow, samples, vali, path, 0, ani_spline, lr, meta, animation_ft=animation_ft, vali=vali)
    lr=round(lr,digits=6)
    make_slide("$path/spline/$(nummer(0)).pdf","$path/metadaten.pdf",title="Algo(batchsize=$batchsize, epochs=0, lr=$lr)")  
    batches=round(Int,(size(samples,2)/batchsize))
    zeit = time()
    i=0
    loss = [2.0]#validate_pushfwd_negll(flow,samples)]#mvnormal_negll_flow(flow.flow.fs[2],flow.flow.fs[1](samples))]
    loss_hist=c
    while (time()-zeit) < 600
        i=i+1
        if i>batches
            i=1
        end
        println("$i von $batches Iterationen")
        #flow, opt_state, loss_hist = AdaptiveFlows.optimize_flow_sequentially(samples[1:end,((i-1)*batchsize)+1:i*batchsize],flow, Adam(lr), 
        #                                            nbatches=minibatches,nepochs=epochs, shuffle_samples=false);
        flow, opt_state, loss_hist =AdaptiveFlows.optimize_flow(samples[1:end,((i-1)*batchsize)+1:i*batchsize],flow, Adam(lr),  # Precompile to be faster
                    loss=AdaptiveFlows.negll_flow, #loss_history = loss_hist,
                    logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
                    nbatches=minibatches,nepochs=epochs, shuffle_samples=false);
        #for element in loss_hist[2][1]
        #    push!(loss, element)
        #end
        push!(loss, mean(loss_hist[2][1]))#validate_pushfwd_negll(flow,samples))#mvnormal_negll_flow(flow.flow.fs[2],flow.flow.fs[1](samples)))#element)
        push!(vali,  mean(loss_hist[2][1]))#validate_pushfwd_negll(flow,iid))

        if (true)#(i%eperplot) == 0
            create_Plots(flow, samples, loss, path, i*epochs, ani_spline, round(lr,digits=6),meta,animation_ft=animation_ft, vali=vali)
    
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
    # Mittelwert berechnen
    midloss = mean(vali[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    midloss2 = mean(loss[end - ceil(Int, 0.1 * length(vali)) + 1:end])
    file = open("$path2/2D.txt", "a")
    write(file, "$epochs, $lrf, $midloss, $midloss2\n")
    close(file)
    epochs=batches
    #gif(animation_ft, "$path/ftruth/transform.gif", fps=min(ceil(Int,(epochs/eperplot)/15),10))
    gif(ani_spline, "$path/spline/spline.gif", fps=3)#min(ceil(Int,(epochs/eperplot)/15),10))
    lr=round(lr,digits=6)
    make_slide("$path/ftruth/$(nummer(epochs)).pdf","$path/spline/$(nummer(epochs)).pdf","$path/Loss_vali/$(nummer(epochs)).pdf",
                title="Training(batchsize=$batchsize, epochs=$epochs, minib=$minibatches, lr=$lr)")

    #return flow, loss, lr
    return BAT.CustomTransform(flow)
end


function validate_pushfwd_negll(flow,samp)
    f=AdaptiveFlows.PushForwardLogDensity(FlowModule(InvMulAdd(I(size(samp,1)), 
        zeros(size(samp,1))), false), AdaptiveFlows.std_normal_logpdf)
    return AdaptiveFlows.negll_flow_loss(flow.flow.fs[2],flow.flow.fs[1](samp), f(flow.flow.fs[1](samp)), f)
end

function validate_pushfwd_KLDiv(flow,samp)
    f=AdaptiveFlows.PushForwardLogDensity(FlowModule(InvMulAdd(I(size(samp,1)), 
        zeros(size(samp,1))), false), AdaptiveFlows.std_normal_logpdf)
    #return AdaptiveFlows.KLDiv_flow_loss(flow.flow.fs[2],flow.flow.fs[1](samp), f(flow.flow.fs[1](samp)), f)
    x=flow.flow.fs[1](samp)
    logd_orig = f(flow.flow.fs[1](samp))
    nsamples = size(x, 2) 
    flow_corr = flow.flow.fs[2]#fchain(flow,f.f)
    #logpdf_y = logpdfs[2].logdensity
    y, ladj = with_logabsdet_jacobian(flow_corr, x)
    ll = target_logpdf(x) - vec(ladj)
    KLDiv = sum(exp.(ll) .* ((ll) - AdaptiveFlows.std_normal_logpdf(y))) / nsamples
end

function negll_flow_loss2(flow::F, x::AbstractMatrix{<:Real}, logpdf::Function) where F<:AbstractFlow
    nsamples = size(x, 2) 
    flow_corr = fchain(flow,logpdf.f)
    y, ladj = with_logabsdet_jacobian(flow_corr, x)
    ll = (sum(logpdf.logdensity(y)) + sum(ladj)) / nsamples
    return -ll
end

function train_flow2(samples::Matrix, path::String, minibatches, epochs, batchsize; 
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

    create_Plots(flow, samples, loss, path, 0, ani_spline, lr, meta, animation_ft=animation_ft)
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

        create_Plots(flow, samples, loss, path, eperplot*epoch, ani_spline, round(lr,digits=6), meta, animation_ft=animation_ft)
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
    #run(`python /net/e4-nfs-home.e4.physik.tu-dortmund.de/home/wweber/Slides/gif.py $path/ftruth`)
    #run(`python /net/e4-nfs-home.e4.physik.tu-dortmund.de/home/wweber/Slides/gif.py $path/spline`)   
    #gif(animation_ft, "$path/ftruth/transform.gif", fps=min(ceil(Int,(epochs/eperplot)/15),10))
    gif(ani_spline, "$path/spline/spline.gif", fps=10)#min(ceil(Int,(epochs/eperplot)/15),10))
    lr=round(lr,digits=6)
    make_slide("$path/ftruth/$(nummer(epochs)).pdf","$path/spline/$(nummer(epochs)).pdf","$path/Loss_vali/$(nummer(epochs)).pdf",
                title="Training(batchsize=$batchsize, epochs=$epochs, minib=$minibatches, lr=$lr)")

    return flow, loss, lr
end


function create_Plots(flow, samples, loss, path, epoch, ani_spline, lr, meta::Plots.Plot; vali=[0], animation_ft)
    x = 1:length(loss);
    l =round(vali[end],digits=4);
    plot(x, loss, label="Loss", xlabel="Epoch", ylabel="Loss", title="Loss, End=$l", left_margin = 9Plots.mm, bottom_margin=7Plots.mm);
    x = 1:length(vali);
    #l =round(vali[end],digits=4);
    #learn=plot!(x, vali, label="Loss");
    learn=ylims!(1.55,1.75);
    savefig("$path/Loss_vali/$(nummer(epoch)).pdf");
    f=plot_flow(flow,samples);
    #f=title!("$epoch epochs, lr = $lr");
    savefig("$path/ftruth/$(nummer(epoch)).pdf");
    #frame(animation_ft, f);
    ###plot_spline(flow,samples);
    s=title!("$epoch epochs lr= $lr");
    savefig("$path/spline/$(nummer(epoch)).pdf");
    a = plot(f,meta,layout=(1,2),size=(1200,450),margins=9Plots.mm)
    b = plot(learn,s,layout=(1,2),size=(1200,450),margins=9Plots.mm)
    frame(ani_spline, plot(a,b,layout=(2,1),size=(1200,900),margins=9Plots.mm));

    closeall();

    return nothing
end

function plot_metadaten(path, samplesize,minibatches, epochs, batchsize, lr, K, lrf)
    x=plot(size=(800, 600), legend=false, ticks=false, border=false, axis=false);
    annotate!(0.5,1-0.1,"Pfad: $(path[1:49])");
    annotate!(0.5,1-0.2,"$(path[50:end])");
    annotate!(0.5,1-0.3,"Samplesize: $samplesize");
    annotate!(0.5,1-0.4,"Knots: $K");
    annotate!(0.5,1-0.5,"Sample_batchsize: $batchsize, Train_batchsize: $(Int(batchsize/minibatches))");
    annotate!(0.5,1-0.6,"Epochs: $epochs");
    annotate!(0.5,1-0.7,"Start LR: $lr, LR-Factor: $lrf");
    savefig("$path/metadaten.pdf");
    return x
end

function plot_flow(flow,samples)
    p=plot(flat2batsamples(flow(samples)'))
    x_values = Vector(range(-5.5, stop=5.5, length=1000))
    f(x) = densityof(Normal(0,1.0),x)
    y_values = f.(x_values)
    plot!(x_values, y_values,density=true, linewidth=2.5,legend =:topright, label ="N(0,1)", color="black")
    ylims!(0,0.5)
    return p  
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