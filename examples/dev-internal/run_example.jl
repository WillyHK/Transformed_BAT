println(Base.Threads.nthreads())

ENV["GKSwstype"] = "nul"
include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/dev-internal/transformed_example.jl")
include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/ExamplePosterior.jl")
gr()  

dims = 1
Knots = 20
walker=1000
inburn=100
n_samp=2*10^6
testname = "test1d_$dims-Dimensionen-NoOPTuning_BachelorDistribution_$Knots-K_$(n_samp/(1000)^2)_M_samples+"
println("################################################################")
println("# $testname")
println("################################################################")
path = make_Path("$testname")
peaks=3
#distri = get_triplemode(dims, peaks=peaks)
distri = get_bachelor(dims) # get_rings2d()# 
marginal = get_bachelor(1)
context = BATContext(ad = ADModule(:ForwardDiff))
density, trafo = BAT.transform_and_unshape(BAT.PriorToGaussian(), distri, context)
target_logpdf = x -> BAT.checked_logdensityof(density).(x)


nuridentflow = FlowSampling(make_Path("$testname/notrain_$dims"), distri, use_mala=false, n_samp=n_samp,Knots=Knots,
                                    marginaldistribution=marginal, identystart=true, 
                                    tuner=BAT.TransformedMCMCNoOpTuning(),burnin=inburn,pretrafo=BAT.DoNotTransform())


minibatches=4
epochs=100
samp = BAT2Matrix(nuridentflow.result_trafo.v)[:,rand(1:length(nuridentflow.result_trafo.v),10^6)]
#saveSamples(path,samp,name="samples_identy.jls")


samp = BAT2Matrix(bat_sample(MixtureModel(Normal, [(-10.0, 1.0), (0.0, 1.0), (10.0, 1.0)], [0.1, 0.899, 0.001]),BAT.IIDSampling()).result.v)
#plot(flat2batsamples(samp'), density=true,right_margin=9Plots.mm)
#title!("$(size(samp))")
#savefig("$path/traindata.pdf")
#flow, mull = tempering_test(samp,make_Path("$testname/temp_train"),minibatches,epochs, K=20)#,flow2=nuridentflow.flow)


flow, opt_state, loss_hist = train(samp, target_logpdf, flow= build_flow(samp./2, [InvMulAdd, RQSplineCouplingModule(size(samp,1), K = Knots)]),#nuridentflow.flow, K=20,
                                    batches=minibatches, epochs=epochs, opti=Adam(0.005), shuffle=true)
plot_flow_alldimension(path,flow,samp,1)
for i in 1:dims
    plot_loss_alldimension(path,loss_hist[2][i],name="loss_$i.pdf")
end
saveFlow(path,flow)

trainedflow = FlowSampling(make_Path("$testname/flowtrain_$dims"), distri, use_mala=false, n_samp=n_samp,Knots=Knots,
                                    marginaldistribution=marginal, flow=flow, identystart=false, 
                                    tuner=BAT.TransformedMCMCNoOpTuning(),burnin=inburn,pretrafo=BAT.DoNotTransform())
samp = BAT2Matrix(trainedflow.result_trafo.v)
#saveSamples(path,samp,name="samples_flow.jls")



ENDE

#mcmc=test_MCMC(distri,n_samp, simulate_walker=false)
#plot_samples(make_Path("$testname/mcmc"), mcmc,marginal)

density, trafo = BAT.transform_and_unshape(BAT.PriorToGaussian(), distri, context)
s = cholesky(Positive, BAT._approx_cov(density)).L
noFlow = BAT.CustomTransform(Mul(s))

my_result = BAT.bat_sample_impl(distri, TransformedMCMCSampling(pre_transform=PriorToGaussian(), tuning_alg=TransformedAdaptiveMHTuning(), nchains=4, nsteps=4*1000, adaptive_transform=noFlow), context).result
println(my_result)


ENDE
#SplineWatch_FlowSampling(path, distri,Knots=5,n_samp=200000)
#SplineWatch_FlowSampling(path, distri,Knots=10,n_samp=200000)
#SplineWatch_FlowSampling(path, distri,Knots=20,n_samp=200000)
#SplineWatch_FlowSampling(path, distri,Knots=40,n_samp=200000)

plot(flat2batsamples(samp'), density=true,right_margin=9Plots.mm)
title!("$(size(samp))")
if (dims == 1)
    x_values = Vector(range(minimum(samp), stop=maximum(samp), length=1000))
    y(x) = densityof(BAT.transform_and_unshape(DoNotTransform(), distri, context)[1],[x])
    y_values = y.(x_values)
    factor = distri.prior.bounds.vol.hi[1]-distri.prior.bounds.vol.lo[1]
    plot!(x_values, y_values*factor,density=true, linewidth=3.2,legend =:bottom, label ="truth", color="black")
end
savefig("$path/traindata_flow_nomala.pdf")
#make_slide("$path/traindata.pdf",title="Trainingsdata $n_samp iid samp, lr konst")
ENDE
target_logpdf = x-> BAT.checked_logdensityof(posterior).(x)

test=[]
nwalker=100
while (length(test)<nwalker)
    t=BAT.mcmc_init!(BAT.TransformedMCMCSampling(),
        posterior, 100,
        BAT.TransformedMCMCChainPoolInit(),
        BAT.TransformedMCMCNoOpTuning(), # TODO: part of algorithm? # MCMCTuner
        true,
        x->x,
        context)[1]
    chains = getproperty.(t, :samples)
    for c in chains
        push!(test,c.v[end])
    end
end
#samp = Matrix{Float64}(test[1:nwalker]')


function iterate_FlowSampling(sampPerRun=100000)
    x=FlowSampling(path, distri,use_mala=false,n_samp=sampPerRun,Knots=Knots)
    samp = BAT2Matrix(x.result.v)
    flow = x.flow
    i=0
    while (size(samp,2)<n_samp) # TODO hier wird leider noch jedes mal die lr zurückgesetzt #evtl Plots einfach in BAT.jl machen?
        path2=make_Path("$testname/$i")
        x=FlowSampling(path2, distri,use_mala=false,n_samp=sampPerRun,flow=flow)
        samp = hcat(samp,BAT2Matrix(x.result.v))
        plot_samples(path2, samp,get_triplemode(1))
        flow = x.flow
        i=i+1
    end
end


function test_sampling()
    iters = 5
    use_mala=true
    tuning=MCMCFlowTuning()
    tau=0.01
    nchains=[1]
    nsteps=[100]
    nwalker=[1000]
    for nwalk in nwalker
        for nchain in nchains
            for nstep in nsteps
                testname2 = "$testname/$nchain-$nwalk-$nstep"
                p = make_Path(testname2)

                iterative_test(posterior, iters, p, samp, 
                                use_mala=use_mala,
                                tuning=tuning,
                                tau=tau,
                                nchains=nchain,
                                nsteps=nstep,
                                nwalker=nwalk)
            end
        end
    end
end

function test_algorithmus(samples)
    nepochs=[3,5,10]
    n_minibatch=[1]
    batchsizes = [1000]
    lern = [0.995, 0.985]
    for size in batchsizes
        for batches in n_minibatch
            for nepoch in nepochs
                for lrf in lern
                    #path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
                    @time train_flow(samples,path, batches, nepoch,lr=0.01, batchsize=size,K=Knots,lrf=lrf);
                end
            end
        end
    end
    size=1000
    batches=2
    nepochs=10
    lrf=0,98
    path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
    @time train_flow(samples,path, batches, nepoch,lr=0.01, batchsize=size,K=K,lrf=lrf);
end


function test_training(samples)
    nepochs=[300]
    n_minibatch=[1]
    batchsizes = [size(samples,2)]
    lern = [1, 0.995, 0.99, 0.985, 0.98]
    for size in batchsizes
        for batches in n_minibatch
            for nepoch in nepochs
                for lrf in lern
                    path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
                    @time train_flow2(samp,path, batches, nepoch,size,lr=0.01, K=Knots,lrf=lrf);
                end
            end
        end
    end
end

function test_algorithmus2(samples,nepoch,lrf;runtime=600)
    batches=1
    size=1000

   timed_training(iid,samples,path, batches, nepoch,target_logpdf,lr=0.01,batchsize=size,K=Knots,lrf=lrf,runtime=runtime)#,loss_f=AdaptiveFlows.KLDiv_flow);
end

function test_batchwise(samples,nepoch,lrf;batches=10)
    K=30
    minibatches=1
    size=1000
    return batchwise_training(iid,samples,path, minibatches, nepoch,target_logpdf, #loss_f=AdaptiveFlows.KLDiv_flow,
                              lr=0.01,batchsize=size,K=K,lrf=lrf,batches=batches);
end

function test_Knots(samples,nepoch,lrf)
    batches=1
    size=1000
   # path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
   flow, opt_state, loss_hist =AdaptiveFlows.optimize_flow(samples[1:end,((i-1)*batchsize)+1:i*batchsize],flow, Adam(lr),  # Precompile to be faster
                loss=AdaptiveFlows.negll_flow, #loss_history = loss_hist,
                logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
   nbatches=minibatches,nepochs=epochs, shuffle_samples=false);
    @time timed_training(iid,samples,path, batches, nepoch,target_logpdf,lr=0.01, batchsize=size,K=Knots,lrf=lrf);
end
#test_training(samp)
#test_algorithmus2(samp,parse(Int, ARGS[2]),parse(Float64, ARGS[1]))
#knot_gif(path,samp,target_logpdf,bis=1)
#@time test_algorithmus2(samp,1,0.995,runtime=120)
@time flow,vali = test_batchwise(samp,10,0.99,batches=2)

#file = open("$path/2D.txt", "a")
#    write(file, "Epoch, lrf,vali, train\n")
#close(file)
for epochs in [3,5,7]
    for lrf in [0.99 0.97 0.95]
        #test_algorithmus2(samp,epochs,lrf,runtime=1200)
        test_batchwise(samp,epochs,lrf,batches=500)
    end
end


println("ende")

#bild= "/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/wweber/Pictures/dpg2.jpg"
#makeBild(bild, path)


function runTestFlow(x::Int64)
    directory = "/ceph/groups/e4/users/wweber/private/Master/Plots/tryCeph"
    lr = 4f-3
    nknots = 40
    nbatches = 10
    nepochs = 10
    nneuron = 10
    nhidden = 2
    wanna_use_GPU = false
    if x == 1
        testname = "2D_mix"
        x_limit = 10 #rings=2 # for plotting
        posterior=get_mix(2)

        test(posterior, directory, testname, lr, nknots, nbatches, nepochs, nneuron, nhidden, wanna_use_GPU, x_limit)
        println("Test $x durchgelaufen")
    elseif x == 2
        testname = "4D_mix"
        x_limit = 10 # for plotting
        posterior= get_mix(4)

        test(posterior, directory, testname, lr, nknots, nbatches, nepochs, nneuron, nhidden, wanna_use_GPU, x_limit)
        println("Test $x durchgelaufen")
    elseif x == 3
        testname = "7D_mix"
        x_limit = 10 # for plotting
        posterior= get_mix(7)

        test(posterior, directory, testname, lr, nknots, nbatches, nepochs, nneuron, nhidden, wanna_use_GPU, x_limit)
        println("Test $x durchgelaufen")
    elseif x == 4
        testname = "20D_mix"
        x_limit = 10 # for plotting
        posterior= get_mix(20)

        test(posterior, directory, testname, lr, nknots, nbatches, nepochs, nneuron, nhidden, wanna_use_GPU, x_limit)
        println("Test $x durchgelaufen")
    elseif x == 5
        testname = "10D_mix"
        x_limit = 10 # for plotting
        posterior= get_mix(10)

        test(posterior, directory, testname, lr, nknots, nbatches, nepochs, nneuron, nhidden, wanna_use_GPU, x_limit)
        println("Test $x durchgelaufen")
    else
        println("Testnumber $x not known!")
    end
end
