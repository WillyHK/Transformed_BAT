println(Base.Threads.nthreads())

ENV["GKSwstype"] = "nul"
include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/dev-internal/transformed_example.jl")
include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/ExamplePosterior.jl")
gr()  

testname = "mcmc-studie"
path = make_Path("$testname")
context = BATContext(ad = ADModule(:ForwardDiff))
dims = 1
Knots = 20
#distri = get_testcase(dims)
distri = get_triplemode(dims)

posterior, trafo = BAT.transform_and_unshape(BAT.DoNotTransform(), distri, context)
target_logpdf = x-> BAT.checked_logdensityof(posterior).(x)
n_samp=500000

iid = get_iid([-3,0,3],dims,n_samp)
samp=iid

# TODO KLDiv_flow gibt leider irgendwie regelmäßig negative Werte und trainiert auch nicht richtig :(
# a,b,c=train(samp,target_logpdf,loss=AdaptiveFlows.KLDiv_flow,epochs=10,batches=10)
# plot_loss_alldimension(path,c[2][1])

if dims < 5
    flow = build_flow(samp./(std(samp)/2), [InvMulAdd, RQSplineCouplingModule(dims, K = Knots)])
    mcmc, flow=EnsembleSampling(posterior,BAT.CustomTransform(flow),nsteps=Int(n_samp/1000)+1,nwalker=1000, 
                                tuning=MCMCFlowTuning(),use_mala=true); # Altenative: TransformedMCMCNoOpTuning()
    samp=mcmc

    #mcmc=test_MCMC(posterior,n_samp)#,path)
    #samp=mcmc[1:end,1:n_samp]
end

plot(flat2batsamples(samp'), density=true,right_margin=9Plots.mm)
title!("$(size(samp)), mala, flow")
if (dims == 1)
    x_values = Vector(range(minimum(samp), stop=maximum(samp), length=1000))
    y(x) = densityof(posterior,[x])
    y_values = y.(x_values)
    factor = posterior.prior.bounds.vol.hi[1]-posterior.prior.bounds.vol.lo[1]
    plot!(x_values, y_values*factor,density=true, linewidth=3.2,legend =:topright, label ="truth", color="black")
end
savefig("$path/traindata.pdf")
#make_slide("$path/traindata.pdf",title="Trainingsdata $n_samp iid samp, lr konst")


BEENDE

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
    K=20
    nepochs=[3,5,10]
    n_minibatch=[1]
    batchsizes = [1000]
    lern = [0.995, 0.985]
    for size in batchsizes
        for batches in n_minibatch
            for nepoch in nepochs
                for lrf in lern
                    #path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
                    @time train_flow(samples,path, batches, nepoch,lr=0.01, batchsize=size,K=K,lrf=lrf);
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
    K=20
    nepochs=[300]
    n_minibatch=[1]
    batchsizes = [size(samples,2)]
    lern = [1, 0.995, 0.99, 0.985, 0.98]
    for size in batchsizes
        for batches in n_minibatch
            for nepoch in nepochs
                for lrf in lern
                    path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
                    @time train_flow2(samp,path, batches, nepoch,size,lr=0.01, K=K,lrf=lrf);
                end
            end
        end
    end
end

function test_algorithmus2(samples,nepoch,lrf;runtime=600)
    K=20
    batches=1
    size=1000

   timed_training(iid,samples,path, batches, nepoch,target_logpdf,lr=0.01,batchsize=size,K=K,lrf=lrf,runtime=runtime)#,loss_f=AdaptiveFlows.KLDiv_flow);
end

function test_batchwise(samples,nepoch,lrf;batches=10)
    K=20
    minibatches=1
    size=1000
    return batchwise_training(iid,samples,path, minibatches, nepoch,target_logpdf,lr=0.01,batchsize=size,K=K,lrf=lrf,batches=batches)#,loss_f=AdaptiveFlows.KLDiv_flow);
end

function test_Knots(samples,nepoch,lrf)
    K=20
    batches=1
    size=1000
   # path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
   flow, opt_state, loss_hist =AdaptiveFlows.optimize_flow(samples[1:end,((i-1)*batchsize)+1:i*batchsize],flow, Adam(lr),  # Precompile to be faster
                loss=AdaptiveFlows.negll_flow, #loss_history = loss_hist,
                logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
   nbatches=minibatches,nepochs=epochs, shuffle_samples=false);
    @time timed_training(iid,samples,path, batches, nepoch,target_logpdf,lr=0.01, batchsize=size,K=8,lrf=lrf);
end
#test_training(samp)
#test_algorithmus2(samp,parse(Int, ARGS[2]),parse(Float64, ARGS[1]))
#knot_gif(path,samp,target_logpdf,bis=1)
#@time test_algorithmus2(samp,1,0.995,runtime=120)
@time flow,vali = test_batchwise(samp,10,0.99,batches=2)

#file = open("$path/2D.txt", "a")
#    write(file, "Epoch, lrf,vali, train\n")
#close(file)
for epochs in [5, 9, 13, 17, 21]
    for lrf in [0.975 0.97 0.965 0.96 0.955 0.95 0.945]
        #test_algorithmus2(samp,epochs,lrf,runtime=1200)
        test_batchwise(samp,epochs,lrf,batches=100)
    end
end


println("ende")

#bild= "/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/wweber/Pictures/dpg2.png"
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
