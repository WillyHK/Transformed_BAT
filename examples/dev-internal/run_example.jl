println(Base.Threads.nthreads())

ENV["GKSwstype"] = "nul"
include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/dev-internal/transformed_example.jl")
include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/ExamplePosterior.jl")
gr()  

testname = "15m_2D"#Animation_Algorithmus_MCMC_1000_Samples_Descent_timed"
distri = get_triplemode(1)
posterior=distri
context = BATContext(ad = ADModule(:ForwardDiff))
posterior, trafo = BAT.transform_and_unshape(BAT.DoNotTransform(), distri, context)
#path = make_Path(testname)

n_samp=500000
iid=Matrix(rand(MixtureModel(Normal, [(-3,1.0),(0,1.0),(3,1.0)],[1/3,1/3,1/3]),n_samp)')
mcmc=test_MCMC(posterior,n_samp)#,path)
samp=mcmc[1:end,1:n_samp]
#samp2=iid[1:end,1:n_samp]
#ff= BAT.CustomTransform(build_flow(samp))
#mala, flow=EnsembleSampling(posterior,ff,nsteps=Int(n_samp/1000)+1,nwalker=100, tuning=TransformedMCMCNoOpTuning(),use_mala=false);
#plot(flat2batsamples(mala'))
#samp=Matrix(mala)[1:end,1:n_samp+1]
#samp=iid[1:end,rand(1:length(iid),n_samp)]#[1:end,1:100001]

x_values = Vector(range(minimum(samp), stop=maximum(samp), length=1000))
y(x) = densityof(posterior,[x])
y_values = y.(x_values)
plot(flat2batsamples(samp'), density=true,right_margin=9Plots.mm)
plot!(x_values, y_values*20,density=true, linewidth=3.2,legend =:bottomright, label ="truth", color="black")
#plot!(flat2batsamples(iid2), linetype = :steppre)


#savefig("$path/traindata.pdf")
#make_slide("$path/traindata.pdf",title="Trainingsdata $n_samp iid samp, lr konst")

#plot(flat2batsamples(iid))

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
                    path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
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

function test_algorithmus2(samples,nepoch,lrf)
    K=20
    batches=1
    size=1000
    path = make_Path("$testname/$size-$nepoch-$batches-$lrf")
    @time timed_training(iid,samples,path, batches, nepoch,lr=0.01, batchsize=size,K=K,lrf=lrf);
end

#test_training(samp)
test_algorithmus2(samp,parse(Int, ARGS[2]),parse(Float64, ARGS[1]))

println("ende")

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
