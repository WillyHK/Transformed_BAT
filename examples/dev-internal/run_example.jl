println(Base.Threads.nthreads())

include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/dev-internal/transformed_example.jl")
include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/ExamplePosterior.jl")
    
testname = "testcase"
distri = get_testcase(1)

context = BATContext(ad = ADModule(:ForwardDiff))
posterior, trafo = BAT.transform_and_unshape(PriorToGaussian(), distri, context)
path = make_Path(testname)
mcmc=test_MCMC(posterior,path)

#f= BAT.CustomTransform(build_flow(mcmc))
#mala, flow=EnsembleSampling(posterior,f);
#plot(flat2batsamples(mala'))


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

                iterative_test(posterior, iters, p, mcmc, 
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

function test_training(samples)
    nepochs=[1,10,100]
    n_minibatch=[1,10,100]
    batchsizes = [1000, 100]
    for size in batchsizes
        for batches in n_minibatch
            for nepoch in nepochs
                path = make_Path("$testname/lrstieg/$size-$nepoch-$batches")
                @time train_flow(samples,path, batches, nepoch,size)
            end
        end
    end
end

test_training(mcmc)#[1:end,1:200001])

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

# @time begin # mag das Cluster nicht so
#     using Base.Threads
#     Threads.@threads for i in 4:4
#         runTest(i)
#     end
# end
#@time runTestFlow(4)
#@time runTestFlow(5)
#@time runTestFlow(3)
#@time runTestFlow(2)
#@time runTestFlow(1)
#@time runTestFlow(parse(Int,ARGS[1]))

