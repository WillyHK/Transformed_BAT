println(Base.Threads.nthreads())

include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/dev-internal/transformed_example.jl")
include("/ceph/groups/e4/users/wweber/private/Master/Code/Transformed_BAT/examples/ExamplePosterior.jl")
    
testname = "Animation2"#"testcase"
distri = get_triplemode(1)#testcase(1)
posterior=distri
context = BATContext(ad = ADModule(:ForwardDiff))
posterior, trafo = BAT.transform_and_unshape(BAT.DoNotTransform(), distri, context)
path = make_Path(testname)

iid=Matrix(rand(MixtureModel(Normal, [(-3,1.0),(0,1.0),(3,1.0)],[1/3,1/3,1/3]),10^6)')
#mcmc=test_MCMC(posterior,path)
n_samp=500000
samp=iid[1:end,1:n_samp]
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


savefig("$path/traindata.pdf")
make_slide("$path/traindata.pdf",title="Trainingsdata $n_samp iid samp, lr konst")

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

function test_training(samples)
    nepochs=[5,100]#,20,80]
    n_minibatch=[1]#,10]#,10]#,100]
    batchsizes = [1000]
    lern = [0.01,0.1]#,0.001, 0.1]
    for size in batchsizes
        for batches in n_minibatch
            for nepoch in nepochs
                for lr in lern
                    path = make_Path("$testname/lrstieg/$size-$nepoch-$batches")
                    @time train_flow(samples,path, batches, nepoch,size,lr=lr)
                    #path = make_Path("$testname/lrstieg2/$size-$nepoch-$batches")
                    #@time train_flow2(samples,path, batches, nepoch,length(samples),lr=lr)#size)
                end
            end
        end
    end
end

#test_training(samp)
K=20
@time flow, loss, lr = train_flow2(samp,path, 1, 200,length(samp),lr=0.01, K=K,lrf=1,eperplot=10);
#@time flow,loss,lr = train_flow2(samp,path, 1, 5,length(samp),lr=0.001,K=K,flow=flow,loss=loss,eperplot=1);
#@time train_flow(samp,path, 1, 1,lr=0.01, batchsize=1000,K=K,lrf=0.99,eperplot=50);
#@time train_flow(samp,path, 1, 5,lr=0.01, batchsize=1000,K=K,lrf=1,eperplot=50);
#@time train_flow(samp,path, 1, 5,lr=0.01, batchsize=1000,K=K,lrf=0.99,eperplot=50);

#K=40
#@time train_flow(samp,path, 1, 1,lr=0.01, batchsize=1000,K=K,lrf=1,eperplot=50);
#@time train_flow(samp,path, 1, 5,lr=0.01, batchsize=1000,K=K,lrf=1,eperplot=50);
#@time train_flow(samp,path, 1, 5,lr=0.01, batchsize=1000,K=K,lrf=0.99,eperplot=50);
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

