context = BATContext(ad = ADModule(:ForwardDiff))
dims=5
mala = true
path = make_Path("Mixture_dims=$dims-Mala")
tf=1.0
smallpeak=0.01
k=20
pretrafo=BAT.PriorToGaussian()


model=MixtureModel(Normal, [(-15.0, 0.1),(0.0,0.2),(15.0, 1.0)], [smallpeak, 5*smallpeak, 1-smallpeak*6])

post= get_posterior(model,dims,tf=tf)
marginal = get_posterior(model,1,tf=tf)
post, trafo = BAT.transform_and_unshape(BAT.DoNotTransform(), post, context)
marginal, trafo2 = BAT.transform_and_unshape(BAT.DoNotTransform(), marginal, context)
target_logpdf = x -> BAT.checked_logdensityof(post).(x)

standard = @time bat_sample(post, MCMCSampling(nsteps = 10^6, nchains = 4,strict=false,
                                trafo=pretrafo, 
                                init=MCMCChainPoolInit(init_tries_per_chain=BAT.ClosedInterval(4,128)),
                                burnin=MCMCMultiCycleBurnin()),      
                                context)
samp = BAT2Matrix(standard.result.v)
plot_samples(path,samp,marginal)

#iid = BAT2Matrix(rand(get_prior(model,dims),10^5))
#iid=BAT2Matrix(standard.result_trafo.v)
iid = rand(MvNormal(zeros(dims),ones(dims)*1.2),10^5) # ohne mal 1.2 manchmal flow zu klein !!!!!!!!!!

walker=1000
n_samp = length(standard.result.v)
inburn = 1000


ensemble= FlowSampling(make_Path("ensemble_without_flow",path), post, use_mala=mala, n_samp=n_samp,Knots=20, walker=walker,
                                marginaldistribution=marginal, identystart=false, dims=dims, 
                                flow = build_flow(iid, [InvMulAdd, RQSplineCouplingModule(size(iid,1), K = k)]),
                                tuner=BAT.TransformedMCMCNoOpTuning(),burnin=inburn,pretrafo=pretrafo)


# flow = nothing
# flowpath = "/ceph/groups/e4/users/wweber/private/Master/Flows/Problem_1"
# if isfile("$flowpath/flow_$dims.jls")
#     flow = loadFlow(flowpath, dims)
# else
#     plot_samples(path,iid,marginal,name="traindata.pdf")
#     
#     flow = build_flow(iid, [InvMulAdd, RQSplineCouplingModule(size(iid,1), K = k)])
#     x= train(iid[:,1:3],target_logpdf)
#     x = @time train(iid[:,rand(1:length(standard.result_trafo.v),5*10^5)],target_logpdf,epochs=100,batches=4,flow=flow,shuffle=true)
#     flow = x.result
#     plot_loss_alldimension(path,x.loss_hist[2][1])
#     plot_flow_alldimension(path,flow,iid,1)
#     saveFlow(flowpath,flow,name="flow_$dims.jls")
# end
# 
# ensemble= FlowSampling(make_Path("const_flow",path), post, use_mala=mala, n_samp=n_samp,Knots=20, walker=walker,
#                                     marginaldistribution=marginal, identystart=false, flow=flow, dims=dims,
#                                     tuner=BAT.TransformedMCMCNoOpTuning(),burnin=inburn,pretrafo=pretrafo)
# 
# samp = BAT2Matrix(ensemble.result.v)
# plot_samples(path,samp,marginal,name="flowresult_with_burnin.pdf")


#ensemble= FlowSampling(make_Path("train_flow",path), post, use_mala=mala, n_samp=n_samp,Knots=20, walker=walker,
#                                marginaldistribution=marginal, identystart=false, flow=flow, dims=dims,
#                                tuner=BAT.MCMCFlowTuning(),burnin=inburn,pretrafo=pretrafo)
                                

ensemble= FlowSampling(make_Path("train_new_flow",path), post, use_mala=mala, n_samp=n_samp,Knots=20, walker=walker,
                                marginaldistribution=marginal, identystart=false, dims=dims, 
                                flow = build_flow(iid, [InvMulAdd, RQSplineCouplingModule(size(iid,1), K = k)]),
                                tuner=BAT.MCMCFlowTuning(),burnin=inburn,pretrafo=pretrafo)








# ensemble = BAT.bat_sample_impl(post, 
#                                 TransformedMCMCSampling(pre_transform=pretrafo, 
#                                                         init=TransformedMCMCEnsemblePoolInit(),
#                                                         tuning_alg=BAT.TransformedMCMCNoOpTuning(), 
#                                                         tau=0.5, nsteps=Int(n_samp/walker),#+burnin-1,
#                                                         adaptive_transform=BAT.CustomTransform(flow), use_mala=mala,
#                                                         nwalker=walker),
#                                                         context);

# 
# 
# standard2 = bat_sample(post, TransformedMCMCSampling(nsteps = 10^3, nchains = 4,strict=false)).result