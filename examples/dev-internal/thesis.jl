context = BATContext(ad = ADModule(:ForwardDiff))
dims=3
path = make_Path("Thesis_dims=$dims-Ptg_tau_1")
tf=1.0
smallpeak=0.1
k=20
pretrafo=BAT.PriorToGaussian()

model=MixtureModel(Normal, [(-10.0, 1.0),(10.0, 1.0)], [smallpeak, 1-smallpeak])

post=get_posterior(model,dims)
marginal = get_posterior(model,1)
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
iid=BAT2Matrix(standard.result_trafo.v)[:,1:10^5]
plot_samples(path,iid,marginal,name="traindata.pdf")

flow = build_flow(iid, [InvMulAdd, RQSplineCouplingModule(size(iid,1), K = k)])
x= train(iid[:,1:3],target_logpdf)
x = @time train(iid,target_logpdf,epochs=100,batches=4,flow=flow,shuffle=true)
flow = x.result
plot_loss_alldimension(path,x.loss_hist[2][1])
plot_flow_alldimension(path,flow,iid,1)
saveFlow(path,flow)
# minibatches=1
# epochs=1
# flow, loss = tempertrain(model,iid, path, minibatches, epochs)
# plot_loss_alldimension(path,loss)
# plot_flow_alldimension(path,flow,iid,0)


# ensemble = @time FlowSampling(path, post, use_mala=false, n_samp=10^4,Knots=k, pretrafo=pretrafo,
#                                 marginaldistribution=marginal, identystart=false, flow=flow,
#                                 tuner=BAT.TransformedMCMCNoOpTuning(),burnin=100,walker=1000)
# plot_samples(path,BAT2Matrix(ensemble.result.v),marginal)


walker=1000
n_samp = 10^6#length(standard.v)
inburn = 1000

ensemble= FlowSampling(make_Path("Hilfe3",path), post, use_mala=false, n_samp=n_samp,Knots=20, walker=walker,
                                    marginaldistribution=marginal, identystart=false, flow=flow, dims=dims,
                                    tuner=BAT.TransformedMCMCNoOpTuning(),burnin=inburn,pretrafo=pretrafo)

# ensemble = BAT.bat_sample_impl(post, 
#                                 TransformedMCMCSampling(pre_transform=pretrafo, 
#                                                         init=TransformedMCMCEnsemblePoolInit(),
#                                                         tuning_alg=BAT.TransformedMCMCNoOpTuning(), 
#                                                         tau=0.5, nsteps=Int(n_samp/walker),#+burnin-1,
#                                                         adaptive_transform=BAT.CustomTransform(flow), use_mala=false,
#                                                         nwalker=walker),
#                                                         context);
samp = BAT2Matrix(ensemble.result.v)
plot_samples(path,samp,marginal,name="flowresult_with_burnin.pdf")
# 
# 
# standard2 = bat_sample(post, TransformedMCMCSampling(nsteps = 10^3, nchains = 4,strict=false)).result