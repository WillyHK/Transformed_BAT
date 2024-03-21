using IntervalSets   # for specifying the prior
include("/ceph/groups/e4/users/wweber/private/Master/Code/EFT/EFTfitter.jl/src/EFTfitter.jl")

area = 2
parameters = BAT.distprod(
    ctW =  -area..area, 
    ctZ =  -area..area,
    ctl1 =  -area..area,
    cte1 = -area..area,
    cQe1 =  -area..area,
)

function AFB_t_ctW(params)
    c = [0.7568082758654278, 1.5947561013056617, 0.8911031800088027, 0.5270755684996008, 1.3352168287626491, 0.8980806408222466]
    return ((c[1] + c[2] * params.ctW + c[3] * params.ctW^2) - (c[4] + c[5] * params.ctW + c[6] * params.ctW^2)) / ((c[1] + c[2] * params.ctW + c[3] * params.ctW^2) + (c[4] + c[5] * params.ctW + c[6] * params.ctW^2))
end

function AFB_t_ctZ(params)
    c = [1.0820998619073685, -1.8076713006052538, 1.0535769677010902, 0.7533246917811985, -1.5021686755762658, 1.065838516052281]
    return ((c[1] + c[2] * params.ctZ + c[3] * params.ctZ^2) - (c[4] + c[5] * params.ctZ + c[6] * params.ctZ^2)) / ((c[1] + c[2] * params.ctZ + c[3] * params.ctZ^2) + (c[4] + c[5] * params.ctZ + c[6] * params.ctZ^2))
end

function AFB_t_ctl1(params)
    c = [1.6618182057826034, -1.4094327599412346, 0.4888110407592913, 1.1586293710230111, -1.700211565368665, 0.9515048559818883]
    return ((c[1] + c[2] * params.ctl1 + c[3] * params.ctl1^2) - (c[4] + c[5] * params.ctl1 + c[6] * params.ctl1^2)) / ((c[1] + c[2] * params.ctl1 + c[3] * params.ctl1^2) + (c[4] + c[5] * params.ctl1 + c[6] * params.ctl1^2))
end

function AFB_t_cte1(params)
    c = [1.8643171960599245, -1.5493880694921662, 1.0467870335888683, 1.3008441681203575, -0.8933121368324438, 0.5395038783756824]
    return ((c[1] + c[2] * params.cte1 + c[3] * params.cte1^2) - (c[4] + c[5] * params.cte1 + c[6] * params.cte1^2)) / ((c[1] + c[2] * params.cte1 + c[3] * params.cte1^2) + (c[4] + c[5] * params.cte1 + c[6] * params.cte1^2))
end

function AFB_t_cQe1(params)
    c = [1.8094631569064132, -1.015457732647517, 0.5333123087005521, 1.253476723614385, -1.1612167257971293, 1.0483132671886215]
    return ((c[1] + c[2] * params.cQe1 + c[3] * params.cQe1^2) - (c[4] + c[5] * params.cQe1 + c[6] * params.cQe1^2)) / ((c[1] + c[2] * params.cQe1 + c[3] * params.cQe1^2) + (c[4] + c[5] * params.cQe1 + c[6] * params.cQe1^2))
end


measurements = (
    Meas1 = EFTfitter.Measurement(AFB_t_ctW, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
    Meas2 = EFTfitter.Measurement(AFB_t_ctZ, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
    Meas3 = EFTfitter.Measurement(AFB_t_ctl1, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
    Meas4 = EFTfitter.Measurement(AFB_t_cte1, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
    Meas5 = EFTfitter.Measurement(AFB_t_cQe1, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
)

function plot_physics()
    path2 = make_Path("Physiktheorie")
    x_values = [parameters.ctW.a:0.001:parameters.ctW.b,
                parameters.ctZ.a:0.001:parameters.ctZ.b,
                parameters.ctl1.a:0.001:parameters.ctl1.b,
                parameters.cte1.a:0.001:parameters.cte1.b,
                parameters.cQe1.a:0.001:parameters.cQe1.b]

    funcs = [x-> AFB_t_ctW((ctW=x,)),
             x-> AFB_t_ctZ((ctZ=x,)),
             x-> AFB_t_ctl1((ctl1=x,)),
             x-> AFB_t_cte1((cte1=x,)),
             x-> AFB_t_cQe1((cQe1=x,))]

    for i in 1:5
        plot(x_values[i],funcs[i],label="function")
        y_values = measurements[i].value*ones(length(x_values[i]))
        error_values = ones(length(x_values[i]))*measurements[i].uncertainties.stat
        plot!(x_values[i], y_values, label="value", color=:blue)
        plot!(fill_between = (x_values[i], y_values .- error_values, y_values .+ error_values), alpha=0.3, label="uncertaintie")
        savefig("$path2/Variable$i.pdf")
    end
end

correlations = (
    stat = EFTfitter.NoCorrelation(active=true),
)

model = EFTfitter.EFTfitterModel(parameters, measurements, correlations)

posterior = PosteriorMeasure(model)

algorithm = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^6, nchains = 4, strict = false)
samples = bat_sample(posterior, algorithm).result;

plot(samples)



###########################
#### Hier ansetzen
###########################
context = BATContext(ad = ADModule(:ForwardDiff))
dims=length(measurements)
mala = false
path = make_Path("Physik_$dims-80k")
savefig("$path/MCMC.pdf")
tf=1.0
smallpeak=0.01
k=80
pretrafo=BAT.PriorToGaussian()

post=posterior

standard = @time bat_sample(post, MCMCSampling(nsteps = 10^6, nchains = 4,strict=false,
                                trafo=pretrafo, 
                                init=MCMCChainPoolInit(init_tries_per_chain=BAT.ClosedInterval(4,128)),
                                burnin=MCMCMultiCycleBurnin()),      
                                context)
plot(standard.result)
savefig("$path/MCMC2.pdf")

iid = rand(MvNormal(zeros(dims),ones(dims)*1.2),10^5) # ohne mal 1.4 manchmal flow zu klein !!!!!!!!!!
walker=1000
n_samp = 5*10^6#length(standard.result.v)
inburn = 1000

ensemble= FlowSampling(make_Path("ensemble_without_flow",path), post, use_mala=mala, n_samp=n_samp,Knots=k, walker=walker,
                                identystart=false, dims=dims, marginal=false,
                                flow = build_flow(iid, [InvMulAdd, RQSplineCouplingModule(size(iid,1), K = k)]),
                                tuner=BAT.TransformedMCMCNoOpTuning(),burnin=inburn,pretrafo=pretrafo)

ensemble= FlowSampling(make_Path("train_new_flow",path), post, use_mala=mala, n_samp=n_samp,Knots=k, walker=walker,
                                identystart=false, dims=dims, marginal=false,
                                flow = build_flow(iid, [InvMulAdd, RQSplineCouplingModule(size(iid,1), K = k)]),
                                tuner=BAT.MCMCFlowTuning(),burnin=inburn,pretrafo=pretrafo)




