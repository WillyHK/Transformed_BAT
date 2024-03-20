using EFTfitter
#using BAT            # for sampling
#using IntervalSets   # for specifying the prior
#using Distributions  # for specifying the prior
#using Plots

parameters = BAT.distprod(
    ctW =  -3..3, 
    ctZ =  -3..3,
    ctl1 =  -3..3,
    cte1 = -3..3,
    cQe1 =  -3..3
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
    Meas1 = Measurement(AFB_t_ctW, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
    Meas2 = Measurement(AFB_t_ctZ, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
    Meas3 = Measurement(AFB_t_ctl1, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
    Meas4 = Measurement(AFB_t_cte1, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
    Meas5 = Measurement(AFB_t_cQe1, 0.17978, uncertainties = (stat = 0.0021, ), active = true), 
)

correlations = (
    stat = NoCorrelation(active=true),
)

model = EFTfitterModel(parameters, measurements, correlations)

posterior = PosteriorMeasure(model)

algorithm = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^6, nchains = 4, strict = false)
samples = bat_sample(posterior, algorithm).result;

plot(samples)