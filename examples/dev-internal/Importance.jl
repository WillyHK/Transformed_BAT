#using BenchmarkTools
dims=100
path = make_Path("Importance-$dims")
num_samples =5*10^6
smallpeak = 0.1
model = MixtureModel(Normal, [(-10.0, 1.0),(10.0, 1.0)], [smallpeak, 1-smallpeak])
#smallpeak = 0.01
#model = MixtureModel(Normal, [(-10.0, 0.1),(0.0,0.2),(10.0, 1.0)], [smallpeak, 5*smallpeak, 1-smallpeak*6])
x_range = range(-15, stop = 15, length = 1000)
theorie = pdf(model, x_range)

function log_likelihood(model, x)
    log_likelihood = 0.0
    for xi in x
        log_likelihood += log(pdf(model, xi))
    end
    return log_likelihood
end

# #################################################
# # Ohne Flow
# ################################################
# function importance_sampling(num_samples)
#     samples = []
#     weights = zeros(0)
#     logds = zeros(0)
#     
#     for _ in 1:num_samples
#         # Sample aus der uniformen Prior-Verteilung
#         sample = rand(Uniform(-15, 15), dims)
#         
#         # Berechnen der Gewichte
#         weight = exp(log_likelihood(model,sample))
#         logd = log_likelihood(model,sample)
#         
#         push!(logds,logd)
#         push!(samples, sample)
#         push!(weights, weight)
#     end
#     
#     # Normierung der Gewichte
#     weights /= sum(weights)
# 
#     return (BAT2Matrix(samples),logds, weights)
# end
# 
# s2,l2,w2 = importance_sampling(10)
# s2,l2,w2 = importance_sampling(num_samples)
# 
# plot(flat2batsamples(s2',w=w2),bins=400)
# plot!(x_range, theorie, xlabel="x", ylabel="Dichte", label="Mixture Model", color=:black,linewidth=2.5)
# savefig("$path/uniformsampling.pdf")

##################################################################
# Jetzt mit Flow
##################################################################
flow = nothing
flowpath = "/ceph/groups/e4/users/wweber/private/Master/Flows/Importance"
iid = BAT2Matrix(rand(get_prior(model,dims),5*10^5))
if isfile("$flowpath/flow_$dims.jls")
    flow = loadFlow(flowpath, dims)
else
    train(iid[:,1:3],AdaptiveFlows.std_normal_logpdf)
    flow, opti, loss_hist = train(iid,AdaptiveFlows.std_normal_logpdf, epochs=25, batches=4, 
                                    shuffle=true, opti=Adam(5f-2), K = 20)
    plot_loss_alldimension(path,loss_hist[2][1])
    saveFlow(flowpath,flow,name="flow_$dims.jls")
    saveFlow(path,flow,name="flow_$dims.jls")
end
plot_flow_alldimension(path,flow,iid,0)
invf = inverse(flow)
plot(flat2batsamples(invf(rand(MvNormal(zeros(dims),ones(dims)),10^5))'))
savefig("$path/iflowOnNormal.pdf")

function logpdf_normal(x)
    return log(pdf(MvNormal(zeros(dims),ones(dims)),x))
end

function importance_sampling_flow(num_samples)

    points_n = rand(MvNormal(zeros(dims),ones(dims)),num_samples)
    points, ladj = with_logabsdet_jacobian(invf.flow,points_n)

    points2=nestedview(points)
    points2_n=nestedview(points_n)
    logds = log_likelihood.(model,points2) + vec(ladj) - logpdf_normal.(points2_n)
    weights = exp.(logds)
    samples = points2
    
    # Normierung der Gewichte
    weights /= sum(weights)

    return (BAT2Matrix(samples),logds, weights)
end

samp,l,w = importance_sampling_flow(10)
samp,l,w = importance_sampling_flow(num_samples)

plot(flat2batsamples(samp',w=w),bins=500)
plot!(x_range, theorie, xlabel="x", ylabel="Dichte", label="Mixture Model", color=:black,linewidth=1.2)
savefig("$path/flowsampling.pdf")


# function train_flow(flow)
#     flowpath = "/ceph/groups/e4/users/wweber/private/Master/Flows/Importance"
#     iid = BAT2Matrix(rand(get_prior(model,dims),10^4))
#     flow, opti, loss_hist = train(iid,AdaptiveFlows.std_normal_logpdf, 
#         epochs=20, batches=4, shuffle=true, opti=Adam(5f-4), K = 20,flow=flow);
#     plot(flat2batsamples(flow(iid)'))
#     saveFlow(flowpath,flow,name="flow_$dims.jls")
#     return flow
# end
# 
# for i in 1:10
#     flow = train_flow(flow)
# end