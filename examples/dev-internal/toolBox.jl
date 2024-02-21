
using Images
function image2samples(path::String; target_samples=5*10^5)::Matrix{Float64}
    im = load(path);

    im = Gray.(im)
    position_mask = Float64.(im) .< 1

    lines = findall(position_mask)
    line_ids = getfield.(lines, :I)

    desired_samples = min(target_samples, length(line_ids))
    selected_indices = rand(1:length(line_ids), desired_samples)

    raw_samples = [0 1; -1 0] * hcat(collect.(line_ids[selected_indices])...) 
    raw_samples = hcat(fill(raw_samples, 10)...) 

    #raw_samples += 2 .* randn(2, size(raw_samples, 2)) #einfach künstlicher Noise?

    samp = gpu(raw_samples);

    return samp./(std(samp)/2)
end


function makeBild(bild, path)
    samp = image2samples(bild)
    plot(flat2batsamples(samp'), density=true,right_margin=9Plots.mm)
    savefig("$path/true.pdf")

    target_logpdf = x-> logpdf(MvNormal(zeros(2),I(2)),x)
    #using BenchmarkTools
    flow=build_flow(samp, [InvMulAdd, RQSplineCouplingModule(size(samp,1), K = 40)])
    lr=1f-3
    plot_flow(flow,samp)
    ylims!(-2.5,2.5)
    xlims!(-2.5,2.5)
    savefig("$path/dpg0.pdf")
    for i in 1:9
        @time flow, opt_state, loss_hist = AdaptiveFlows.optimize_flow(samp,flow, Adam(lr),
                               loss=AdaptiveFlows.negll_flow, #loss_history = loss_hist,
                               logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
                               nbatches=5,nepochs=1, shuffle_samples=true)
        plot_flow(flow,samp)
        ylims!(-2.5,2.5)
        xlims!(-2.5,2.5)
        savefig("$path/dpg$i.pdf")
        #lr=lr*0.95
    end
    for i in 10:20
        @time flow, opt_state, loss_hist = AdaptiveFlows.optimize_flow(samp,flow, Adam(lr),
                               loss=AdaptiveFlows.negll_flow, #loss_history = loss_hist,
                               logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
                               nbatches=10,nepochs=5, shuffle_samples=true)
        plot_flow(flow,samp)
        ylims!(-2.5,2.5)
        xlims!(-2.5,2.5)
        savefig("$path/dpg9$i.pdf")
        #lr=lr*0.95
    end
end