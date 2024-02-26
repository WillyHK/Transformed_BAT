function normalize(data::Matrix)
    mean_value = mean(data)
    std_value = std(data)
    normalized_data = (data .- mean_value) ./ std_value
    return normalized_data
end

function train(samples::Matrix, target_logpdf;  batches=1, epochs=1, opti=Adam(1f-3), K = 4, shuffle=false, loss=AdaptiveFlows.negll_flow,
            flow = build_flow((normalize(samples)*(maximum(abs.(samp)/5))), [InvMulAdd, RQSplineCouplingModule(size(samples,1), K = K)]))
    #target_logpdf = x -> logpdf
    return AdaptiveFlows.optimize_flow(samples,flow, opti,  
                loss=loss,
                logpdf = (target_logpdf,AdaptiveFlows.std_normal_logpdf),
                nbatches=batches,nepochs=epochs, shuffle_samples=shuffle)
end


#####################################################################
# Helpfull functions for plotting and stuff like that
#####################################################################
function make_Path(testname::String,dir::String="/ceph/groups/e4/users/wweber/private/Master/Plots")
    path = "$dir/$testname"
    while (isdir(path))
        println("WARNUNG: testname existiert bereits, mache Trick.")
        path = "$path/Trick"
    end
    mkpath(path)
    return path
end


function flat2batsamples(smpls_flat)
    n = length(smpls_flat[:,1])
    smpls = [smpls_flat[i,1:end] for i in 1:n]
    weights = ones(length(smpls))
    logvals = zeros(length(smpls))
    return BAT.DensitySampleVector(smpls, logvals, weight = weights)
end


# using Images
function image2samples(path::String; target_samples=5*10^5)::Matrix{Float64}
    im = load(path);

    im = Gray.(im)
    position_mask = Float64.(im) .< 1

    lines = findall(position_mask)
    line_ids = getfield.(lines, :I)

    raw_samples = [0 1; -1 0] * hcat(collect.(line_ids)...) 
    raw_samples = hcat(fill(raw_samples, 10)...) 

    #raw_samples += 2 .* randn(2, size(raw_samples, 2)) #einfach künstlicher Noise?

    samp = flatview(raw_samples)[1:end,rand(1:length(line_ids), target_samples)]#gpu(raw_samples);

    return normalize(samp)
end


function create_Animation(flow, samples, loss, path, epoch, ani_spline, lr, meta::Plots.Plot; vali=[0], animation_ft, makeSlide=false)
    l = round(vali[end],digits=4);
    x = 1:length(loss);
    learn=plot(x, loss, label="Loss", xlabel="Epoch", ylabel="Loss", title="Loss, End=$l", left_margin = 9Plots.mm, bottom_margin=7Plots.mm);
    x = 1:length(vali);
    plot!(x, vali, label="Validation Loss");
    #ylims!(1.55,1.85);
    savefig("$path/Loss_vali/$(nummer(epoch)).pdf");

    f=plot_flow(flow,samples);
    savefig("$path/ftruth/$(nummer(epoch)).pdf");
    #frame(animation_ft, f);

    plot_spline(flow,samples);
    s=title!("$epoch epochs lr= $lr");
    savefig("$path/spline/$(nummer(epoch)).pdf");

    a = plot(f,meta,layout=(1,2),size=(1200,450),margins=9Plots.mm)
    b = plot(learn,s,layout=(1,2),size=(1200,450),margins=9Plots.mm)
    p = plot(a,b,layout=(2,1),size=(1200,900),margins=9Plots.mm)
    frame(ani_spline, p);

    if (makeSlide)
        savefig(meta,"$path/meta.pdf")
        make_slide("$path/ftruth/$(nummer(epoch)).pdf","$path/meta.pdf",title="Epoch = $epoch, Loss = $l ")
    end
    closeall();
    return nothing
end


function plot_loss_alldimension(path,loss)
    l = round.(loss[end],digits=4);
    p= plot(xlabel="Epoch", ylabel="Loss", title="Loss, End=$(join(string.(l), ", ")) ", left_margin = 9Plots.mm, bottom_margin=7Plots.mm);
    if l isa Float64
        title!("Loss, End=$l")
    end
    for i in 1:size(loss[1],1)
        plot!(1:length(loss), [x[i] for x in loss], label="NN $i")
    end
    plot!(legend=:topright)
    savefig("$path/Loss_vali.pdf");

    return p
end

function plot_metadaten(path, samplesize,minibatches, epochs, batchsize, lr, K, lrf)
    x=plot(size=(800, 600), legend=false, ticks=false, border=false, axis=false);
    annotate!(0.5,1-0.1,"Pfad: $(path[1:49])");
    annotate!(0.5,1-0.2,"$(path[50:end])");
    annotate!(0.5,1-0.3,"Samplesize: $samplesize");
    annotate!(0.5,1-0.4,"Knots: $K");
    annotate!(0.5,1-0.5,"Sample_batchsize: $batchsize, Train_batchsize: $(Int(batchsize/minibatches))");
    annotate!(0.5,1-0.6,"Epochs: $epochs");
    annotate!(0.5,1-0.7,"Start LR: $lr, LR-Factor: $lrf");
    savefig("$path/metadaten.pdf");
    return x
end

function plot_flow(flow,samples; dimension = 1, legend=false)
    p=plot(flat2batsamples((flow(samples)[dimension:dimension,1:end])'))
    x_values = Vector(range(-5.5, stop=5.5, length=1000))
    f(x) = densityof(Normal(0,1.0),x)
    y_values = f.(x_values)
    plot!(x_values, y_values, density=true, linewidth=2.5, label ="N(0,1)", color="black")
    if legend 
        plot!(legend =:topright)
    end
    ylims!(0,0.5)
    return p  
end

function plot_flow_alldimension(path, flow,samples,epoch)
    for i in 1:size(samples,1)
        plot_flow(flow,samples, dimension = i)
        savefig("$path/dim_$i/$(nummer(epoch)).pdf");
    end
    return true
end

function plot_spline(flow, samples; sigma=false) # doesnt make sense for more than one dimension, because then isnt static anymore
    #p=plot(flat2batsamples(samp'),alpha=0.3)

    w,h,d = MonotonicSplines.get_params(flow.flow.fs[2].flow.fs[1].nn(flow.flow.fs[1](samples)[[false],:], 
                                        flow.flow.fs[2].flow.fs[1].nn_parameters, 
                                        flow.flow.fs[2].flow.fs[1].nn_state)[1], 1)
    x = inverse(flow.flow.fs[1])(reshape(w[:,1,1],1,length(w[:,1,1])))'
    #println(reshape(w[:,1,1],1,length(w[:,1,1])))
    y = h[:,1,1]                        
    p=plot(x,y, seriestype = :scatter, label="Knots", legend =true,xlabel="Input value", ylabel="Output value")
    x=range(minimum(x)-0.5,stop=maximum(x)+0.5,length=10^4)
    y = reshape(flow(Matrix(reshape(x,1,10^4))),10^4,1)
    plot!(x,y, linewidth = 2.5, label="Spline function")
    if (sigma)
        plot!(x, ones(length(x))*-2, fillrange=ones(length(x))*2, fillalpha=0.25,c=:yellow ,label="2 sigma region of probability")
        x1,x2 = quantile(samples[1,:],0.96), quantile(samples[1,:],0.04)
        plot!(Shape([x1, x2, x2, x1], [-5.5,-5.5,5.5,5.5]), fillalpha=0.25, c=:yellow, linewidth=0.0, label=false, line=false)
    end
    ylims!(-5.5,5.5)
    #savefig("$path/spline.pdf")
    return p
end

function make_slide(pdfpath1; slidepath = "/ceph/groups/e4/users/wweber/private/Master/Slides", title="Title")
         
    title = replace(title, "_" => "\\_")
    file = open("$slidepath/plots.tex", "a")
    write(file, "\\begin{frame}{$title}\n")
    write(file, "   \\begin{figure}\n")
    write(file, "       \\centering\n")
    write(file, "       \\includegraphics[width=0.7\\textwidth]{$pdfpath1}\n")
    write(file, "   \\end{figure}\n")
    write(file, "\\end{frame}\n\n")
    close(file)
end

function make_slide(pdfpath1,pdfpath2; slidepath = "/ceph/groups/e4/users/wweber/private/Master/Slides", title="Title")
         
    title = replace(title, "_" => "\\_")
    file = open("$slidepath/plots.tex", "a")
    write(file, "\\begin{frame}{$title}\n")
    write(file, "   \\begin{figure}\n")
    write(file, "       \\centering\n")
    write(file, "       \\begin{subfigure}{0.49\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath1}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "       \\begin{subfigure}{0.49\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath2}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "   \\end{figure}\n")
    write(file, "\\end{frame}\n\n")
    close(file)
end

function nummer(x; digits=5)
    return string(x, pad=digits, base=10)
end

function make_slide(pdfpath1,pdfpath2,pdfpath3; slidepath = "/ceph/groups/e4/users/wweber/private/Master/Slides", title="Title")
         
    title = replace(title, "_" => "\\_")
    file = open("$slidepath/plots.tex", "a")
    write(file, "\\begin{frame}{$title}\n")
    write(file, "   \\begin{figure}\n")
    write(file, "       \\centering\n")
    write(file, "       \\begin{subfigure}{0.44\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath1}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "       \\begin{subfigure}{0.44\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath2}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "       \\begin{subfigure}{0.35\\textwidth}\n")
    write(file, "           \\includegraphics[width=\\textwidth]{$pdfpath3}\n")
    write(file, "       \\end{subfigure}\n")
    write(file, "   \\end{figure}\n")
    write(file, "\\end{frame}\n\n")
    close(file)
end

#####################################################################################################
# Create IID samples  (fomal modell = model = MixtureModel([MvNormal([0],I(1)),MvNormal([3],I(1))]))
#####################################################################################################
function get_iid(peakpositions,dim,n)
    peakss::Vector{MvNormal} = []
    for peak in collect(Iterators.product(fill(peakpositions, dim)...))
        push!(peakss, MvNormal(collect(peak),I(dim)))
    end
    model = MixtureModel(peakss)
    return Matrix(rand(model,n))
end
