##################################################################################
# Nehme Samples die als dxn-Matrix vorliegen und bringe sie ins Bat-Plot-Format
##################################################################################
function flat2batsamples(smpls_flat)
    n = length(smpls_flat[:,1])
    smpls = [smpls_flat[i,1:end] for i in 1:n]
    weights = ones(length(smpls))
    logvals = zeros(length(smpls))
    return BAT.DensitySampleVector(smpls, logvals, weight = weights)
end

##################################################################################
# Batsamples to Matrix
##################################################################################
function BAT2Matrix(x::Vector)::Matrix{Float64}
    zeilen = length(x[1])
    spalten = length(x)
    matrix::Matrix{Float64} = zeros(zeilen,spalten)
    for i in 1:zeilen
        for j in 1:spalten
            matrix[i,j] = x[j][i]
        end
    end
    return matrix
end


##################################################################################
# Collection of interesting BAT-Posteriors
##################################################################################
function get_exp(dim = 3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-10,10) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(Exponential(1.0), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end

function get_normal(dim = 3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-10,10) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(Normal(0,1.0), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end


function get_testcase(dim = 3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Normal(0,1.0) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(Uniform(-5,5), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end

function get_modemode(dim = 3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-10,10) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(MixtureModel(Normal, [(0,0.5),(0,3.0)],[0.5,0.5]), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end

function get_dualmode(dim = 3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-10,10) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(MixtureModel(Normal, [(-2,1.0),(2,1.0)],[0.5,0.5]), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end

function get_triplemode(dim = 3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-10,10) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(MixtureModel(Normal, [(-3,1.0),(0,1.0),(3,1.0)],[1/3,1/3,1/3]), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end


function get_funnel(dim = 3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-20,20) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin
        return LogDVal(logpdf(BAT.FunnelDistribution(n=dim),collect(params))[1])#Base.values(params))))
    end

    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end


function get_mix(dim = 3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-20,20) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(MixtureModel(Normal, [(-5,1.0),(5,1.0),(0,4.)],[0.4,0.4,0.2]), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end


function get_rings2d(radius = [0.5,1.0,1.5], variances = [0.05, 0.05, 0.05])
    prior = BAT.NamedTupleDist(a = Uniform(-3,3), b = Uniform(-3,3)) # Modell

    likelihood = params -> begin # Wahrscheinlichkeit  # Sind das die wahren Werte/Messwerte?
        num_rings = length(radius)
        means = []
        [push!(means,[[radius[j]*cos(i), radius[j]*sin(i)] for i in range(0,2pi,100)]) for j in 1:num_rings]
        
        ll = 0
        for j in 1:num_rings
            for i in 1:length(means[j])-1
                ring_dist = MvNormal(means[j][i], variances[j])
                ll += pdf(ring_dist, [params[1],params[2]])
            end
        end
        return LogDVal(log(ll))
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end


function get_myPosterior()
    prior = BAT.NamedTupleDist(a = Uniform(-25,25), b = Uniform(-25,25), c=Uniform(-25,25)) # Modell

    likelihood = params -> begin # Wahrscheinlichkeit  # Sind das die wahren Werte/Messwerte?
        r1 = logpdf(Normal(0,1), params[1])
        r2 = logpdf(Normal(0,pdf(Normal(0,1),params[1])^2), params[2])
        r3 = logpdf(Normal(0,pdf(Normal(0,1),params[1])^4), params[3]) # logpdf: log(probability density function)
        return LogDVal(r1+r2+r3)
    end

    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end


function get_kamm(dim=3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-15,15) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(MixtureModel(Normal, [(-7,1.0),(-3.5,1.0),(0,1.0),(3.5,1.0),(7,1.0)],[0.2,0.2,0.2,0.2,0.2]), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end


function get_multimodal(dim=3)
    label = [Symbol(string(Char(i))) for i in 97:97+dim-1]
    value = [ Uniform(-25,25) for i in 1:dim]
    prior = BAT.NamedTupleDist((; zip(label,value)...)) # Modell

    likelihood = params -> begin      
        r=0
        for i in 1:dim
            r+= logpdf(MixtureModel(Normal, [(-16,1.0),(-8,1.0),(0,1.0),(8,1.0),(16,1.0)],[0.35,0.3,0.2,0.1,0.05]), params[i])
        end
        return LogDVal(r)
    end
    posterior = PosteriorDensity(likelihood, prior);
    return posterior
end