# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::typeof(bat_integrate), ::Val{:algorithm}, ::AnySampleable) = AHMIntegration()
bat_default(::typeof(bat_integrate), ::Val{:algorithm}, ::SampledDensity) = BridgeSampling()
