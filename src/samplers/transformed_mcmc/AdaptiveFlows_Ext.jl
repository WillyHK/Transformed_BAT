# This file is a part of BAT.jl, licensed under the MIT License (MIT).

(f::AdaptiveFlows.AbstractFlow)(x::ShapedAsNTArray, vs::AbstractValueShape) = vs.(nestedview(f(Matrix(flatview(unshaped.(x))))))
(f::AdaptiveFlows.AbstractFlow)(x::SubArray) = f(Vector(x))
(f::AdaptiveFlows.AbstractFlow)(x::ArrayOfSimilarArrays) = nestedview(f(flatview(x)))
(f::AdaptiveFlows.AbstractFlow)(x::DensitySampleVector) = apply_flow_to_density_samples(f, x)
(f::AdaptiveFlows.AbstractFlow)(x::ElasticArrays.ElasticMatrix) = f(Matrix(reshape(x[1], :, 1)))

function ChangesOfVariables.with_logabsdet_jacobian(f::AdaptiveFlows.AbstractFlow, x::ArrayOfSimilarArrays)
    y, ladj = with_logabsdet_jacobian(f, Matrix(flatview(x)))
    return nestedview(y), ladj
end    

function ChangesOfVariables.with_logabsdet_jacobian(f::AdaptiveFlows.AbstractFlow, x::SubArray)
    return with_logabsdet_jacobian(f, Vector(x))
end    

function apply_flow_to_density_samples(f::AdaptiveFlows.AbstractFlow, x::DensitySampleVector)
    v_flat = flatview(x.v)
    y, ladj = with_logabsdet_jacobian(f, Matrix(v_flat))
    return DensitySampleVector((nestedview(y), x.logd - vec(ladj), x.weight,  x.aux, x.info))
end
