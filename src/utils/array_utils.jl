# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_iscontiguous(A::Array) = true
_iscontiguous(A::AbstractArray) = Base.iscontiguous(A)


_car_cdr_impl() = ()
_car_cdr_impl(x, y...) = (x, (y...,))
_car_cdr(tp::Tuple) = _car_cdr_impl(tp...)

function _all_lteq(A::AbstractArray, B::AbstractArray, C::AbstractArray)
    axes(A) == axes(B) == axes(C) || throw(DimensionMismatch("A, B and C must have the same indices"))
    all(x[1] <= x[2] <= x[3] for x in zip(A, B, C))
end


@inline function _all_lteq_impl(a::Real, B::AbstractArray, c::Real)
    result = 0
    @inbounds @simd for b in B
        result += ifelse(a <= b <= c, 1, 0)
    end
    result == length(eachindex(B))
end

_all_lteq(a::Real, B::AbstractArray, c::Real) = _all_lteq_impl(a, B, c)




"""
    @propagate_inbounds sum_first_dim(A::AbstractArray, j::Integer, ks::Integer...)

*BAT-internal, not part of stable public API.*

Calculate the equivalent of `sum(A[:, j, ks...])`.
"""
Base.@propagate_inbounds function sum_first_dim(A::AbstractArray, j::Integer, ks::Integer...)
    s = zero(eltype(A))
    @boundscheck if !Base.checkbounds_indices(Bool, Base.tail(axes(A)), (j, ks...))
        throw(BoundsError(A, (:, j)))
    end
    @inbounds for i in axes(A, 1)
        s += A[i, j, ks...]
    end
    s
end


"""
    @propagate_inbounds sum_first_dim(A::AbstractArray)

*BAT-internal, not part of stable public API.*

If `A` is a vector, return `sum(A)`, else `sum(A, 1)[:]`.
"""
sum_first_dim(A::AbstractArray) = sum(A, 1)[:]
sum_first_dim(A::AbstractVector) = sum(A)


const SingleArrayIndex = Union{Integer, CartesianIndex}


convert_numtype(::Type{T}, x::T) where {T<:Real} = x
convert_numtype(::Type{T}, x::Real) where {T<:Real} = convert(T, x)

convert_numtype(::Type{T}, x::AbstractArray{T}) where {T<:Real} = x
convert_numtype(::Type{T}, x::AbstractArray{<:Real}) where {T<:Real} = convert.(T, x)

convert_numtype(::Type{T}, x::ShapedAsNT{<:Any,<:AbstractArray{T}}) where {T<:Real} = x
convert_numtype(::Type{T}, x::ShapedAsNT{<:Any,<:AbstractArray{<:Real}}) where {T<:Real} =
    varshape(x)(convert_numtype(T, unshaped(x)))

convert_numtype(::Type{T}, x::ShapedAsNTArray{<:Any,N,<:AbstractArray{<:AbstractArray{T}}}) where {T<:Real,N} = x
convert_numtype(::Type{T}, x::ShapedAsNTArray{<:Any,N,<:AbstractArray{<:AbstractArray{<:Real}}}) where {T<:Real,N} =
    varshape(x).(convert_numtype(T, unshaped.(x)))

convert_numtype(::Type{T}, x::ArrayOfSimilarArrays{T,M,N}) where {T<:Real,M,N} = x
convert_numtype(::Type{T}, x::VectorOfSimilarArrays{<:Real,M,N}) where {T<:Real,M,N} =
    ArrayOfSimilarArrays{T,M,N}(convert_numtype(T, flatview(x)))

any_isinf(trg_v::Real) = isinf(trg_v)
any_isinf(trg_v::AbstractVector{<:Real}) = any(isinf, trg_v)


# Similar to ForwardDiffPullbacks._fieldvals:

object_contents(x::Tuple) = x
object_contents(x::AbstractArray) = x
object_contents(x::NamedTuple) = values(x)

@generated function object_contents(x)
    accessors = [:(getfield(x, $i)) for i in 1:fieldcount(x)]
    :(($(accessors...),))
end
