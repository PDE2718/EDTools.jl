ordered_pair(i::T, j::T) where {T} = (i ≤ j) ? (i, j) : (j, i)
pbcid(x::T, L::T) where {T<:Integer} = mod1(x, L)
pbcid(x::NTuple{N,T}, L::NTuple{N,T}) where {N,T<:Integer} = CartesianIndex(mod1.(Tuple(x), L))

@inline function pbcshift(p, s, L)
    return mod1.(Tuple(p) .+ Tuple(s), L) |> CartesianIndex
end