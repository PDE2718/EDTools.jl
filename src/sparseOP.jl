using LinearAlgebra
using SparseArrays
using Base
import Base:*,+,-
using Base.Threads

mutable struct DictLinMap{Tv, Ti, M, N} <: AbstractSparseMatrix{Tv,Ti}
    dict::Dict{Tuple{Ti, Ti}, Tv}
    # table::Vector{Pair{Tuple{Ti, Ti}, Tv}}
    table::Vector{Vector{Pair{Ti, Tv}}}
    # spfmt::Union{SparseMatrixCSC, Nothing}
    uptodate::Bool
end
function DictLinMap(Tv::Type, M::Ti, N::Ti) where Ti <: Integer
    d = Dict{Tuple{Ti, Ti}, Tv}()
    v = [Vector{Pair{Ti,Tv}}[] for i ∈ 1:M]
    H = DictLinMap{Tv,Ti,M,N}(d, v, false)
    return H
end

Base.size(H::DictLinMap{Tv, Ti, M, N}) where {Tv, Ti, M, N} = (M, N)
function Base.size(H::DictLinMap{Tv, Ti, M, N}, dim::Int) where {Tv, Ti, M, N}
    if dim == 1 return M
    elseif dim == 2 return N
    else error("dim error")
    end
end
Base.eltype(H::DictLinMap{Tv,Ti,M,N}) where {Tv,Ti,M,N} = Tv
function Base.getindex(H::DictLinMap{Tv, Ti, M, N}, i::Ti, j::Ti) where {Tv, Ti, M, N}
    return get(H.dict, (i,j), zero(Tv))
    # return get(H.dict, (i,j), H.default_val)
end
function Base.setindex!(H::DictLinMap{Tv, Ti, M, N}, value, i::Ti, j::Ti) where {Tv, Ti, M, N}
    # set!(H.dict,(i,j),value)
    H.dict[(i,j)] = value
    H.uptodate = false
    return nothing
end
function Base.zero(H::DictLinMap{Tv,Ti,M,N}) where {Tv,Ti,M,N}
    return DictLinMap(Tv, Ti(M), Ti(N))
end
function update!(H::DictLinMap{Tv, Ti, M, N}) where {Tv, Ti, M, N}
    H.uptodate = true
    @threads for row ∈ H.table
        rsize = length(row)
        sizehint!(row,2rsize)
        empty!(row)
    end
    for (k,v) ∈ H.dict
        push!(H.table[k[1]],k[2]=>v)
    end
    @threads for row in H.table
        sort!(row,by=x->x.first)
    end
end
function muladd!(ϕ::Vector{T}, H::DictLinMap{Tv, Ti, M, N}, ψ::Vector{T}) where {Tv, Ti, M, N, T}
    if ~H.uptodate update!(H) end
    @threads for r ∈ eachindex(H.table)
        for (c, v) ∈ H.table[r]
            ϕ[r] += ψ[c]*v
        end
        # ϕ[k[1]] += ψ[k[2]] * v
    end
end
function *(H::DictLinMap{Tv,Ti,M,N}, ψ::Vector{T}) where {Tv,Ti,M,N,T}
    ϕ = zero(ψ)
    muladd!(ϕ, H, ψ)
    return ϕ
end
function *(α::Number, H::DictLinMap{Tv,Ti,M,N}) where {Tv,Ti,M,N,T}
    Hp = deepcopy(H)
    Hp.uptodate = false
    Hp.dict.vals .*= α
    return Hp
end
*(H::DictLinMap{Tv, Ti, M, N}, α::Number) where {Tv,Ti,M,N,T} = α * H
function +(H::DictLinMap{Tv,Ti,M,N}, G::DictLinMap{Tv,Ti,M,N}) where {Tv,Ti,M,N}
    V = deepcopy(H)
    V.uptodate = false
    for (k,v) ∈ G
        V[k...] += v
    end
    return V
end

# # [test]
NN = 2^16
AA = DictLinMap(ComplexF64, NN, NN)
AA_sp = spzeros(ComplexF64,NN,NN)
rand_data = [(rand(1:NN),rand(1:NN))=>rand(ComplexF64) for i ∈ 1:100NN]
Iv = [t.first[1] for t ∈ rand_data]
Jv = [t.first[2] for t ∈ rand_data]
Vv = [t.second for t ∈ rand_data]
@elapsed for q ∈ rand_data
    AA[q.first...] = q.second
end
@elapsed for q ∈ rand_data
    AA_sp[q.first...] = q.second
end
@elapsed AA_sp = sparse(Iv,Jv,Vv,NN,NN)

AA.table
# AA_dense = AA[:,:]
x = randn(ComplexF64,NN)
using BenchmarkTools
@btime norm(AA*x)

# (AA * x - AA_sp * x) |> norm

# using BenchmarkTools

# @btime $(AA) * $(x)
# @btime $(AA_dense) * $(x)
# @btime $(AA_sp) * $(x)




