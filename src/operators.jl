abstract type Qop end
mutable struct OpTerm{N} <: Qop
    ladders::NTuple{N,Ladder}
    α::UniformScaling # scaling系数
    ϕ::Union{Nothing,FBbasis} # 基
    table::Union{Nothing,AbstractVector} # 表形式
    matrix::Union{Nothing,AbstractMatrix} # 矩阵形式
end
function OpTerm(ladders... ;α=I, ϕ=nothing)
    return OpTerm((ladders...,),α,ϕ,nothing,nothing)
end

function table!(op::OpTerm)
    @assert ~isnothing(op.ϕ) "The basis is not assigned!"
    if ~isnothing(op.table) return nothing end
    ϕ = op.ϕ
    ε::Int = ϕ.ε
    s::Int = 0
    i::Int = 0
    vt = falses(ϕ.n)
    op.table = Tuple{Int,Int,Int}[]
    for j ∈ eachindex(ϕ.kets)
        s = apply_Ladders!(vt, ϕ.kets[j], op.ladders, ε)
        if s ≠ 0
            i = get(ϕ.b, vt, 0)
            if i ≠ 0
                push!(op.table, (i, j, s))
            end
        end
    end
    return nothing
end

function matrix!(op::OpTerm)
    if ~isnothing(op.matrix) return nothing end
    table!(op)
    N = op.ϕ.N
    op.matrix = sparse(
        [x[1] for x ∈ op.table], [x[2] for x ∈ op.table],
        [x[3] for x ∈ op.table], N,N)
    # if S.M |> isdiag
    #     S.M = S.M |> diag |> Vector |> Diagonal
    # end
    return nothing
end

import Base.Matrix
import LinearAlgebra.Matrix
function Matrix(op::OpTerm)
    matrix!(op)
    return op.α * op.matrix
end

import Base.*
function *(op::OpTerm, ψ::AbstractVector)
    matrix!(op)
    return op.α * (op.matrix * ψ)
end
function *(op1::OpTerm, op2::OpTerm)
    matrix!(op1)
    matrix!(op2)
    return (op1.α * op2.α) * (op1.matrix * op2.matrix)
end
function *(S::OpTerm, A::Union{Number, UniformScaling, AbstractMatrix})
    matrix!(S)
    return (S.α * S.matrix) * A
end
function *(A::Union{Number,UniformScaling,AbstractMatrix}, S::OpTerm)
    matrix!(S)
    return A * (S.α * S.matrix)
end

import Base.adjoint
import LinearAlgebra.adjoint
function adjoint(s::OpTerm)
    s_conj = OpTerm(
        conj.(reverse(s.ladders)),
        conj(s.α),
        s.ϕ,
        nothing,nothing
    )
    return s_conj
end

import Base.+
import Base.-
function +(S1::OpTerm, S2::OpTerm)
    matrix!(S1)
    matrix!(S2)
    return S1.α * S1.matrix + S2.α * S2.matrix
end
function +(S::OpTerm, A::Union{Number,UniformScaling,AbstractMatrix})
    matrix!(S)
    return S.α * S.matrix + A
end
function +(A::Union{Number,UniformScaling,AbstractMatrix}, S::OpTerm)
    matrix!(S)
    return A + S.α * S.matrix
end
function -(S1::OpTerm, S2::OpTerm)
    matrix!(S1)
    matrix!(S2)
    return S1.α * S1.matrix - S2.α * S2.matrix
end
function -(S::OpTerm, A::Union{Number,UniformScaling,AbstractMatrix})
    matrix!(S)
    return S.α * S.matrix - A
end
function -(A::Union{Number,UniformScaling,AbstractMatrix}, S::OpTerm)
    matrix!(S)
    return A - S.α * S.matrix
end

function wf_from_ket(ϕ::FBbasis, k::BitVector)::Vector{ComplexF64}
    @assert length(k) == ϕ.n
    q = ϕ.b[k]
    ψ = zeros(ComplexF64, ϕ.N)
    ψ[q] = 1
    return ψ
end

wf_from_ket(ϕ::FBbasis, k::Vector{Bool}) = wf_from_ket(ϕ::FBbasis, BitVector(k))
wf_from_ket(ϕ::FBbasis, s::String) = wf_from_ket(ϕ, bvector(s))

## [TEST]
# ϕ00 = FBbasis(4, 2, :fermion, true)
# ϕ00.kets

# ϕ00 = FBbasis(4, 0, :fermion, false)
# ϕ00.kets

# n2 = LadderSequence([ddLadder(2), uuLadder(3)]; α = 1im)
# n2 |> printLadderSequence
# n2_dagger = conj(n2)

# to_matrix(ϕ00, n2) |> display
# to_matrix(ϕ00, n2_dagger) |> display

# wf_from_ket(ϕ00, "1010") == wf_from_ket(ϕ00, Bool[1,0,1,0])