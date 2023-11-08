# 阻塞hardcore bosons气体的动力学

using Revise
using EDTools
using LinearAlgebra, SparseArrays
using Statistics
using Base.Threads
BLAS.set_num_threads(16)
# using DifferentialEquations

L::Int = 24
N::Int = L-4
λ = 0.2

ordered_pair(i::T, j::T) where {T} = (i ≤ j) ? (i, j) : (j, i)
# pbcid(x::T, L::T) where {T<:Integer} = mod1(x, L)
# pbcid(x::NTuple{N,T}, L::NTuple{N,T}) where {N,T<:Integer} = CartesianIndex(mod1.(Tuple(x), L))

begin
    adj_pairs = NTuple{2,Int}[]
    # use PBC
    for i ∈ 1:L
        push!(adj_pairs,ordered_pair(i,pbcid(i+1,L)))
        push!(adj_pairs,ordered_pair(i,pbcid(i+2,L)))
    end
    # # OBC case !
    # for i ∈ 1:L-2
    #     push!(adj_pairs, (i, i+1))
    #     push!(adj_pairs, (i, i+2))
    # end
    # push!(adj_pairs, (L-1,L))

    nblist = [Int[] for i ∈ 1:L]
    for (i,j) ∈ adj_pairs
        push!(nblist[i],j)
        push!(nblist[j],i)
    end
    sort!.(nblist)
    com_nbs = Vector{Int}[]
    for (i,j) ∈ adj_pairs
        push!(com_nbs, nblist[i] ∩ nblist[j])
    end
    sort!.(com_nbs)
end

nblist
adj_pairs
com_nbs

ϕ = FBbasis(L, N, :hcboson, true)
ni = [densities(i;ϕ=ϕ) for i ∈ 1:L]
tij = [hopping(i,j;ϕ=ϕ) for (i,j) ∈ adj_pairs]
tji = [hopping(j,i;ϕ=ϕ) for (i,j) ∈ adj_pairs]
all_ops = vcat(ni,tij,tji)

using Base.Threads
@threads for op ∈ all_ops
    table!(op)
    matrix!(op)
end

Cij = [I - reduce(*,ni[k] for k ∈ com_nb) for com_nb ∈ com_nbs]

Top = let T = spzeros(ComplexF64, ϕ.N, ϕ.N)
    for k ∈ eachindex(tij, tji, Cij)
        # i, j = adj_pairs[k]
        T -= Cij[k] * (tij[k] + tji[k])
    end
    T
end
Vop = let V = spzeros(ComplexF64, ϕ.N, ϕ.N)
    for k ∈ eachindex(tij, tji, Cij)
        i, j = adj_pairs[k]
        V += Cij[k] * (ni[i] + ni[j] - 2 * ni[i] * ni[j])
    end
    V
end
H = λ * Top + (1-λ) * Vop
H_dense = let Hp = deepcopy(H)
    Hp = all(isreal,Hp) ? real.(Hp) : Hp
    Hp = Matrix(Hp)
    Hp = ishermitian(Hp) ? Hermitian(Hp) : Hp
    Hp
end
# using AppleAccelerate
# BLAS.lbt_get_config()
@time ω,F = eigen(H_dense)

unoccpupied_ini = (9, 11, 17, 22)
ψ0_bv = BitVector(i ∉ unoccpupied_ini for i ∈ 1:L)
ψ0 = wf_from_ket(ϕ, ψ0_bv)
ψ0_id = ϕ.b[ψ0_bv]

function autocorr_i(ψ,i)
    nn = ni[i].matrix
    cc = dot(ψ, nn, ψ) * nn[ψ0_id, ψ0_id]
    return real(cc)
end
function autocorr(ψ)
    η = N / L
    c = sum(i -> autocorr_i(ψ, i), 1:L)/(L*η*(1-η)) - η/(1-η)
    return c
end

# randψ = randn(ComplexF64, size(ψ0)) |> normalize
# autocorr_i(randψ, 12)
# autocorr(ψ0)
# autocorr(randψ)

function ψt(t)
    return F * (Diagonal(cis.(-ω.*t)) * (F' * ψ0))
end

tgrid = vcat(exp10.(-2:0.02:10))
ψs = [ψt(t) for t ∈ tgrid]
autocorr_vals = [autocorr(ψ) for ψ ∈ ψs]
using Plots, LaTeXStrings, Measures

plt1 = plot(tgrid[2:end],autocorr_vals[2:end],xscale = :log10,ylims=(0.,1.2), xminorgrid = true, xlims = (1e-1,1e10), size = [600,300], xlabel=L"t", ylabel=L"c(t)", labels = latexstring("\\lambda=$(λ)"),yminorgrid = true, framestyle = :box )

savefig(plt1,"QLG_lambda02.pdf")

ρA = rdm_generator(ϕ, collect(1:10))

S_von_A = [ρA(ψ) |> vonNeumann_entropy for ψ ∈ ψs]

plt2 = plot(tgrid[2:end], S_von_A[2:end], xscale=:log10, xminorgrid=true, xlims=(1e-1, 1e10), size=[600, 300], xlabel=L"t", ylabel=L"S_A", labels=latexstring("\\lambda=$(λ)"), yminorgrid=true, framestyle=:box)

savefig(plt2, "SA.pdf")
# eigen
# miH = -1im * H
# miHt = DiffEqArrayOperator(miH)
# prob = ODEProblem(miHt, ψ0, (0., 1000.))
# tgrid = vcat([0.0],exp10.(0:0.1:3))
# sol = solve(prob, LinearExponential(krylov=:adaptive);
#     saveat = tgrid,
# )



# H_fermion = H |> deepcopy
# H_hcboson = H |> deepcopy

# H_fermion .- H_hcboson

# using JLD2
# F = eigen(H_dense)

# F.values
# using Plots
# jldsave("lambda02.jld2"; H=H, F=F)

# ψ0_bv = fill(true,L)
# ψ0_bv[[9,11,17,22]].=false
# wf_from_ket(ϕ,ψ0_bv)


# jldopen("lambda02.jld2")["F"]

# ϕ = FBbasis(L, N, :hcboson, true)

# Kij = hopping_ij()