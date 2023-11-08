# 复现大师兄的ED结果

begin
using Revise
using EDTools
using LinearAlgebra, SparseArrays, Statistics, Graphs, KrylovKit
using Base.Threads
end
nthreads()
# using Distributed
# addprocs(128)
begin
function nextPXPconfig!(conf, n, G)
    nbs = neighbors(G,n)
    nbs = nbs[nbs .< n]
    N = length(conf)
    for i ∈ 1:N
        if ~any(conf[i][nbs])
            push!(conf,vcat(conf[i],true))
        end
        push!(conf[i], false)
    end
end
function PXPconfig(G)
    nmax = nv(G)
    conf = [[false].==true,[true].==true]
    for n = 2:nmax
        nextPXPconfig!(conf, n, G)
    end
    return conf
end
end

L = (6,6)
n_site = prod(L)
Vac = zeros(Bool, L)
lattice = CartesianIndices(Vac)
Lids = LinearIndices(Vac)
G = SimpleGraph(length(Vac))
for r ∈ lattice
    rU = pbcshift(r, (0, 1), L)
    rR = pbcshift(r, (1, 0), L)
    add_edge!(G, Lids[r], Lids[rU])
    add_edge!(G, Lids[r], Lids[rR])
end
kets = sort(PXPconfig(G); by = sum)
ϕ = FBbasis(kets, :hcboson)

bi = [annihilation(i;ϕ=ϕ) for i ∈ Lids]
b̂i = [creation(i;ϕ=ϕ) for i ∈ Lids]
ni = [densities(i;ϕ=ϕ) for i ∈ Lids]
all_ops = vcat(bi[:],b̂i[:],ni[:])

@threads for op ∈ all_ops
    table!(op)
end
@threads for nop ∈ ni
    matrix!(nop)
end

begin
H_IJV = NTuple{3,Int}[]
for (b,b̂) ∈ zip(bi,b̂i)
    append!(H_IJV,b.table)
    append!(H_IJV,b̂.table)
end
H_sp = sparse([[x[i] for x ∈ H_IJV] for i ∈ 1:3]...,ϕ.N,ϕ.N)
end

eigH = eigsolve(H_sp, randn(ComplexF64,ϕ.N), 1, :SR; ishermitian=true)
Egs = eigH[1][1]./prod(L)
ψgs = eigH[2][1]
cdw_op = (sum(((-1)^(r[1]+r[2]))*ni[r] for r ∈ lattice))^2 |> dropzeros
N_op = sum(n for n ∈ ni)

function observables_(ψ)
    Etrial = dot(ψ, H_sp, ψ) / n_site |> real
    m = (2 / n_site) * √(dot(ψ, cdw_op, ψ) |> real)
    N = dot(ψ, N_op, ψ) |> real
    Nsq = dot(ψ, N_op^2, ψ) |> real
    n̄ = N / n_site
    κ = L[1] / n_site * (Nsq - N^2)
    return (Egs = Etrial, m = m, n̄ = n̄, κ = κ)
end

trial_A = (x->isodd(x[1] + x[2])).(lattice)[:]
trial_B = (x->iseven(x[1] + x[2])).(lattice)[:]
function trial_sgn(k::BitVector,occ::BitVector)
    if any(k[occ.==false])
        return 0
    else
        return (-1)^(count(k[occ]))
    end
end

ψA = [trial_sgn(k,trial_A) for k ∈ kets] |> normalize
ψB = [trial_sgn(k,trial_B) for k ∈ kets] |> normalize
ψtrial = (ψA + ψB) |> normalize

observables_(ψgs)
observables_(ψtrial)
dot(ψtrial,ψgs) |> abs

# theoretical value for ψ_trial on finite lattice
m_trial_th = let N = prod(L)
    1/2 * √(1+2/N)
end
observables_(ψtrial)[:m]