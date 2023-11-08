using Revise
using EDTools
using LinearAlgebra, SparseArrays
using Graphs
using DataFrames

L = 9
Lids = 1:L
nbs = [pbcid.((i-1,i+1),L) for i ∈ Lids]

ϕ = FBbasis(L, 0, :hcboson, false)
# ϕ.kets .= ϕ.kets[subspace_perm]
# for i ∈ eachindex(ϕ.kets)
#     ϕ.b[ϕ.kets[i]] = i
# end
ni = [densities(i;ϕ=ϕ) for i ∈ Lids];
bi = [annihilation(i; ϕ=ϕ) for i ∈ Lids];
b̂i = [creation(i; ϕ=ϕ) for i ∈ Lids];
all_ops = vcat(ni,bi,b̂i)
for op ∈ all_ops
    matrix!(op)
end

function ∏k(i::Int)
    Λ = nbs[i]
    Λ = sum(prod((m==l ? ni[m] : (I-ni[m])) for m ∈ Λ) for l ∈ Λ)
end
ϕ
∏i = [∏k(i) for i ∈ Lids]
dropzeros!.(∏i)

H = sum(∏i[i]*(bi[i] + b̂i[i]) for i ∈ Lids) |> dropzeros
H
H_g = SimpleGraph(ϕ.N)
for (x,y,z) ∈ zip(findnz(H)...)
    add_edge!(H_g,x,y)
end
H_g
H_subspace = connected_components(H_g)
# subspace_perm = vcat(H_subspace...)
kets_subs = [ϕ.kets[s] for s ∈ H_subspace]
length.(H_subspace)
H_eigen = [eigen(Matrix(H[s,s])) for s ∈ H_subspace]

subspace_info = DataFrame(
    subspace_size = length.(H_subspace),
    sub_gs_energy = [H_eigen[i].values[1] for i ∈ eachindex(H_subspace)]
)
subspace_info |> display

eigen(H|>Matrix)

kets_subs[4]

gs_sub = 4
Egs = H_eigen[gs_sub].values[1]
ψgs = H_eigen[gs_sub].vectors[:,1]

maximum(H_eigen[gs_sub].vectors[:, 1].^2)
kets_subs[3]

H_eigen[gs_sub].vectors[:, 1]

n_gs = sum(nket .* abs2(ψ) for (nket,ψ) ∈ zip(kets_subs[gs_sub],H_eigen[gs_sub].vectors[:, 1]))
n_gs

function find_block(A::SparseMatrixCSC)
    @assert size(A)[1] == size(A)[2]
    xs,ys,zs = findnz(A)
    blocks = Set{Int}[Set([xs[1],ys[1]])]    
    for (x,y) ∈ zip(xs,ys)
        newblk = true
        for s ∈ blocks
            if (x ∈ s || y ∈ s)
                push!(s,x)
                push!(s,y)
                newblk = false
                break
            end
        end
        if newblk
            push!(blocks,Set([x,y]))
        end
    end
    return blocks
end

blks = find_block(H)
inters = (blks|>permutedims) .∩ blks
inters
count(x->~isempty(x), inters)

length.(blks) |> sum
using Plots
length.(blks) |> sort |> plot