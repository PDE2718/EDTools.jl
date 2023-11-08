using Revise, EDTools, LinearAlgebra, SparseArrays, Graphs, DataFrames

begin
L = (3,4)
Lids = LinearIndices(L)
nbs = [
    [LinearIndices(L)[pbcshift(p,s,L)]
    for s ∈ [(-1, 0), (1, 0), (0, 1), (0, -1)]] |> sort
    for p ∈ CartesianIndices(L)
]
end

ϕ = FBbasis(prod(L), 0, :hcboson, false)
ni = [densities(i;ϕ=ϕ) for i ∈ Lids];
bi = [annihilation(i; ϕ=ϕ) for i ∈ Lids];
b̂i = [creation(i; ϕ=ϕ) for i ∈ Lids];
all_ops = vcat(ni[:],bi[:],b̂i[:])
for op ∈ all_ops
    matrix!(op)
end
N_op = sum(ni)

β = 20.0
μ = 0.8
ξ = 0.005

function Pk(i::Int, ξ)
    Λ = nbs[i]
    Λ = sum(prod((m==l ? ni[m] : (I-ni[m])) for m ∈ Λ) for l ∈ Λ)
    return (I-ξ) * Λ + ξ * (I-Λ)
end
P1nb = [Pk(i, ξ) for i ∈ Lids]

num_ψ = 50
function E_ρ_μ(μ)
    H = -sum(P1nb[i] * (bi[i] + b̂i[i]) + μ * ni[i] for i ∈ Lids)
    H = Symmetric(Matrix(H))
    Es, ψs = eigen(H,1:num_ψ)
    Es ./= prod(L)
    ρs = [dot(ψ,N_op,ψ) for ψ ∈ eachcol(ψs)] ./ prod(L)
    return [E=>ρ for (E,ρ) ∈ zip(Es,ρs)]
end
μgrid = LinRange(-3,2,50)
res = E_ρ_μ.(μgrid)

Es = [[x[i][1] for x ∈ res] for i ∈ 1:num_ψ]
ρs = [[x[i][2] for x ∈ res] for i ∈ 1:num_ψ]
using Plots

plot(μgrid,Es, xlims = (-3,3))
scatter(μgrid,ρs,labels = nothing, marker = (3,0.2,stroke(0)), xlims = (-3,3))
plot!(μgrid,ρs[1],marker = (5,), color = 1)
scatter(ρs,Es)

E_ρ_μ(0.0)


E_ρ_μ(0.780)
E_ρ_μ(0.785)

H = -sum(P1nb[i] * (bi[i] + b̂i[i]) + μ * ni[i] for i ∈ Lids)

eigvals
Hd = Symmetric(Matrix(H))
Es, ψs = eigen(Hd)
ExpβE = normalize( exp.(- β .* BigFloat.(Es)), 1)
ET = dot(Es,ExpβE) ./ prod(L)
ρT = sum(ExpβE[i] * dot(ψs[:,i], N_op, ψs[:,i]) for i ∈ eachindex(ExpβE))./prod(L)

# -0.51819 & 0.3124


ψgs = ψs[:,1]



dropzeros!(H)
H_g = SimpleGraph(ϕ.N)
for (x,y,z) ∈ zip(findnz(H)...)
    add_edge!(H_g,x,y)
end
H_g
H_subspace = connected_components(H_g)
H_subspace = sort(filter(h -> length(h) > 50, H_subspace), by=length, rev=true)
length.(H_subspace)

# subspace_perm = vcat(H_subspace...)
kets_subs = [ϕ.kets[s] for s ∈ H_subspace]
H_mat_subs = [Matrix(H[s, s]) |> Symmetric for s ∈ H_subspace]
H_eigen = [eigen(h, -prod(L)*0.6, -prod(L)*0.3) for h ∈ H_mat_subs]
E_spec = [F.values for F ∈ H_eigen]
id_subs = [fill(i,size(E_spec[i])) for i ∈ eachindex(E_spec)]
GSinfo = [F.values[1] => F.vectors[:,1] for F ∈ H_eigen]

using Plots, Measures, LaTeXStrings
plt1 = scatter(id_subs, (1/16) .* E_spec, labels = nothing,
    marker = (4,0.5,stroke(2)),
    shape = :hline, markeralpha = 0.5, lw=2,
    xlabel = "subspace ID (sorted by size)",
    ylabel = "Energy per site"
)
plt1 |> display

@inline function udlr(i::Int)
    r0::Int = c0::Int = 4
    N::Int = r0 * c0
    c::Int, r::Int = divrem(i - 1, r0)
    return [
        r == 0 ? (i + r0 - 1) : (i - 1),
        r == r0 - 1 ? (i - r0 + 1) : (i + 1),
        c == 0 ? (i - r0 + N) : (i - r0),
        c == c0 - 1 ? (i + r0 - N) : (i + r0),
    ]
end
function num_flip(ψ::BitVector)
    n = 0
    for i ∈ eachindex(ψ)
        if sum(ψ[udlr(i)]) == 1
            n += 1
        end
    end
    return n
end
using FFTW,Statistics

function Sk(ψ::BitVector)
    ϕ = reshape(Float64.(ψ),L)
    return abs2.(fft(ϕ))
end
function Sk(kets::Vector{BitVector},ψ)
    S = zeros(L)
    for (c,ϕ) ∈ zip(ψ,kets)
        S += abs2(c) * Sk(ϕ)
    end
    return S
end
function n̄cal(kets::Vector{BitVector}, ψ)
    ni = 0.0
    for (c, ϕ) ∈ zip(ψ, kets)
        ni += abs2(c) * mean(ϕ)
    end
    return ni
end
function pbcfill(A)
    B = fftshift(A)
    B = vcat(B,B[1:1,1:end])
    B = hcat(B,B[1:end,1:1])
    return B
end

subid_cur = 1
whichlevel = 1
ψgs_cur = H_eigen[subid_cur].vectors[:, whichlevel]
Egs_cur = H_eigen[subid_cur].values[whichlevel] / prod(L)
kets_cur = kets_subs[subid_cur]
Skninj = Sk(kets_cur,ψgs_cur)
ninj = real.(ifft(Skninj)) / prod(L)
nbar = n̄cal(kets_cur, ψgs_cur)
δniδnj = (ninj .- nbar^2)
zdata =  δniδnj |> pbcfill
pbcgrid(n::Int) = -(n÷2):(n÷2)
nbar
rd6(x) = round(x; digits=6)

plt_correlation = heatmap( size = (450,410),
    # 1:5,
    # 1:5,
    zdata,
    xlabel = L"\Delta r",
    # title = L"\langle \delta n(r) \cdot \delta n(r+\Delta r) \rangle",
    title = "E = $(Egs_cur |> rd6), n = $(nbar |> rd6)",
    clims = (-0.15,0.15),
    # xlims = 2.5 .*(-1,1),
    # ylims = 2.5 .*(-1,1),
    c=cgrad(:RdBu_9, rev=true),
    aspect_ratio = :equal, framestyle = :box,
    margin = 0mm,
    right_margin = 8mm,
)

using BenchmarkTools

firstfew = (0,100)
scatter(log10.(length.(H_subspace)), xlims = firstfew)
scatter([v.values[1] for v ∈ H_eigen],
    xlims = firstfew,
)

begin
nth_sub = 10
flipsite_num = [num_flip(ψ) for ψ ∈ kets_subs[nth_sub]]
amp_occ = abs2.(H_eigen[nth_sub].vectors[:,1])
flipsite_bin = 0:maximum(flipsite_num)
flipsite_amp = zeros(size(flipsite_bin))
for (b,a) ∈ zip(flipsite_num, amp_occ)
    flipsite_amp[b] += a
end
scatter(flipsite_bin, flipsite_amp)
end

scatter(flpsites,amp_occ)
histogram2d(amp_occ,flpsites)

subspace_info = DataFrame(
    subspace_size=length.(H_subspace),
    sub_gs_energy=[H_eigen[i].values[1] for i ∈ eachindex(H_subspace)]
)


gs_ε = subspace_info[!,:sub_gs_energy][1] / prod(L)

# gs_ε = subspace_info[!,:sub_gs_energy][2] / prod(L)
gs_n = let N̂ = Vector(N_op[diagind(N_op)])[H_subspace[1]] |> Diagonal, ψgs = H_eigen[1].vectors[:, 1]
    dot(ψgs, N̂, ψgs) / prod(L)
end

kets_subs[1][1][Lids]
kets_subs[1][400][Lids]
kets_subs[1][401][Lids]


subspace_info[1:10,:] |> display

kets_subs[2][10][Lids]
kets_subs[1][10][Lids]


ψ0 = kets_subs[2][10][Lids]
subspace_gen(ψ0)

function subspace_gen(ψ0)
    tocheck = Set([ψ0])
    checked = empty(tocheck)
    while ~isempty(tocheck)
        conf = pop!(tocheck)
        for i ∈ Lids
            conf[i] ⊻= true
            if count(conf[nbs[i]]) == 1 && conf ∉ checked
                push!(tocheck, deepcopy(conf))
            end
            conf[i] ⊻= true
        end
        push!(checked, deepcopy(conf))
    end
    return checked
end

# function car_ind(i,rows,cols)
#     r = mod1(i,rows)
#     c = i ÷ rows + 1
#     return r,c
# end
# function lin_ind(r,c,rows,cols)
#     i = (c-1) * rows + r
#     return i
# end
# function nbid(i, rows, cols)
#     r, c = car_ind(i, rows, cols)
#     return (
#         lin_ind(mod1(r + 1, rows), c, rows, cols),
#         lin_ind(mod1(r - 1, rows), c, rows, cols),
#         lin_ind(r, mod1(c + 1, cols), rows, cols),
#         lin_ind(r, mod1(c - 1, cols), rows, cols),
#     )
# end

# using BenchmarkTools
# @btime nbid(5,10,10)