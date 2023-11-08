using Revise
using EDTools
using LinearAlgebra, SparseArrays

n = 8
k = 4
V = 1
ϕ = FBbasis(n,k,:fermion,true)

begin
H_Ladders = LadderSequence[]
    for j ∈ 1:n
        jp = mod1(j + 1, n)
        cc = hopping_ij(jp,j,-1)
        cc_hc = adjoint(cc)
        nn = densities_ij(j,jp,V)
        push!(H_Ladders, cc)
        push!(H_Ladders, cc_hc)
        push!(H_Ladders, nn)
    end
    for i ∈ eachindex(H_Ladders)
        Matrix!(ϕ,H_Ladders[i])
    end
end

H_sparse = sum(Matrix(x) for x ∈ H_Ladders)
H_dense = H_sparse |> Matrix |> Hermitian
H_eigen = eigen(H_dense)
# F * Diagonal(εs) * F' - H_dense

function evo_t(H::Eigen,ψ0::Vector,t::Real)
    B = H.vectors
    λ = H.values
    ψt = B * (cis.(-t .* λ).*((B')*ψ0))
    return ψt
end
nj = [density_j(j) for j ∈ 1:n]
for u ∈ nj
    Matrix!(ϕ, u)
end
@inline pj1(j,ψ,η) = η * (1 -  ψ⋅(nj[j]*ψ)) |> real
@inline function randMj(j,ψ,η)
    p1 = pj1(j, ψ, η)
    if rand()<p1
        return (√η)I * (I - nj[j].M)
    else
        return (√(1 - η))I + (1 - √(1 - η)) * nj[j].M
    end
end
function randM_projection(ψ,η)
    ψt = deepcopy(ψ)
    for j ∈ 1:n
        ψt = randMj(j,ψt,η) * ψt
    end
    normalize!(ψt)
    if any(isnan, ψt)
        return randM_projection(ψ, η)
    else
        return ψt
    end
end

ψ0 = wf_from_ket(ϕ, repeat(Bool[1, 0], n ÷ 2))
ψ0 = normalize(ψ0 .+ 1e-12randn(size(ψ0)))
# ψ0 = wf_from_ket(ϕ, "10101010")
ρ_half = rdm_generator(ϕ, collect(1:(n÷2)))

τ = 0.01
tgrid = 0:τ:20
Uτ = cis(-τ * H_dense)

function S_traj(g)
    η = g * τ
    S = zeros(length(tgrid))
    ψt = deepcopy(ψ0)
    for i ∈ 2:length(tgrid)
        ψt = Uτ * ψt
        if η ≠ 0
            ψt = randM_projection(ψt, η)
        end
        S[i] = ψt |> ρ_half |> vonNeumann_entropy
    end
    return S
end

# ψgrid = [evo_t(H_eigen, ψ0, t) for t ∈ tgrid]

using Statistics
glist = [0.0, 0.5, 1.0, 2.0, 4.0]
r_resemble = 100
Straj = [[S_traj(g) for i ∈ 1:r_resemble] for g ∈ glist]
Straj_mean = [mean(Straj[i]) for i ∈ eachindex(glist)]

using Plots, Measures, LaTeXStrings
plot(tgrid, Straj_mean,
    ylims = (0,3.5),
    xlims=(0,20),
    # xlabel
    labels = ["g=$(g)" for g ∈ glist] |> permutedims,
    framestyle=:box
)
