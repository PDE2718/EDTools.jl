# density_matrix
@inline function density_matrix(ψ)
    return Hermitian(ψ * ψ')
end

function rdm_generator(ϕ::FBbasis, sites::Vector{Int})
    @assert sites == unique(sites)
    A = falses(ϕ.n)
    A[sites] .= true
    B = (A .== false)
    ϕA = unique(k[A] for k ∈ ϕ.kets)
    ϕB = unique(k[B] for k ∈ ϕ.kets)
    idA = Dict(ϕA[i] => i for i ∈ eachindex(ϕA))
    idB = Dict(ϕB[i] => i for i ∈ eachindex(ϕB))
    nA = length(ϕA)
    N = ϕ.N
    ψids = Tuple{Int,Int}[(idA[k[A]], idB[k[B]]) for k ∈ ϕ.kets]
    ρids = LinearIndices((nA,nA))
    sumids = Tuple{Int,Int,Int}[]
    ip = iq = rp = rq = 0
    @inbounds for p ∈ 1:ϕ.N, q ∈ 1:ϕ.N
        ip, rp = ψids[p]
        iq, rq = ψids[q]
        if rp == rq
            push!(sumids, (p,q,ρids[ip,iq]))
        end
    end
    function ρA(ψ)
        @assert length(ψ) == N
        ρ = zeros(ComplexF64, nA, nA)
        for (i,j,k) ∈ sumids
            ρ[k] += ψ[i] * ψ[j]'
        end
        return ρ
    end
    return ρA
end

xlogx(x::Real) = x>0 ? x*log(x) : 0.0
function vonNeumann_entropy(ρ)
    λ = eigvals(ρ)
    S = -sum(xlogx, λ)
    # S = -tr(ρ*log(ρ))
    return S
end
