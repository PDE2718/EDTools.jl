struct FBbasis
    b::Dict{BitVector,Int} # Basis ket => number Dict
    kets::Vector{BitVector} # Basis kets
    n::Int                  # total site number
    k::Int                  # occupied site number / total particles
    N::Int                  # Hilbert space size
    ε::Int                  # sign, +1 for hard core boson / -1 for fermion
    stype::Symbol           # :hcboson / :fermion
    conservation::Bool      # 
end

function bvector(m::Int, n::Int)::BitVector
    bv = falses(n)
    @inbounds for i ∈ 1:n
        bv[i] = isodd(m)
        m >>= 1
    end
    return bv
end
function bvector(bstring::String)::BitVector
    return [x for x ∈ bstring] .== '1'
end

# :hcboson - hard core boson | :fermion - fermion
# conservation - true for the subspace with fixed particle number, false for the full Fock space
function FBbasis(kets::Vector{BitVector}, stype::Symbol, conservation = false)
    n = length(kets[1])
    N = length(kets)
    k = 0
    b = Dict(kets[i] => i for i ∈ eachindex(kets))
    ε = (stype == :fermion) ? (-1) : (+1)
    return FBbasis(b, kets, n, k, N, ε, stype, conservation)
end

function FBbasis(n::Int, k::Int, stype::Symbol, conservation::Bool=true)
    @assert stype == :hcboson || stype == :fermion
    @assert (k == 0) ⊻ conservation
    # --------------------------------------------
    kets = if conservation
        BitVector[
            let v = falses(n)
                v[q] .= true
                v
            end for q ∈ CoolLexCombinations(n, k)
        ]
    else
        BitVector[bvector(m, n) for m ∈ 0:(2^n-1)]
    end
    sort!(kets,by=count)
    b = Dict(kets[i] => i for i ∈ eachindex(kets))
    N = length(kets)
    ε = (stype == :fermion) ? (-1) : (+1)
    return FBbasis(b, kets, n, k, N, ε, stype, conservation)
end
