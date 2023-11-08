function creation(i::Int; α=I, ϕ=nothing)
    return OpTerm(uuLadder(i); α=α, ϕ=ϕ)
end
function annihilation(i::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(i); α=α, ϕ=ϕ)
end
function hopping(i::Int,j::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(i),uuLadder(j); α=α, ϕ=ϕ)
end
function densities(i::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(i),uuLadder(i); α=α, ϕ=ϕ)
end
function densities(i::Int,j::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(i),uuLadder(i),ddLadder(j), uuLadder(j); α=α, ϕ=ϕ)
end