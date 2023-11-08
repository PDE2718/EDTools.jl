# true for creation / false for annihilation
struct Ladder
    site::Int
    flag::Bool
end
Ladder(sf::Tuple{Int,Bool}) = Ladder(sf[1], sf[2])
uuLadder(s::Int) = Ladder(s, true)
ddLadder(s::Int) = Ladder(s, false)
Ladders(ladder...) = (ladder...,)
Base.conj(c::Ladder) = Ladder(c.site, ~c.flag)

# 用来对算符正规排序，同类算符看site，不同类算符降算符更小
function Base.isless(a1::Ladder, a2::Ladder)
    if a1.flag == a2.flag
        return isless(a1.site, a2.site)
    else
        return isless(a1.flag, a2.flag)
    end
end

function apply_Ladder!(v::BitVector, a::Ladder, ε::Int)::Int
    if a.flag == v[a.site]
        return 0
    else
        v[a.site] = a.flag
        return ε^count(v[1:(a.site-1)])
    end
end

function apply_Ladders!(vt::BitVector, v::BitVector, S::NTuple{N,Ladder}, ε::Int)::Int where N
    s = 1
    copy!(vt, v)
    for a ∈ S
        s *= apply_Ladder!(vt, a, ε)
        if s == 0
            return 0
        end
    end
    return s
end

# 一个算符总是系列升降算符的乘积
# struct LadderOp
#     seq_original::Vector{Ladder}     # 原始序列
#     # seq_proper::Vector{Ladder}       # 正规序列
#     # perm::Vector{Int}                # 排序perm
#     # s::Int                           # 正规排序相对原始序列的附带符号，非常重要
#     α::UniformScaling # scaling系数
#     M::Union{Nothing,AbstractMatrix} # 矩阵形式
#     IJV::Union{Nothing,AbstractVector}
#     LadderSequence(seq_original, seq_proper, perm, s, α) = new(
#         seq_original, seq_proper, perm, s, α, nothing, nothing
#     )
# end

# function LadderSequence(seq_original, seq_proper, perm, s, α)
#     LadderSequence(seq_original, seq_proper, perm, s, , nothing)
# end

# function LadderSequence(seq_original::Vector{Ladder}, α::UniformScaling = I)
#     perm = sortperm(seq_original)
#     seq_proper = seq_original[perm]
#     s = levicivita(perm)
#     return LadderSequence(seq_original, seq_proper, perm, s, α)
# end
# function LadderSequence(seq::Vector{Tuple{Int,Bool}}, α::UniformScaling = I)
#     seq_original = Ladder.(seq)
#     return LadderSequence(seq_original, α)
# end

# function printLadder(a::Ladder)
#     if a.flag
#         printstyled(" â_$(a.site) ", color=9)
#     else
#         printstyled(" a_$(a.site) ", color=39)
#     end
# end

# function printLadderSequence(S::LadderSequence)
#     begin
#         print("original :   ")
#         for a in reverse(S.seq_original)
#             printLadder(a)
#         end
#     end
#     print("\n")
#     begin
#         print("proper   :  $(S.s==1 ? '+' : '-')")
#         for a in reverse(S.seq_proper)
#             printLadder(a)
#         end
#     end
#     print("\n")
# end