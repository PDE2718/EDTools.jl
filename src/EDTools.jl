module EDTools

#####
# Exact-Diagonalization toolkits for babies
# 1. constructing many-body operators, hamiltionian
# 2. quantum dynamics
# 3. partial trace and correlation

using Combinatorics
using LinearAlgebra
using SparseArrays
# using SuiteSparseGraphBLAS
# import Base.conj
# import Base.*
# import Base.Matrix
# import LinearAlgebra.adjoint

include("ladders.jl")
include("basis.jl")
include("operators.jl")
include("entropy.jl")
include("conventions.jl")
include("extratools.jl")

for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

end # module EDToolKits