using Nemo
using AbstractAlgebra          
using StaticArrays

const FmpzMat = Nemo.MatElem{Nemo.ZZRingElem}

function Base.convert(
    ::Type{FmpzMat},
    mat::Union{Matrix{T}, MMatrix{N,N,T}},
) where {N, T<:Real}
    return Nemo.matrix(Nemo.ZZ, BigInt.(round.(mat)))
end

function Base.convert(::Type{MMatrix{N,N,BigInt}}, mat::FmpzMat) where {N}
    A = Matrix{BigInt}(mat)
    return convert(MMatrix{N,N,BigInt}, A)
end

function Base.convert(::Type{Matrix{BigInt}}, mat::FmpzMat)
    return Matrix{BigInt}(mat)
end

"""
    snf_with_transform(L::Lattice)
    snf_with_transform(mat::MMatrix{M,N}) where {M,N}
    snf_with_transform(mat::Matrix)

Wrapper of AbstractAlgebra.snf_with_transform. Input is converted to BigInt.
Returns (S,U,V) such that U*mat*V = S, where S is the Smith normal form of mat.
"""
function snf_with_transform(mat::MMatrix{M,N}) where {M,N}
    mat = convert(FmpzMat, mat)
    (S, U, V) = AbstractAlgebra.snf_with_transform(mat) # U*mat*V = S
    U = convert(MMatrix{size(U)...,BigInt}, U)
    V = convert(MMatrix{size(V)...,BigInt}, V)
    S = convert(MMatrix{size(S)...,BigInt}, S)
    return S, U, V
end

function snf_with_transform(mat::Matrix)
    mat = convert(FmpzMat, mat)
    (S, U, V) = AbstractAlgebra.snf_with_transform(mat) # U*mat*V = S
    U = convert(Matrix{BigInt}, U)
    V = convert(Matrix{BigInt}, V)
    S = convert(Matrix{BigInt}, S)
    return S, U, V
end

"""
    hnf(mat::MMatrix{M,N}) where {M, N}
    hnf(mat::Matrix)

Wrapper of Nemo.hnf. Input is converted to BigInt.
Returns H = mat*U, s.t. H is in Hermite Normal Form and U is unimodular.
"""
function hnf(mat::MMatrix{M,N}) where {M, N}
    mat = convert(FmpzMat, mat)
    H = transpose(Nemo.hnf(transpose(mat)))
    H = convert(MMatrix{size(H)...,BigInt}, H)
    return H
end

function hnf(mat::Matrix)
    mat = convert(FmpzMat, mat)
    H = transpose(Nemo.hnf(transpose(mat)))
    H = convert(Matrix{BigInt}, H)
    return H
end

"""
    lll(mat::MMatrix{M,N}) where {M, N}
    lll(mat::Matrix)

Wrapper of Nemo.lll. Input is converted to BigInt.
Computes L such that mat*T = L for some unimodular T.
"""
function lll(mat::MMatrix{M,N}) where {M, N}
    mat = convert(FmpzMat, mat)
    L = Nemo.lll(transpose(mat))
    L = transpose(L)
    L = convert(MMatrix{size(L)...,BigInt}, L)
    return L
end

function lll(mat::Matrix)
    mat = convert(FmpzMat, mat)
    L = Nemo.lll(transpose(mat))
    L = transpose(L)
    L = convert(Matrix{BigInt}, L)
    return L
end
