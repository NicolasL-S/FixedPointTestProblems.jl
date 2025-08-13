# Canonical polyadic decomposition (CPD) by alternating least squares (ALS)
# Adapts the ALS solution from TensorDecompositions.jl
# https://github.com/yunjhongwu/TensorDecompositions.jl/blob/master/src/candecomp.jl

matsq!(B, A) = mul!(B, A', A)

function gen_collinear_tensor(sizet, rTrue, T, randomize)
    c = 0.9 # collinearity
    l1 = l2 = 1 # magnitude of the types of noise

    K = c * ones(rTrue,rTrue) + (1 - c) * I

    C = factorize(K)
    U = Matrix{T}[]
    local N1, N2
    for _ in 1:3
        if randomize
            M = randn(sizet, rTrue)
            N1 = randn(sizet,sizet,sizet)
            N2 = randn(sizet,sizet,sizet)
        else
			M = stable_rand(sizet,rTrue; T, D = Normal())
            N1 = stable_rand(sizet,sizet,sizet; T, D = Normal())
            N2 = stable_rand(sizet,sizet,sizet; T, D = Normal())
        end
        Q,_ = qr(M)
        push!(U, Q * C.U)
    end
    Z = full(ktensor(U))

    nZ = norm(Z)
    nN1 = norm(N1)

    Zprime = Z + 1 / sqrt(100/l1 - 1) * nZ / nN1 * N1 # modify Z with the two different types of noise
    N2Zprime = N2 .* Zprime
    return Zprime + (1/sqrt(100/l2 - 1) * norm(Zprime) / norm(N2Zprime)) * N2Zprime
end

function one_ALS_iter!(factors_out, factors_in, V, unfolds, gram, lbds)
	r = length(gram)
	Nf = length(factors_in) ÷ r

	factors_out .= factors_in
	factors = ntuple(i -> reshape(view(factors_out, Nf * (i - 1) + 1:Nf * i), Nf ÷ r,r),r)
	broadcast(matsq!, gram, factors)
	for i in 1:r
		idx = [r:-1:i + 1; i - 1:-1:1]
		factors[i] .= (unfolds[i] * khatrirao!(V, factors[idx])) / reduce((x, y) -> x .* y, gram[idx])
		factors[i] ./= sum!(abs, lbds, factors[i])
		i < r && matsq!(gram[i], factors[i]) # Needless to do it for i = r since we redo all of them at the next iteration
	end
end

"""
Canonical polyadic decomposition (CPD) by alternating least squares (ALS).
It adapts the ALS algorithm from TensorDecompositions.jl by applying one single iteration as map.
Source: https://github.com/yunjhongwu/TensorDecompositions.jl/blob/master/src/candecomp.jl

Keyword arguments:
; sizet = 50, r = 3, T = typeof(1.), randomize = false

where
sizet is the side of one dimension of the tensor
r is the rank.
"""
function gen_als_canonical_tensor(; sizet = 50, r = 3, T = typeof(1.), randomize = false)
    !randomize && init_stable_rand()
    tnsr = gen_collinear_tensor(sizet, r, Float64, randomize)

    x0 = randomize ? rand(r^2 * sizet) : stable_rand(r^2 * sizet)
	
    gram = ntuple(i -> Matrix{T}(undef, r, r), r)
	unfolds = ntuple(i -> _row_unfold(tnsr, i), r)
    lbds = Matrix{T}(undef, 1, r)
    V = Matrix{Float64}(undef, length(tnsr) ÷ minimum(size(tnsr)), r)

	map! = (factors_out, factors_in) -> one_ALS_iter!(factors_out, factors_in, V, unfolds, gram, lbds)

    return (x0 = x0, map! = map!, obj = nothing)
end