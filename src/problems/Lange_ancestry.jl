function sample_genotypes(f, q)

	# Get the number of markers (p) and the number of ancestral populations (K).
	p, K = size(f)

	# Independently for each marker, and for each of the two allele copies, draw the population of origin.
	z1 = sample(collect(1:K),aweights(q), p; replace = true)
	z2 = sample(collect(1:K),aweights(q), p; replace = true)

	# For each marker, and for each of the two allele copies, draw the allele, and output the 
	# (unphased) genotype, represented as an allele count.
	out = zeros(p)
	for i in eachindex(out)
		out[i] = (rand() < f[i,z1[i]]) + (rand() < f[i,z2[i]])
	end
	return out
end

function admix_generate(p = 100, K = 3, n = 150; randomize = false)
	# A vector that specifies, for each test individual, the number of populations contributing to 
	# the individual's genome.

	# GENERATE TRAINING AND TEST DATA
	# markers              : p
	# ancestral populations: K
	# training samples     : n

	# Generate, for each ancestral population, the total proportion of chromosomes that are 
	# attributed to that population.
	p_deme = 0.2 .+ 0.6 * (randomize ? rand(K) : stable_rand(K))
	p_deme ./= sum(p_deme)

	# Generate, for each ancestral population, the allele frequencies at at all the markers, 
	# assuming the markers are unlinked.
	f = randomize ? rand(p , K) : stable_rand(p , K)

	# The admixture proportions are stored as an n x k matrix, and the genotypes are stored as an 
	# n x p matrix, where n is the number of samples, p is the number of markers, and k is the 
	# number of ancestral populations.
	q_train    = zeros(n, K)
	geno_train = zeros(n, p)

	# Generate the single-origin training samples.
	for i in 1:n
		# Randomly sample the ancestral population of origin, then sample
		# the genotypes given the population-specific allele frequencies and
		# the ancestral population of origin.
		k = sample(collect(1:K),aweights(p_deme))
		q_train[i,k] = 1
		geno_train[i,:] .= sample_genotypes(f, q_train[i,:])
	end

	return(geno_train)
end

@inline function expit(x)
	expx = exp(x)
	return expx / (1 + expx)
end

function EMadmixture!(param_out, param_in, X, K, m, n0, n1, FF, OneMinusFF, Q, r, r_sums, S1, S2, Xzeros, Xones, Xtwos, FFsq)
	eps = 1e-6
	n, p = size(X)
	m .= eps
	n0 .= eps
	n1 .= eps
	FF .= expit.(reshape(param_in[1: (p * K)], (p, K)))
	OneMinusFF .= 1 .- FF
	Q .= exp.(reshape(param_in[(p * K + 1): (p * K + n * K)], (n, K)))
	Q ./= sum(Q; dims = 2)

    @inbounds for j ∈ 1:K, k ∈ 1:K
        @. FFsq[:,1, j, k] = OneMinusFF[:,j] * OneMinusFF[:,k]
        @. FFsq[:,2, j, k] = OneMinusFF[:,j] *         FF[:,k]
        @. FFsq[:,3, j, k] = FF[:,j]         * OneMinusFF[:,k]
        @. FFsq[:,4, j, k] = FF[:,j]         *         FF[:,k]
    end
    
	@inbounds for i ∈ 1:n
		Xzeros .= X[i,:] .== 0
		Xones  .= X[i,:] .== 1
		Xtwos  .= X[i,:] .== 2
        
		for j ∈ 1:K, k ∈ 1:K
			@. r[:,1, j, k] = Xzeros * FFsq[:,1, j, k]
			@. r[:,2, j, k] = Xones  * FFsq[:,2, j, k]
			@. r[:,3, j, k] = Xones  * FFsq[:,3, j, k]
			@. r[:,4, j, k] = Xtwos  * FFsq[:,4, j, k]
			@. r[:,:, j, k] *= Q[i, j] * Q[i, k]
		end
		for i1 in 1:p
			s = 0
			for i2 in 1:4, i3 in 1:K, i4 in 1:K
				s += r[i1, i2, i3, i4]
			end
			for i2 in 1:4, i3 in 1:K, i4 in 1:K
				r[i1, i2, i3, i4] /= s
			end
		end

		r_sums .= 0
		for i1 in 1:K, i2 in 1:K, i3 in 1:p, i4 in 1:4
			r_sums[i1, i2] += r[i3,i4,i1,i2]
		end
		for l in 1:K, i1 in 1:K
			m[i,l] += r_sums[l,i1] + r_sums[l,i1]
		end

		S1 .= sum(r; dims = 3)
		S2 .= sum(r; dims = 4)
		n0 .+= S2[:,1,:,1] .+ S2[:,2,:,1] .+ S1[:,1,1,:] .+ S1[:,3,1,:]
		n1 .+= S2[:,3,:,1] .+ S2[:,4,:,1] .+ S1[:,2,1,:] .+ S1[:,4,1,:]
	end
	param_out[1:p * K] .= log.(vec(n1 ./ n0))
    param_out[p * K+1:end] .= log.(vec(m))
	return param_out
end

function LogLikAdmixTrans(param, X, K, FF, Q)
  n, p = size(X)
  FF .= expit.(reshape(param[1: (p * K)], (p, K)))
  Q .= exp.(reshape(param[(p * K + 1):(p * K + n * K)], (n, K)))
  Q ./= sum(Q; dims = 2)
  return sum(X .* log.(Q * FF')) + sum((2 .- X) .* log.(Q * (1 .- FF')))
end

function gen_new_problem_ancestry!(starts, Xs, K, p, n)
    Ftmp = Vector{eltype(starts[1])}(undef, p * K)
    for i in eachindex(starts)
        Ftmp .= rand(p * K)
        starts[i][1:p * K] .= log.(Ftmp ./ (1 .- Ftmp))
        starts[i][p * K+1:end] .= rand(n * K)
		Xs[i] .= admix_generate(p, K, n)
    end
end

"""
David H. Alexander, John Novembre, and Kenneth Lange. Fast model-based estimation of ancestry in 
unrelated individuals. Genome Research, 19:1655–1664, July 2009.

This code is adapted to Julia from the R package dareemtest_v3 created for the simulations in
N.C. Henderson, R. Varadhan, Damped Anderson acceleration with restarts and monotonicity control 
for accelerating EM and EM-like algorithms, J. Comput. Graph. Stat. 28 (4) (2019) 834–846. The

By default, the problem is generated with a precomputed randomly generated with parameters p = 100, 
K = 3, n = 150, e = 0.01 to guarantee consistent comparisons.

Keyword arguments:
; K = 3, p = 100, n = 150, T = typeof(1.), randomize = false
"""
function gen_lange_ancestry(; K = 3, p = 100, n = 150, T = typeof(1.), randomize = false)

	start = Vector{T}(undef,(p + n) * K)
	!randomize && init_stable_rand() 
	Ftmp = randomize ? rand(p * K) : stable_rand(p * K)
	start[1:p * K] .= log.(Ftmp ./ (1 .- Ftmp))
	start[p * K+1:end] .= randomize ? rand(n * K) : stable_rand(n * K)

	# The data
	X = admix_generate(p, K, n; randomize)

	# objective (with temp storage)
	FF = Matrix{T}(undef,p, K)
	Q = Matrix{T}(undef,n, K)
	obj = param -> -LogLikAdmixTrans(param, X, K, FF, Q)

	# map! (with temp storage)
	m = Matrix{T}(undef, n, K)
	n0 = Matrix{T}(undef, p, K)
	n1 = Matrix{T}(undef, p, K)
	OneMinusFF = similar(FF)
	r = Array{T}(undef,p, 4, K, K)
	r_sums = Array{T}(undef,K, K)
	S1 = Array{T}(undef,p, 4, 1, K)
	S2 = Array{T}(undef,p, 4, K, 1)
	Xzeros = BitVector(undef, p)
	Xones = similar(Xzeros)
	Xtwos = similar(Xzeros)
	FFsq = Array{T}(undef, p, 4, K, K)
	map! = (param_out, param_in) -> EMadmixture!(param_out, param_in, X, K, m, n0, n1, FF, 
		OneMinusFF, Q, r, r_sums, S1, S2, Xzeros, Xones, Xtwos, FFsq)

	return (x0 = start, map! = map!, obj = obj)
end
