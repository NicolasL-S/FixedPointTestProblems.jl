# Generating data from two two-dimensional gaussian processes
@inline function update_cov!(cov, μ, Z, data, sum_Z)
	cov1 = cov2 = cov3 = 0
    for i in axes(data,1)
        d1 = data[i,1] - μ[1]
        d2 = data[i,2] - μ[2]
        cov1 += d1^2 * Z[i]
        cov2 += d1 * d2 * Z[i]
        cov3 += d2^2 * Z[i]
    end
    cov[1] = cov1 / sum_Z
    cov[2] = cov2 / sum_Z
    cov[3] = cov3 / sum_Z
end

function make_spd!(M, cov; tol = 1e-7)
	M .= Hermitian([cov[1] cov[2];cov[2] cov[3]])
	vals, vecs = eigen(M)
	if minimum(vals) < tol
		vals .= max.(vals, tol)
		M .= Hermitian(vecs * diagm(vals) * vecs')
	end
	return M
end

@inline function safe_dists(θ, containers)
	μ_1, cov_1, covar_1, μ_2, cov_2, covar_2, μ_3, cov_3, covar_3 = containers

    # We will use the convention that θ's 11 entries are (μ_1, cov_1, μ_2, cov_2, μ_3, cov_3, τ1, τ2).
    μ_1 .= θ[1:2]; cov_1 .= θ[3:5]
    μ_2 .= θ[6:7]; cov_2 .= θ[8:10]
    μ_3 .= θ[11:12]; cov_3 .= θ[13:15]

	Hcovar_1 = make_spd!(covar_1, cov_1)
	Hcovar_2 = make_spd!(covar_2, cov_2)
	Hcovar_3 = make_spd!(covar_3, cov_3)

    tol = 1e-4
	τ = (θ[16], θ[17], 1 - θ[16] - θ[17])
	minimum(τ) < tol && (τ = max.(τ, tol) .* (1 / sum(max.(τ, tol))))

    return MultivariateNormal(μ_1,Hcovar_1), 
        MultivariateNormal(μ_2,Hcovar_2), 
        MultivariateNormal(μ_3,Hcovar_3), τ
end

function update_θ!(updated_θ, θ, data, Z1, Z2, Z3, containers)
    md_1, md_2, md_3, (τ1, τ2, τ3) = safe_dists(θ, containers)

	μ_1, cov_1, _, μ_2, cov_2, _, μ_3, cov_3, _ = containers

    # Getting Z
    for i in axes(data, 1)
		pdf_1 = pdf(md_1, data[i,:])
		pdf_2 = pdf(md_2, data[i,:])
		pdf_3 = pdf(md_3, data[i,:])
		den_inv = 1/(τ1 * pdf_1 + τ2 * pdf_2 + τ3 * pdf_3)
		Z1[i] = τ1 * pdf_1 * den_inv
		Z2[i] = τ2 * pdf_2 * den_inv
		Z3[i] = τ3 * pdf_3 * den_inv
    end

    sum_Z1 = sum(Z1)
    sum_Z2 = sum(Z2)
    sum_Z3 = sum(Z3)
    μ_1 .= data'Z1 ./ sum_Z1
    μ_2 .= data'Z2 ./ sum_Z2
    μ_3 .= data'Z3 ./ sum_Z3
    update_cov!(cov_1, μ_1, Z1, data, sum_Z1)
    update_cov!(cov_2, μ_2, Z2, data, sum_Z2)
    update_cov!(cov_3, μ_3, Z3, data, sum_Z3)

    updated_θ[1:2] .= μ_1
    updated_θ[3:5] .= cov_1
    updated_θ[6:7] .= μ_2
    updated_θ[8:10] .= cov_2
    updated_θ[11:12] .= μ_3
    updated_θ[13:15] .= cov_3
    updated_θ[16] = sum_Z1 / length(Z1)
    updated_θ[17] = sum_Z2 / length(Z2)
    return updated_θ
end

function ll(θ, data, containers)
    md_1, md_2, md_3, (τ1, τ2, τ3)  = safe_dists(θ, containers)

	ll = 0.
    for i in axes(data, 1)
		ll += log(τ1 * pdf(md_1, data[i,:]) + τ2 * pdf(md_2, data[i,:]) + τ3 * pdf(md_3, data[i,:]))
	end
    return ll
end

"""
Expectation maximization to parametrize a mixture of 3 bivariate normal distributions.
The code is adapted from the applications of FixedPointAcceleration (which has 2 normals).
See https://s-baumann.github.io/FixedPointAcceleration.jl/dev/4_Applications/ for an in-depth explanation.

Keyword arguments:
; randomize = false, true_τ1 = 0.3, true_τ2 = 0.2, N = 1000, T = typeof(1.)
where true_τ1 and true_τ2 are the proportion of points in the 1st and 2nd distribution.
"""
function gen_em_mixture_of_3_normals(; randomize = false, true_τ1 = 0.3, true_τ2 = 0.2, N = 1000, T = typeof(1.))
	n1 = Int(ceil(true_τ1*N))
	n2 = Int(ceil(true_τ2*N))
	n3 = N - n1 - n2
    
    # Generating data from two two-dimensional gaussian processes
    # Note: these serve to generate the data if random_start == true. Later, they serve as cache.
    # Eventually, other parameters like μ and cov could be set by kwargs

    μ_1 = T[0.0,8.0]
    cov_1 = T[2.0, 0.5, 2.0]
    covar_1 = Hermitian([cov_1[1] cov_1[2]; cov_1[2] cov_1[3]])
    md_1 = MultivariateNormal(μ_1, covar_1)
    μ_2 = T[-4.0,10.0]
    cov_2 = T[2.0,-0.5,12.0]
    covar_2 = Hermitian([cov_2[1] cov_2[2]; cov_2[2] cov_2[3]])
    md_2 = MultivariateNormal(μ_2, covar_2)
    μ_3 = T[-3.,2]
    cov_3 = T[4.0,2.,3.0]
    covar_3 = Hermitian([cov_3[1] cov_3[2]; cov_3[2] cov_3[3]])
    md_3 = MultivariateNormal(μ_3, covar_3)

    if randomize
        data = vcat(rand(md_1, n1)', rand(md_2, n2)', rand(md_3, n3)')
    else
        init_stable_rand()
        data = vcat(gen_2normals(md_1, n1)', gen_2normals(md_2, n2)', gen_2normals(md_3, n3)')
    end

    #sc = scatter(data[:,1], data[:,2]) #To visualize the data

    # Initializing Z
    Z1 = Array{T}(undef,N)
    Z2 = similar(Z1)
	Z3 = similar(Z1)
    r = randomize
    μ_1_guess = T[0.0 + r * (1. - 2rand()),7.5 + r * (1. - 2rand())]
    cov_1_guess = T[2.0 + r * rand(), 0., 1 + r * rand()]
    μ_2_guess = T[-4.0 + r * (1. - 2rand()),12.0 + r * (1. - 2rand())]
    cov_2_guess = T[2.0 + r * rand(),0,2.0 + r * rand()]
    μ_3_guess = T[-2.0 + r * (1. - 2rand()),0. + r * (1. - 2rand())]
    cov_3_guess = T[4.0 + r * rand(),0,2.0 + r * rand()]
    τ1_guess = T(0.4 + r * 0.1 * (1. - 2rand()))
    τ2_guess = T(0.2 + r * 0.1 * (1. - 2rand()))
	x0 = T[μ_1_guess;cov_1_guess;μ_2_guess;cov_2_guess;μ_3_guess;cov_3_guess;τ1_guess; τ2_guess] # eyeballing the parameters

	containers = (μ_1, cov_1, covar_1, μ_2, cov_2, covar_2, μ_3, cov_3, covar_3)
    map! = (updated_θ, θ) -> update_θ!(updated_θ, θ, data, Z1, Z2, Z3, containers)
    obj = θ -> -ll(θ, data, containers) # Negative log likelihood since we are minimizing
    return (x0 = x0, map! = map!, obj = obj)
end