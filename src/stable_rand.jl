function _stable_rand(prev_r, dims...; T = typeof(1.), D = Uniform())
	r = prev_r[1]
	out = Array{T}(undef, dims)
	for i in eachindex(out)
		r = (r * 3685477807 + 12345) ÷ 65536 % 32768
		out[i] = quantile(D, r / 32768.)
	end
	prev_r[1] = r
	return out
end

function gen_2normals(md :: FullNormal, n)
	length(md.μ) == 2 || throw(ArgumentError("Supply a 2-dimensional, full normal distribution."))
	σ1 = √md.Σ[1,1]
	σ2 = √md.Σ[2,2]
	σ12 = md.Σ[1,2]
	μ1 = md.μ[1]
	μ2 = md.μ[2]
	out = stable_rand(2,n ; D = Normal())
	a = σ12 / σ1^2
	b = √(σ2^2 - σ12^2 / σ1^2)
	for i in axes(out,2)
		out[1,i] = σ1 * out[1,i] + μ1
		out[2,i] = out[1,i] * a + out[2,i] * b + μ2 - a * μ1
	end
	return out
end

function gen_rand_functions()
	prev_r = [1]
	return (dims...; T = typeof(1.), D = Uniform()) -> _stable_rand(prev_r, dims...; T = T, D = D), 
		() -> begin prev_r[1] = 1; return nothing end
end

stable_rand, init_stable_rand = gen_rand_functions()