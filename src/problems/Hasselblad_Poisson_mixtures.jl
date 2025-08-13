# Making that the input parameters are sound by enforcing 0 ≤ p ≤ 1, μ₁, μ₂ ≥ 0.
function safe_input(x)
	tol = 1e-7
	p, μ₁, μ₂ = x
	p < tol && (p = tol)
	p > 1 - tol && (p = 1 - tol)
	μ₁ < tol && (μ₁ = tol)
	μ₂ < tol && (μ₂ = tol)
	return p, μ₁, μ₂
end

function hasselblad_obj(x, freq)
   p, μ₁, μ₂ = safe_input(x)
   if p < 0. || p > 1. || μ₁ < 0. || μ₂ < 0.
        return NaN
   else
        inv_exp_μ₁ = exp(-μ₁)
        inv_exp_μ₂ = exp(-μ₂)
        yfact = μ₁expy = μ₂expy = 1.
        l = 0.
        for y in eachindex(freq)
            l += freq[y] * log((p * inv_exp_μ₁ * μ₁expy + 
                          (1 - p) * inv_exp_μ₂ * μ₂expy) / yfact)
            yfact *= Float64(y)
            μ₁expy *= μ₁
            μ₂expy *= μ₂
        end
		return l
    end
end

function hasselblad_map!(x_out, x_in, freq, sum_freq)
	p, μ₁, μ₂ = safe_input(x_in)
	if p < 0. || p > 1. || μ₁ < 0. || μ₂ < 0.
		x_out .= NaN
	else
		inv_exp_μ₁ = exp(-μ₁)
		inv_exp_μ₂ = exp(-μ₂)
		sum_freq_z₁ = sum_freq_z₂ = sum_freq_y_z₁ = sum_freq_y_z₂ = 0.0
		μ₁expy = μ₂expy = 1.0
		for i in eachindex(freq)
			y = Float64(i) - 1.
			q = p * inv_exp_μ₁ * μ₁expy
			z = q / (q + (1. - p) * inv_exp_μ₂ * μ₂expy)
			sum_freq_z₁   += freq[i] * z
			sum_freq_z₂   += freq[i] * (1.0 - z)
			sum_freq_y_z₁ += y * freq[i] * z
			sum_freq_y_z₂ += y * freq[i] * (1.0 - z)
			μ₁expy *= μ₁
			μ₂expy *= μ₂
		end
		x_out[1] = sum_freq_z₁ / sum_freq
		x_out[2] = sum_freq_y_z₁ / sum_freq_z₁
		x_out[3] = sum_freq_y_z₂ / sum_freq_z₂
	end
	return x_out
end

"""
Accelerating the EM algorithm for Poisson mixtures
See V. Hasselblad, Estimation of finite mixtures of distributions from the exponential family, 
J. Amer. Statist. Assoc. 64 (328) (1969) 1459–1471.

Same spec (and more info) in Lepage-Saucier (2024). "Alternating cyclic vector extrapolation 
technique for accelerating nonlinear optimization algorithms and fixed-point mapping applications," 
Journal of Computational and Applied Mathematics, 439, 115607.

Keyword arguments:
;T = eltype(1.), x0 = T[0.25, 1., 2.], randomize = false

where x0 is [proportion of the first mixture μ₁, μ₂]

Keyword arguments:
; T = typeof(1.), x0 = T[0.25, 1., 2.], randomize = false
"""
function gen_hasselblad_poisson_mixtures(; T = typeof(1.), x0 = T[0.25, 1., 2.], randomize = false)
	freq = (162, 267, 271, 185, 111, 61, 27, 8, 3, 1) # The data
	sum_freq = sum(freq)
	randomize && (x0 = [0.05 + 0.9rand(), 10rand(), 10rand()])
    return (x0 = x0, 
            map! = (x_out, x_in) -> hasselblad_map!(x_out, x_in, freq, sum_freq), 
            obj = x -> -hasselblad_obj(x, freq))  # Minimizing the objective
end