function sumi(f, s :: T, range) :: T where T
	for i in range
		s += f(i)
	end
    return s
end

obj_diag(x, diago, b) = sumi(i -> x[i] * (0.5x[i] * diago[i] - b[i]), zero(eltype(x)), eachindex(x))
map_diag!(x_out, x_in, one_minus_diago, b) =  @. x_out = one_minus_diago * x_in + b

"""
Solving the linear system of equations
```math
							Ax = b
```
where ``A^(n×n)`` is diagonal with elements ``2(1:n)/(n+1)`` and ``b = ones(n)``, using the mapping
```math
							F(x₁) = x₀ - (Ax₀ - b).
```
The starting point is ``x0 = zeros(n)``.

Keyword arguments:
;n = 100, randomize = false, T = typeof(1.)
"""
function gen_linear(;n = 100, randomize = false, T = typeof(1.))
	if randomize
		ε = T(1/n)
		diago = ε .+ (2 - 2ε) * rand(T, n)
		b = rand(T, n)
	else
		diago = T.(2(1:n)/(n+1))
		b = ones(T, n)
	end	
	one_minus_diago = one(T) .- diago

	return (
		x0 = zeros(n), 
		map! = (x_out, x_in) -> map_diag!(x_out, x_in, one_minus_diago, b), 
		obj = x -> obj_diag(x, diago, b)
	)
end