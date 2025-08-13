function sumi(f, s :: T, range) :: T where T
	for i in range
		s += f(i)
	end
    return s
end

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
		tr = ε .+ (2 - 2ε) * rand(T, n)
		b = rand(T, n)
	else
		tr = T.(2(1:n)/(n+1))
		b = ones(T, n)
	end	

	map_diag! = (x_out, x_in) -> @. x_out = x_in - tr .* x_in + 1.
	obj_diag(x) = sumi(i -> x[i] * (0.5tr[i] * x[i] - one(T)), zero(T), 1:n)

	return (x0 = zeros(n), map! = map_diag!, obj = obj_diag)
end
