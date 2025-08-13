function gen_problem_zero(nx, ny, T)

    hx_sq = 1/(T(nx + 1))^2        # steps dx squared
    hy_sq = 1/(T(ny + 1))^2        # steps dy squared

    Tmat = spdiagm(-1 => -ones(T,nx - 1) ./ hx_sq, 0 => 2 * (1 / hx_sq + 1 / hy_sq) * ones(T,nx), 
		1 => -ones(T, nx - 1) ./ hx_sq)
    S = -spdiagm(-1 => ones(T, ny - 1), 1 => ones(T,ny - 1))

    A = spdiagm(0 => ones(T, ny)) ⊗ Tmat .+ S ⊗ (spdiagm(0 => ones(T, nx)) ./ hy_sq)

    return A, zeros(T, nx * ny)
end

function map_Bratu_precond!(x_out, x_in, A, b, precond)
    # Precondition: so Picard also converges. Note: precond = 1 ./ diag(A)
    x_out .= precond .* (b .- A * x_in .+ 6 .* exp.(x_in)) .+ x_in
end

"""
The Bratu problem

The standard Liouville-Bratu-Gelfand equation is a nonlinear version of the Poisson
equation, described as 
```
                           Δu + λeᵘ = 0
```
where ``u`` is a function ``(x,y) ∈ D = [0,1]²`` and ``λ`` is a constant physical parameter. Dirichlet 
boundary conditions are applied such that ``u(x, y) = 0`` on the boundary of ``D``. The specification is 
the same as in https://arxiv.org/abs/2408.16920. It is solved using the inverse of the discrete 
Laplace operator as preconditioner to make sure the Picard iteration converges.
The default is a 100 × 100 grid.

Keyword arguments:
; nx = 100, ny = 100, randomize = false, T = typeof(1.)
"""
function gen_bratu(;nx = 100, ny = 100, randomize = false, T = typeof(1.)) # Also possible  nx = ny = #8 #32 #64
	A, b = gen_problem_zero(nx, ny, T)
	prec = 1 ./ diag(A)
    x0 = randomize ? rand(T, nx * ny) : zeros(T, nx * ny)
    return (x0 = x0, map! = (x_out, x_in) -> map_Bratu_precond!(x_out, x_in, A, b, prec), obj = nothing)
end