function power_iteration!(x_out, x_in, A)
    mul!(x_out, A, x_in)
    x_out ./= maximum(abs, x_out)
end

"""
Power iteration for computing dominant eigenvalues
https://en.wikipedia.org/wiki/Power_iteration

Keyword arguments:
; n = 100, randomize = false, T = typeof(1.)
"""
function gen_power_iteration(; n = 100, randomize = false, T = typeof(1.))
    add_diag = randomize ? collect(1:n) .* rand(n) : 1:n
    x0 = randomize ? rand(n) : ones(T, n)
	A = T.(ones(n,n) .+ Diagonal(add_diag))
    return (x0 = x0, map! = (x_out, x_in) -> power_iteration!(x_out, x_in, A), obj = nothing)
end