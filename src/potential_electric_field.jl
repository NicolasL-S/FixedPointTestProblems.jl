function jacobi!(x_out, x_in, sourceSq)

    dims = size(sourceSq)
    N = dims[1]
    x_inSq = reshape(x_in, dims)
    x_outSq = reshape(x_out, dims)

    h_sq = 1 / (N - 1)^2
    # Only iterate over interior points thus keeping the edges untouched and hence enforcing 
    # the boundary condition that x = 0 at the edges.

    x_outSq[begin,:] .= 0
    x_outSq[end,:] .= 0
    x_outSq[:,begin] .= 0
    x_outSq[:,end] .= 0
    @inbounds for i in 2:N-1
        for j in 2:N-1
            x_outSq[j, i] = 0.25 * (x_inSq[j+1, i] + x_inSq[j, i+1] + x_inSq[j-1, i] + x_inSq[j, i-1] + h_sq * sourceSq[j, i])
        end
    end
    return x_out
end

function gauss_seidel!(x_out, x_in, sourceSq)
	dims = size(sourceSq)
    x_outSq = reshape(x_out, dims)
	N, M = dims
	N == M || throw(DimensionMismatch("The Input array should be square."))
    h_sq = 1/(N-1)^2
    x_out .= x_in
    @inbounds for i in 2: N - 1, j in 2: N - 1
		x_outSq[j, i] = 0.25 * (x_outSq[j+1, i] + x_outSq[j, i+1] + x_outSq[j-1, i] + x_outSq[j, i-1] 
			+ h_sq*sourceSq[j, i])
    end
    return x_out
end

function SOR!(x_out, x_in, sourceSq)
    w = 1.94
    dims = size(sourceSq)
    x_inSq = reshape(x_in, dims)
    x_outSq = reshape(x_out, dims)
    N = dims[1]
    h_sq = 1/(N-1)^2

    #h = 1/(N-1)
    x_outSq .= x_inSq
    for i in 2:N - 1
        for j in 2:N - 1
            new = 0.25 * (x_outSq[j-1, i] + x_outSq[j+1, i] + x_outSq[j, i-1]+ x_outSq[j, i+1] + h_sq * sourceSq[j, i])
            x_outSq[j, i] += w * (new - x_outSq[j, i])
        end
    end
    return x_out
end

function init_problem(;N = 50, p1 = 0.25, p2 = 0.75, p3 = 0.25, p4 = 0.75, p5 = 0.25, p6 = 0.75, T = typeof(1.))
    x0 = zeros(T, N, N)
    h = 1/(N-1)
    source_col1, source_col2 = Int.(floor.((p1, p2) .* N))
    st_col1, end_col1, st_col2, end_col2 = Int.(floor.((p3, p4, p5, p6) .* N))

    x0[st_col1:end_col1, source_col1] .+= 1/(N * 0.5 * h^2)
    x0[st_col2:end_col2, source_col2] .-= 1/(N * 0.5 * h^2)
    return x0
end

"""
Accelerating the Gauss Seidel, Jacobi or SOR methods to solve a Poisson Equation (elliptic Partial 
Differential Equation)
Source of the example: 
https://github.com/numericalalgorithmsgroup/NAGPythonExamples/blob/master/roots/Anderson_Acceleration_Poisson.ipynb

Produces a N×N matrix corresponding to a source term with two line sources with total charge 
of +/- 1 on each line. p1 to p6 determine the positions of the two lines.

Keyword arguments:
; n = 100, randomize = false, T = typeof(1.), method! = gauss_seidel!
"""
function gen_potential_electric_field(; n = 100, randomize = false, T = typeof(1.), algo! = gauss_seidel!) # algo! can be jacobi!, gauss_seidel! or SOR!
    p1, p2, p3, p4, p5, p6 = (0.25, 0.75, 0.25, 0.75, 0.25, 0.75) .+ randomize * 0.1 .* (1. .- 2rand(6))
	sourceSq = init_problem(;n, p1, p2, p3, p4, p5, p6, T)
	return (x0 = vec(copy(sourceSq)), map! = (x_out, x_in) -> algo!(x_out, x_in, sourceSq), 
        obj = nothing)
end



