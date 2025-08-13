#=
Copyright (c) 2015, Nicholas J. Higham and Nataša Strabić.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

function insert_upper!(A, x, offset; rangex = eachindex(x))
	"""
	Reads the upper-triangular elements of a matrix stored in x and writes them in the upper and
	lower part of A. 
	- offset (≥ 0) counts how far from the diagonal the elements are (offset = 0 means the diagonal
	  elements are included, offset = 1 means the diagonal is excluded.).
	- rangex defines the range of elements (assumed to be contiguous) in the vector x where the 
	  elements are stored.
	"""
	n = Int((-1 + √(1 + 8length(rangex)))/2) + offset
	mA, nA = size(A)
	n == mA == nA || throw(DimensionMismatch("A and x do not have the same number of unique elements"))

	k = first(rangex) - 1
	for j in 1 + offset:nA, i in 1:j - offset
		k += 1
		A[i,j] = x[k,1]
		A[j,i] = x[k,1]
	end
	return A
end

function insert_upper(x, offset; rangex = eachindex(x))
	n = Int((-1 + √(1 + 8length(rangex)))/2) + offset
	return insert_upper!(ones(n,n), x, offset)
end

function extract_upper!(x, A, offset; rangex = eachindex(x))
	"""
	Reads the upper-triangular elements of a matrix A and writes them in the vector x. 
	- offset (≥ 0) counts how far from the diagonal the elements should be read (offset = 0 means 
	  the diagonal elements are included, offset = 1 means the diagonal is excluded.).
	- rangex defines the range of elements (assumed to be contiguous) in the vector x where the 
	  elements are stored.
	"""
	n = Int(ceil(-1 + √(1 + 8length(rangex)))/2) + offset
	mA, nA = size(A) 
	n == mA == nA || throw(DimensionMismatch("A and x do not have the same number of unique elements"))

	k = first(rangex) - 1
	for j in 1 + offset:nA, i in 1:j-offset
		k += 1
		x[k] = A[i,j]
	end
	return x
end

function project_s!(Xout, X; ϵ = 0.)
	ϵ = max(ϵ, 0.)
	λ, P = eigen(X; sortby = x -> -x)
    replace!(x -> max(x, ϵ), λ)
    mul!(Xout, (P .* λ'), P')
    return Xout
end

function projys!(ys_out, ys_in, Y, R, S)
	d = size(Y, 1)
	N = (d * (d + 1)) ÷ 2
	
	insert_upper!(Y, ys_in, 0; rangex = 1:N)
	insert_upper!(S, ys_in, 0; rangex = N + 1:2N)
	
	R .= Y .- S
	project_s!(Y, R)
	S .= Y .- R
	for i in axes(Y,1)
		@inbounds Y[i,i] = 1.
	end

	extract_upper!(ys_out, Y, 0; rangex = 1:N)
	extract_upper!(ys_out, S, 0; rangex = N + 1:2N)
end

"""
Sources for the algotithm:
- Higham, N. J. "Computing the Nearest Correlation Matrix — A Problem from Finance." IMA Journal of 
  Numerical Analysis. Vol. 22, Issue 3, 2002.
- Higham, N. J. and Strabić, N. "Anderson acceleration of the alternating projections method for 
computing the nearest correlation matrix." Numer Algor vol. 72, 2016.

Note that the algorithm implemented below is Algorithm 5 of Higham and Strabić (2016), but the
acceleration is performed only on the upper triangular (and diagonal) elements of the Y and S 
matrices, not the whole matrix as in their paper (since it is symmetric).

- Sources for the matrices
Matrices from https://github.com/higham/matrices-correlation-invalid/tree/master

Keyword arguments:
;problem = :mmb13, T = typeof(1.), n_max = 0, randomize = false, n = 100

where 
- problem can be :fing97, :beyu11, :bhwi01, :fing97, :high02, :mmb13, :tec03, :tyda99r1, :tyda99r2, :tyda99r3, :cor1399, :cor3120, :bccd16, :rocky
- n_max takes the first n_max rows/columns of a matrix
- n determines the size of a matrix when randomize is used.
"""
function gen_higham_corr_matrix(; problem = :mmb13, T = typeof(1.), n_max = 0, randomize = false, n = 100)
	if randomize
		A = ones(n,n)
		for i in 2:n, j in 1:i-1
			r = sqrt(rand())
			A[i,j] = r
			A[j,i] = r
		end
		# Making sure at least 1 eigenvalue is negative
		vals, vecs = eigen(A)
		if minimum(vals) >= 0
			vals[1] = -0.1
			A .= vecs * Diagonal(vals) * vecs'
		end
	else
		if problem == :fing97
			A = T[
			1     0.18 -0.13 -0.26  0.19 -0.25 -0.12;
			0.18  1     0.22 -0.14  0.31  0.16  0.09;
			-0.13  0.22  1     0.06 -0.08  0.04  0.04;
			-0.26 -0.14  0.06  1     0.85  0.85  0.85;
			0.19  0.31 -0.08  0.85  1     0.85  0.85;
			-0.25  0.16  0.04  0.85  0.85  1     0.85;
			-0.12  0.09  0.04  0.85  0.85  0.85  1]

		elseif problem == :beyu11
			A = T[
			1      0.2387 0.6161 0.6167 0.6621 0.5173 0.6758 0.7071 0.7983 0.5769 0.4705 0.7881;
			0.2387 1      0.3506 0.3537 0.2959 0.4637 0.1931 0.1202 0.2316 0.1708 0.4047 0.1161;
			0.6161 0.3506 1      0.8579 0.6603 0.4093 0.3826 0.5164 0.6079 0.5574 0.4512 0.5128;
			0.6167 0.3537 0.8579 1      0.7477 0.1803 0.4705 0.6167 0.6218 0.4705 0.3582 0.2966;
			0.6621 0.2959 0.6603 0.7477 1      0.3537 0.7364 0.5670 0.6613 0.5140 0.5140 0.4610;
			0.5173 0.4637 0.4093 0.1803 0.3537 1      0.3582 0.1803 0.0605 0.4705 0.3582 0.6161;
			0.6758 0.1931 0.3826 0.4705 0.7364 0.3582      1 0.4705 0.6424 0.6090 0.4911 0.4962;
			0.7071 0.1202 0.5164 0.6167 0.5670 0.1803 0.4705 1      0.7149 0.4705 0.3582 0.5164;
			0.7983 0.2316 0.6079 0.6218 0.6613 0.0605 0.6424 0.7149 1      0.4371 0.4371 0.6079;
			0.5769 0.1708 0.5574 0.4705 0.5140 0.4705 0.6090 0.4705 0.4371      1 0.3745 0.4512;
			0.4705 0.4047 0.4512 0.3582 0.5140 0.3582 0.4911 0.3582 0.4371 0.3745      1 0.4512;
			0.7881 0.1161 0.5128 0.2966 0.4610 0.6161 0.4962 0.5164 0.6079 0.4512 0.4512      1]

		elseif problem == :bhwi01
			A = T[
			1    -0.50 -0.30 -0.25 -0.70;
			-0.50  1     0.90  0.30  0.70;
			-0.30  0.90  1     0.25  0.20;
			-0.25  0.30  0.25  1     0.75;
			-0.70  0.70  0.20  0.75  1]

		elseif problem == :fing97
			A = T[
			1     0.18 -0.13 -0.26  0.19 -0.25 -0.12;
			0.18  1     0.22 -0.14  0.31  0.16  0.09;
			-0.13  0.22  1     0.06 -0.08  0.04  0.04;
			-0.26 -0.14  0.06  1     0.85  0.85  0.85;
			0.19  0.31 -0.08  0.85  1     0.85  0.85;
			-0.25  0.16  0.04  0.85  0.85  1     0.85;
			-0.12  0.09  0.04  0.85  0.85  0.85  1]

		elseif problem == :high02
			A = T[
			1 1 0;
			1 1 1;
			0 1 1];

		elseif problem == :mmb13
			A = T[
			1.0       3.15946  0.200885   1.00195   -0.580897  0.195261;
			3.15946   1.0     12.6826     3.35052    8.17548  16.936;
			0.200885 12.6826   1.0       -0.0470641  0.680691  1.04237;
			1.00195   3.35052 -0.0470641  1.0       -0.788383 -0.140854;
			-0.580897  8.17548  0.680691  -0.788383   1.0       0.720099;
			0.195261 16.936    1.04237   -0.140854   0.720099  1.0]

		elseif problem == :tec03
			A = T[
			1     -0.55 -0.15 -0.10;
			-0.55  1     0.90  0.90;
			-0.15  0.90  1     0.90;
			-0.10  0.90  0.90  1]

		elseif problem == :tyda99r1
			x = T[0.1, -1, 0.4, 0.8, -0.1, -0.2, 0.7, 0.4, -0.3, 0.8, -0.1, 0.4, 0.8, -0.3, 0, 0.3, 0.2, 0.9, -0.3, -0.5, -0.4, 0.3, 0.6, 0.8, 0.1, 1, -0.2, 0.6]
			A = insert_upper(x,1)

		elseif problem == :tyda99r2
			x = T[0.1, 1, 0.4, 0.8, 0.1, 0.2, 0.7, 0.4, 0.3, 0.8, 0.1, 0.4, 0.8, 0.3, 0, 0.3, 0.2, 0.9, 0.3, 0.5, 0.4, 0.3, 0.6, 0.8, 0.1, 1, 0.2, 0.6]
			A = insert_upper(x,1)

		elseif problem == :tyda99r3
			x = T[-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5]
			A = insert_upper(x,1)
			
		elseif problem == :cor1399
			file = matopen(joinpath(@__DIR__, "data/corr_mat/cor1399.mat"))
			Av = read(file, "x")[:,1]
			A = insert_upper(Av,1)

		elseif problem == :cor3120
			file = matopen(joinpath(@__DIR__, "data/corr_mat/cor3120.mat"))
			Av = read(file, "x")[:,1]
			A = insert_upper(Av,1)

		elseif problem == :bccd16
			file = matopen(joinpath(@__DIR__, "data/corr_mat/bccd16.mat"))
			A = read(file, "A")

		elseif problem == :rocky
			file = matopen(joinpath(@__DIR__, "data/corr_mat/Rocky_Mountain_Region_CORR.mat"))
			A = read(file, "A")

		else
			throw(ArgumentError("Supply a valid matrix problem. Valid names are :fing97, :beyu11, :bhwi01, :fing97, :high02, :mmb13, :tec03, :tyda99r1, :tyda99r2, :tyda99r3, :cor1399, :cor3120, :bccd16, :rocky."))
		end
	end
	n = size(A, 1)
	if n_max > 0 && n_max < n
		n = n_max
		A = A[1:n,1:n]
	end

	Y = copy(A)
	R = similar(Y)
	S = zeros(T, n, n)
	N = n * (n + 1) ÷ 2
	x0 = zeros(T, 2N)
	extract_upper!(x0,Y,0; rangex = 1:N)

	return (x0 = x0, map! = (ys_out, ys_in) -> projys!(ys_out, ys_in, Y, R, S), obj = nothing)
end