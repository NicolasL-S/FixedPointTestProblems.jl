function NS2dLaplaceMatrix(T, n)
#   Assembles 2D Laplacian operator matrix with pure Neumann boundary conditions
#   for a structured cartesian, cell centered (finite volume) mesh with n square
#   cells in x,y of size dx. 2nd order CDS applied for the Laplace operator,
#   which is scaled with a factor $dx^2$, which thus should be multiplied on the
#   rhs-vector too.
#   Adapted from the original MATLAB code written by Franz Hastrup-Nielsen

    aW = ones(T, n, n)
    aS = ones(T, n, n)
    aE = ones(T, n, n)
    aN = ones(T, n, n)
    aP = @. -(aW + aE + aN + aS) #Eq. 5.59
    
	aP[:,begin] .+= aW[:,begin]
	aW[:,begin] .= 0 #West
	aP[begin,:] .+= aS[begin,:]
	aS[begin,:] .= 0 #South
	aP[:,end] .+= aE[:,end]
	aE[:,end] .= 0 #East
	aP[end,:] .+= aN[end,:]
	aN[end,:] .= 0 #North

	return spdiagm(-n => aW[n+1:end], -1 => aS[2:end], 0 => vec(aP), 1 => aN[1:end-1], n => aE[1:end-n])
end

function NS2dHfunctions!(H1, H2, dx, Re, u, v, uP, uw, ue, uFD, us, un, u_ve, u_vw, 
						 dudx, dudxe, dudxw, dudy, dudys, dudyn, v_us, v_un, vFD, vw, ve, vP, vs, vn, 
						 dvdx, dvdxe, dvdxw, dvdy, dvdyn, dvdys)

# Computes H1- and H2-functions for Navier-Stokes equations in 2D for simulation of translating lid 
# flow in square cavity described by the incompressible unsteady 2D Navier-Stokes equations in 
# nondimensional form.  Finite Volume Method on staggered cartesian mesh with uniform square cells. 
# Central difference scheme (CDS) applied throughout the model (convective and diffusive fluxes, 
# gradient, divergence and Laplace operator). 
#
#   H1- and H2-expressions:
#        H1 = 1/dx*(1/Re*dudx-u*u)|_w^e + 1/dy*(1/Re*dudy-u*v)|_s^n
#        H2 = 1/dx*(1/Re*dvdx-u*v)|_w^e + 1/dy*(1/Re*dvdy-v*v)|_s^n
#
# Input  :
#   n    :  Number of cells along x,y-axis
#   dx   :  Cell size in x,y
#   Re   :  Global Reynolds number number
#   u    :  Horizontal velocity component array, u(1:n+2,1:n+1)
#   v    :  Vertical velocity component array, v(1:n+1,1:n+2)
#
# Output :
#   H1   :  H1-function for horizontal momentum equation, H1(1:n,1:n+1) 
#   H2   :  H2-function for vertical momentum equation,  H2(1:n+1,1:n)
#
#   Adapted from the original MATLAB code written by Franz Hastrup-Nielsen

	@inbounds @. begin

		## H1
		# Note: contrary to the initial code, the derivatives are divided by dx only when computing H1
		dudx = u[2:end-1,2:end] - u[2:end-1,1:end-1] #on P-grid
		dudy = u[2:end,2:end-1] - u[1:end-1,2:end-1] #on FD-grid
		dudy[1, :] = 2 * (u[2,2:end-1] - u[1,2:end-1])
		dudy[end, :] = 2 * (u[end,2:end-1] - u[end-1,2:end-1])

		# Note: contrary to the initial code, uP, uFD, v_us, and v_un are divided by 2 only when computing H1
		uP = u[2:end-1,2:end] + u[2:end-1,1:end-1] #Getting u on P grid
		uFD = u[1:end-1,2:end-1] + u[2:end,2:end-1] #Getting u on FD grid
		v_us = v[1:end-1,3:end-1] + v[1:end-1,2:end-2]
		v_un = v[2:end,3:end-1] + v[2:end,2:end-2]

		H1[:,2:end-1] = (1 / (Re * dx^2)) * (dudxe - dudxw + dudyn - dudys) + 
						(0.25 / dx) * (- ue * ue + uw * uw - un * v_un + us * v_us)

		## H2 
		# Note: contrary to the initial code, the derivatives are divided by dx only when computing H2
		dvdx = v[2:end-1,2:end] - v[2:end-1,1:end-1]
		dvdx[:,1] = 2 * (v[2:end-1,2] - v[2:end-1,1])
		dvdx[:,end] = 2 * (v[2:end-1,end] - v[2:end-1,end-1])
		dvdy = v[2:end,2:end-1] - v[1:end-1,2:end-1]

		# Note: contrary to the initial code, vFD, vP, u_ve and u_vw are divided by 2 only when computing H2
		vFD = v[2:end-1,2:end] + v[2:end-1,1:end-1]
		vFD[:,1] = 2v[2:end-1,1]
		vFD[:,end] = 2v[2:end-1,end]
		vP = v[1:end-1,2:end-1] + v[2:end,2:end-1]
		u_ve = u[2:end-2,2:end] + u[3:end-1, 2:end]
		u_vw = u[2:end-2,1:end-1] + u[3:end-1,1:end-1]

		H2[2:end-1,:] = (1 / (Re * dx^2)) * (dvdxe - dvdxw + dvdyn - dvdys) + 
						(0.25 / dx) * (-u_ve * ve + u_vw * vw - vn * vn + vs * vs)
	end
	return H1, H2
end

function one_iter!(uv_out, uv_in, H1, H2, dx, Re, u, v, uP, uw, ue, uFD, us, un, u_ve, u_vw, 
				   dudx, dudxe, dudxw, dudy, dudys, dudyn, v_us, v_un, vFD, vw, ve, vP, vs, vn, 
				   dvdx, dvdxe, dvdxw, dvdy, dvdyn, dvdys, luA, s, P, dt)

	nu = length(u)
	length(uv_out) == length(uv_in) == 2nu || throw(DimensionMismatch("The input vector's dimensions do not match the problem."))
	u[:] .= uv_in[begin: begin - 1 + nu]
	v[:] .= uv_in[begin + nu: end]

	# Updating H1 and H2
	NS2dHfunctions!(H1, H2, dx, Re, u, v, uP, uw, ue, uFD, us, un, u_ve, u_vw, dudx, dudxe, dudxw, 
		dudy, dudys, dudyn, v_us, v_un, vFD, vw, ve, vP, vs, vn, dvdx, dvdxe, dvdxw, dvdy, dvdyn, dvdys)

	@inbounds begin
		# Note, in the previous code s was multiplied by dx and P was centralized. These were removed as they cancel out.
		# @. s = (H1[:,2:end] - H1[:,1:end-1] + H2[2:end,:] - H2[1:end-1,:]) # Without pressure correction
		@. s = H1[:,2:end] - H1[:,1:end-1] + H2[2:end,:] - H2[1:end-1,:] + 
			(1/dt) * (u[2:end-1,2:end] - u[2:end-1,1:end-1] .+ v[2:end,2:end-1] - v[1:end-1,2:end-1]) # With pressure correction

		P[:] .= luA \ vec(s)
		
		@. u[2:end-1,2:end-1] += dt * (H1[:,2:end-1] - P[:,2:end] + P[:,1:end-1])
		@. v[2:end-1,2:end-1] += dt * (H2[2:end-1,:] - P[2:end,:] + P[1:end-1,:])
	end
	uv_out[begin:begin + nu - 1] .= u[:]
	uv_out[begin + nu:end] .= v[:]
	return uv_out
end

"""
Simulates a translating lid ﬂow in a square cavity using unsteady 2D Navier-Stokes equations. The 
code is adapted from the repository https://github.com/franzhastrup/MATLAB-Lid-Driven-Cavity-Flow

Keyword arguments:
; n = 161, randomize = false, T = typeof(1.)
"""
function gen_lid_cavity_flow(; n = 161, randomize = false, T = typeof(1.)) # n is the number of cells along x,y-axis
	Re         = 1000 # global Reynolds number
	Ulid       = -1   # lid velocity (+-1)

	dx = 1/n
	U = 1
	dt = min(dx^2 * Re/4, 1/(U^2 * Re)) # Time step based on stab. analysis.

	# Set initial flow field such that du/dx + dv/dy = 0
	u = zeros(T, n + 2, n + 1) + randomize * (0.2 .- 0.1rand(T, n + 2, n + 1))   # initial horizontal velocity component
	v = zeros(T, n + 1, n + 2) + randomize * (0.2 .- 0.1rand(T, n + 1, n + 2))   # initial vertical velocity component
	u[n + 2, :] .= Ulid                  # set lid velocity in u-array

	# Setting up working arrays and some views to avoid copying in NS2dHfunctions!
	uP = Matrix{T}(undef, n, n); uw = view(uP,:, 1:n - 1); ue = view(uP,:, 2:n)   #East
	uFD = Matrix{T}(undef, n + 1, n - 1); us = view(uFD, 1:n, :); un = view(uFD,2:n + 1, :)
	dudx = Matrix{T}(undef, n, n); dudxe = view(dudx,:, 2:n); dudxw = view(dudx,:, 1:n - 1)
	dudy = Matrix{T}(undef, n + 1, n - 1); dudys = view(dudy, 1:n, :); dudyn = view(dudy,2:n + 1, :)
	u_ve = Matrix{T}(undef, n - 1, n)
	u_vw = Matrix{T}(undef,n - 1, n)
	v_us = Matrix{T}(undef, n, n - 1)
	v_un = Matrix{T}(undef, n, n - 1)
	vFD = Matrix{T}(undef, n - 1, n + 1); vw = view(vFD,:,1:n); ve = view(vFD,:,2:n + 1)
	vP = Matrix{T}(undef, n, n); vn = view(vP, 2:n, :); vs = view(vP,1:n - 1,:)
	dvdx = Matrix{T}(undef, n - 1, n + 1); dvdxe = view(dvdx, :, 2:n + 1); dvdxw = view(dvdx,:,1:n)
	dvdy = Matrix{T}(undef, n, n); dvdyn = view(dvdy, 2:n, :); dvdys = view(dvdy,1:n - 1,:)
	H1 = zeros(size(u) .- (2, 0))
	H2 = zeros(size(v) .- (0, 2))

	# allocate pressure array P[:] = A\s[:];
	P  = zeros(T, n, n)          
	s = H1[:, 2:end]

	luA = lu(NS2dLaplaceMatrix(T, n)) # Creating and factoring A (the Laplacian operator matrix)

	map!(uv_out, uv_in) = one_iter!(uv_out, uv_in, H1, H2, dx, Re, u, v, uP, uw, ue, uFD, us, 
			un, u_ve, u_vw, dudx, dudxe, dudxw, dudy, dudys, dudyn, v_us, v_un, vFD, vw, ve, vP, vs, 
			vn, dvdx, dvdxe, dvdxw, dvdy, dvdyn, dvdys, luA, s, P, dt)

	return (x0 = [vec(u); vec(v)], map! = map!, obj = nothing)
end