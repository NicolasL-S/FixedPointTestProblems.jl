_isbad(x) = isnan(x) || isinf(x)
function InterCensPHEMj!(x_out, x_in, ncoefs, d1, d2, d3, bLi, bRi, Xp, b0, g0, exp_xb0, EZil, EWil, 
	dz, drL2, dw, Dil, Dl, Cil, b01, b11, deriv1, deriv2, exp_xb0Xp, Fl)

	## The model parameters are b0 and g0
    b0 .= x_in[1:ncoefs]
	#g0 .= x_in[ncoefs + 1:end]
	g0 .= max.(x_in[ncoefs + 1:end],1e-7)

	exp_xb0 .= exp.(Xp * b0)
	EZil .= 0
    EWil .= 0
    sum_d1 = sum(d1)
	dz[1:sum_d1] .= 1 .- exp.(-(bRi[d1,:] * g0) .* exp_xb0[d1])
    sum_d2 = sum(d2)
	drL2[1:sum_d2,:] .= bRi[d2,:] .- bLi[d2,:]
	dw[1:sum_d2, :] .= 1 .- exp.(-(drL2[1:sum_d2,:] * g0) .* exp_xb0[d2])
	EZil[d1,:] .= ((bRi[d1,:])' .* g0)' .* vec(exp_xb0[d1] ./ dz[1:sum_d1])
	EWil[d2,:] .= (drL2[1:sum_d2,:]' .* g0)' .* vec(exp_xb0[d2] ./ dw[1:sum_d2, :])
	Dil .= EZil .+ EWil .* d2
	Dl .= sum(Dil; dims = 1)
	Cil .=  (d1 .+ d2) .* bRi .+ d3 .* bLi

	diff1 = 1
	b01 .= b0
	count = 0
    damp = 1e-20
	while(diff1 > 1e-7 && count < 100)
		exp_xb0 .= exp.(Xp * b01)
		exp_xb0Xp .= vec(exp_xb0) .* Xp # Why vec??
		Fl .= sum(Cil .* vec(exp_xb0); dims = 1)
        Fl .= max.(1e-5,Fl)
		deriv1 .= (sum(Dil' * Xp; dims = 1) .- sum((Cil .* (exp_xb0 * (Dl ./ Fl)))' * Xp; dims = 1))[1,:]
		deriv2 .= exp_xb0Xp' * (Cil * (vec(Dl ./ Fl.^2) .* (Cil' * exp_xb0Xp))) .- 
					 (vec(Cil * ((Dl ./ Fl.^2)[1,:] .* (Cil' * exp_xb0))) .* Xp)' * exp_xb0Xp
        if maximum(_isbad.(deriv2)) == 1 && damp  < 0.01
            b01 .= b0
            damp *=1000
        elseif maximum(_isbad.(deriv2)) == 1
            b11 .= b01 .- deriv1
            diff1 = norm(b11 .- b01, Inf)
            b01 .= b11
        else
            b11 .= b01 .- (svd(deriv2 + damp * I) \ deriv1)
            diff1 = norm(b11 .- b01, Inf)
            b01 .= b11
        end
        count += 1
	end
    x_out[1:ncoefs] .= vec(b11)
    x_out[ncoefs + 1:end] .= vec(Dl[1,:] ./ (Cil' * exp.(Xp * x_out[1:ncoefs])))
	return x_out
end

function NegLogLikICRj(x_in, ncoefs, d1, d2, d3, bLi, bRi, Xp, exp_xb, expR, expL)
	exp_xb .= exp.(Xp * x_in[1:ncoefs])
	expR .= exp.(-(bRi * x_in[ncoefs+1:end]) .* exp_xb)
	expL .= exp.(-(bLi * x_in[ncoefs+1:end]) .* exp_xb)
	alllk = (1 .- expR) .* d1 .+ (expL - expR) .* d2 .+ expL .* d3
    return sum(alllk .< 0) > 0 ? Inf : -sum(log.(alllk))
end

function gen_new_problem_censPHEM!(starts, d1s, d2s, d3s, bLis, bRis, Xps)
    for i in eachindex(starts)
        start, _, d1, d2, d3, bLi, bRi, Xp = gen_problem_censPHEM_R()
        starts[i] .= start
        d1s[i] .= d1
        d2s[i] .= d2
        d3s[i] .= d3
        bLis[i] .= bLi
        bRis[i] .= bRi
        Xps[i] .= Xp
    end
end

# Test one problem
using CSV, Tables

"""
Proportional hazards regression with interval censoring

Wang et al. proposed a method to estimate a semiparametric proportional hazard model with interval 
censoring, a common complication in medical and social studies. Their EM estimation is based on a 
two-stage data augmentation with latent Poisson random variables and a monotone spline 
representation of the baseline hazard function.	

Source: L. Wang, C.S. McMahan, M.G. Hudgens, Z.P. Qureshi, A flexible, computationally efficient 
method for fitting the proportional hazards model to interval-censored data, Biometrics 72 (1) 
(2016) 222–231.

The code is ported to Julia from the R package dareemtest (Nicholas Henderson) and PCDSpline (Bo Cai 
and Lianming Wang). Since both packages are either unavailable or outdated, the necessary functions 
to generate the data have been cut and pasted below. But to avoid dependency to rcall, the functions 
necessary to generate a new random problem have been commented out in the source file. In the future, 
it would be useful to simply rewrite the functions Ispline and GenerateICRData in Julia.

Keyword arguments:
; T = typeof(1.), randomize = false
"""
function gen_Wang_PH_interval_censoring(; T = typeof(1.), randomize = false)
	# Creating the problem
	#start, ncoefs, d1, d2, d3, bLi, bRi, Xp = gen_problem_censPHEM_R()
	x0 = [zeros(T, 4);ones(T, 6)] .+ randomize * (one(T) .- 2rand(T, 10)) # ncoef will be 4
	# Importing precomputed data
	d1 = (CSV.File("data/Wang_PH_interval_censoring/d1.csv", header=false) |> Tables.matrix)[:]
	d2 = falses(length(d1))
	d3 = 1 .- d1
	bLi = T.(CSV.File("data/Wang_PH_interval_censoring/bLi.csv", header=false) |> Tables.matrix)
	bRi = T.(CSV.File("data/Wang_PH_interval_censoring/bRi.csv", header=false) |> Tables.matrix)
	Xp = T.(CSV.File("data/Wang_PH_interval_censoring/Xp.csv", header=false) |> Tables.matrix)
	# Storage
	#T = eltype(Xp)
	N, ncoefs = size(Xp)
	exp_xb = Vector{T}(undef,N)
	expR = Vector{T}(undef,N)
	expL = Vector{T}(undef,N)
	L = size(bRi,2)

	obj = x_in -> NegLogLikICRj(x_in, ncoefs, d1, d2, d3, bLi, bRi, Xp, exp_xb, expR, expL)

	b0 = Vector{T}(undef,ncoefs)
	g0  = Vector{T}(undef,L)
	exp_xb0 = similar(exp_xb)
	EZil = Matrix{T}(undef,N,L)
	EWil = Matrix{T}(undef,N,L)
	dz = Vector{T}(undef,N)
	drL2 = Matrix{T}(undef,N, L)
	dw = Matrix{T}(undef,N, L)
	Dil = Matrix{T}(undef,N,L)
	Dl = Matrix{T}(undef,1, L)
	Cil = similar(bRi)
	b01 = Vector{T}(undef,ncoefs)
	b11 = similar(b01)
	deriv1 = Vector{T}(undef,ncoefs)
	deriv2 = Matrix{T}(undef,ncoefs,ncoefs)
	exp_xb0Xp = similar(Xp)
	Fl = Matrix{T}(undef,1,L)

	# Mapping function
	m_censPHEM! = (x_out, x_in) -> InterCensPHEMj!(x_out, x_in, ncoefs, d1, d2, d3, bLi, bRi, Xp, b0, 
		g0, exp_xb0, EZil, EWil, dz, drL2, dw, Dil, Dl, Cil, b01, b11, deriv1, deriv2, exp_xb0Xp, Fl)
	return (x0 = x0, map! = m_censPHEM!, obj = obj)
end

# To generate a new random dataset, uncomment the comment block below

#=
using RCall
R"""

# Bo Cai and Lianming Wang, Oct. 2009

Ispline<-function(x,order,knots){
	# M Spline function with order k=order+1. or I spline with order
	# x is a row vector
	# k is the order of I spline
	# knots are a sequence of increasing points
	# the number of free parameters in M spline is the length of knots plus 1.

	k = order+1
	m = length(knots)
	n = m-2+k # number of parameters
	t = c(rep(1,k)*knots[1], knots[2:(m-1)], rep(1,k)*knots[m]) # newknots

	yy1=array(rep(0,(n+k-1)*length(x)),dim=c(n+k-1, length(x)))
	for (l in k:n){
		yy1[l,]=(x>=t[l] & x<t[l+1])/(t[l+1]-t[l])
	}

	yytem1=yy1
	for (ii in 1:order){
		yytem2=array(rep(0,(n+k-1-ii)*length(x)),dim=c(n+k-1-ii, length(x)))
		for (i in (k-ii):n){
			yytem2[i,]=(ii+1)*((x-t[i])*yytem1[i,]+(t[i+ii+1]-x)*yytem1[i+1,])/(t[i+ii+1]-t[i])/ii
		}
		yytem1=yytem2
	}

	index=rep(0,length(x))
	for (i in 1:length(x)){
		index[i]=sum(t<=x[i])
	}

	yy=array(rep(0,(n-1)*length(x)),dim=c(n-1,length(x)))

	if (order==1){
		for (i in 2:n){
			yy[i-1,]=(i<index-order+1)+(i==index)*(t[i+order+1]-t[i])*yytem2[i,]/(order+1)
		}
	} else {
		for (j in 1:length(x)){
			for (i in 2:n){
				if (i<(index[j]-order+1)){
					yy[i-1,j]=1
				} else if ((i<=index[j]) && (i>=(index[j]-order+1))){
					yy[i-1,j]=(t[(i+order+1):(index[j]+order+1)]-t[i:index[j]])%*%yytem2[i:index[j],j]/(order+1)
				} else{
					yy[i-1,j]=0
				}
			}
		}
	}
	return(yy)
}

GenerateICRData <- function(nsub, beta.true=NULL, event=NULL) {
    pp <- 1/2
    beta.true<- c(-1/2,-1/2,1/2,1/2)
    nBeta <- length(beta.true)

    TT <- YY<- ddelta <- rep(0,nsub)
    Z1 <- rnorm(nsub, mean=1/2, sd=1/2)
    Z2 <- rnorm(nsub, mean=1/2, sd=1/2)
    #Z3 <- rnorm(nsub, mean=1/2, sd=1/2)
    Z3 <- rbinom(nsub,1,pp)
    Z4 <- rbinom(nsub,1,pp)
    ZZ <- cbind(Z1,Z2,Z3,Z4)

    ff <- function(x, u, eterm) {
        Gammat <- log(1 + x) + sqrt(x)
        ans <- 1 - exp(-Gammat*eterm) - u
        return(ans)
    }
    TT <- rep(0, nsub)
    for(k in 1:nsub) {
        uu <- runif(1)
        ee <- exp(sum(ZZ[k,]*beta.true))
        #print(c(ff(1e-10,uu,ee),ff(10000,uu,ee)))
        tmp <- try(uniroot(ff, eterm=ee, u=uu, interval=c(1e-12, 10000000)))
        if(class(tmp) != "try-error") {
            TT[k] <- tmp$root
        } else {
            a1 <- ff(1e-12, uu, ee)
            a2 <- ff(10000000, uu, ee)
            TT[k] <- ifelse(abs(a2) < abs(a1), 10000000, 1e-12)
        }
    }
    #TT <-rweibull(nsub,2,exp(-0.5*ZZ%*%beta.true)*sqrt(2))
    YY <- rexp(nsub,rate = 1)
    ddelta <- (TT<YY)
    #YY1 <- rexp(nsub, rate=2)
    #YY2 <- YY1 + rexp(nsub, rate=2)

    data.mat <- cbind(YY,ZZ,ddelta)
    data.sorted <- data.mat[order(data.mat[,1]),]#---sort the data by Y
    YY <- data.sorted[,1]
    ZZ <- data.sorted[,2:(1+nBeta)]
    ddelta <- data.sorted[,ncol(data.mat)]
    #event <- c(event,mean(ddelta))

    d1 <- ddelta
    d2 <- rep(0,nsub)
    d3 <- 1-ddelta
    Li <- rep(0,nsub)
    Ri <- rep(Inf,nsub)
    Li[d3==1] <- YY[d3==1]
    Ri[d1==1] <- YY[d1==1]
    X <- ZZ
    ans <- list(X=X, Y=YY, TT=TT, Li=Li, Ri=Ri, d1=d1,d2=d2,d3=d3)
    return(ans)
}
"""

function gen_problem_censPHEM_R()
	R""" # Code from icr_timings.R
		nsub <- 2000
		icrdat <- GenerateICRData(nsub) # Generating the data
		equal <- TRUE
		n.int <- 3
		order <- 3
		g0 <- rep(1,(order+n.int))
		b0 <- rep(0,dim(icrdat$X)[2])
		t.s <- seq(min(icrdat$Y),max(icrdat$Y),length.out=100)
		### How to run
		icrdat$Li[icrdat$d1 == 1] <- icrdat$Ri[icrdat$d1 == 1]
		icrdat$Ri[icrdat$d3 == 1] <- icrdat$Li[icrdat$d3 == 1]
		ti <- c(icrdat$Li[icrdat$d1 == 0], icrdat$Ri[icrdat$d3 == 0])
		if (equal == TRUE) {
			ti.max <- max(ti) + 1e-05
			ti.min <- min(ti) - 1e-05
			knots <- seq(ti.min, ti.max, length.out = (n.int + 2))
		}
		if (equal == FALSE) {
			id <- seq(0, 1, length.out = (n.int + 2))
			id <- id[-c(1, (n.int + 2))]
			ti.max <- max(ti) + 1e-05
			ti.min <- min(ti) - 1e-05
			knots <- c(ti.min, quantile(ti, id), ti.max)
		}

		g0 <- rep(1,(order+n.int))
		b0 <- rep(0,dim(icrdat$X)[2])
		t.seq <-seq(min(icrdat$Y),max(icrdat$Y),length.out=100)
		bRi <- t(Ispline(x = icrdat$Ri, order = order, knots = knots))
		bLi <- t(Ispline(x = icrdat$Li, order = order, knots = knots))
		bt <- t(Ispline(x = t.seq, order = order, knots = knots))

		par0 <- c(b0, g0)
		ncoefs <- length(b0)
		pvec.init <- par0
		#map <- InterCensPHEM(pvec.init, ncoefs, icrdat$d1, icrdat$d2, icrdat$d3, bLi, bRi, icrdat$X, bt)
		x_in <- pvec.init
		d1 <- icrdat$d1
		d2 <- icrdat$d2
		d3 <- icrdat$d3
		Xp <- icrdat$X
	"""
	@rget x_in ncoefs d1 d2 d3 bLi bRi Xp
	
	d1 = (d1 .== 1)
	d2 = (d2 .== 1.0)
	return x_in, ncoefs, d1, d2, d3, bLi, bRi, Xp
end
x_in, ncoefs, d1, d2, d3, bLi, bRi, Xp = gen_problem_censPHEM_R()

# Saving
	using CSV, Tables
	cd("/mnt/c/Users/Nicolas/Dropbox/Principal/Projets/Fixed_point_test_problems") # REMOVE
	CSV.write("data/Wang_PH_interval_censoring/d1.csv", Tables.table(d1), writeheader=false)
	CSV.write("data/Wang_PH_interval_censoring/bLi.csv", Tables.table(bLi), writeheader=false)
	CSV.write("data/Wang_PH_interval_censoring/bRi.csv", Tables.table(bRi), writeheader=false)
	CSV.write("data/Wang_PH_interval_censoring/Xp.csv", Tables.table(Xp), writeheader=false)
=#


