module FixedPointTestProblems

	# ---- Imports ----
	using LinearAlgebra
	using MAT: matopen
	using StatsBase
	using CSV, Tables
	using SparseArrays, Kronecker
	using Distributions
	import SchumakerSpline as SS
	import Optim as Opt
	using HCubature
	using TensorToolbox: full, ktensor
	using TensorDecompositions: khatrirao!, _row_unfold
	# import SpeedMapping as SM

	# ---- Includes ----
	include("stable_rand.jl")
	include("problem_dictionary.jl")

	# ---- Exports ----
	export testproblems
end
