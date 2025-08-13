const problems = Dict{AbstractString, Function}()

include("problems/als_canonical_tensor.jl")
problems["ALS for CANDECOMP"] = gen_als_canonical_tensor

include("problems/Bratu.jl")
problems["Bratu"] = gen_bratu

include("problems/consumption_smoothing.jl")
problems["Consumption smoothing"] = gen_consumption_smoothing

include("problems/exchange_economy.jl")
problems["Exchange economy"] = gen_exchange_economy

include("problems/Hasselblad_Poisson_mixtures.jl")
problems["Hasselblad, Poisson mixtures"] = gen_hasselblad_poisson_mixtures

include("problems/Higham_corr_matrix.jl")
problems["Higham, correlation matrix mmb13"] = gen_higham_corr_matrix

include("problems/Lange_ancestry.jl")
problems["Lange, ancestry"] = gen_lange_ancestry

include("problems/lid_cavity_flow.jl")
problems["Lid-driven cavity ﬂow"] = gen_lid_cavity_flow

include("problems/linear.jl")
problems["Linear"] = gen_linear

include("problems/em_mixture_of_3_normals.jl") # Removed because: can't generate multivariate normals, no 
problems["Mixture of 3 normals"] = gen_em_mixture_of_3_normals

include("problems/potential_electric_field.jl")
problems["Electric field, Gauss-Seidel"] = (;n = 100, randomize = false, T = typeof(1.)) -> gen_potential_electric_field(; algo! = gauss_seidel!, randomize, n, T)
problems["Electric field, Jacobi"] = (;n = 100, randomize = false, T = typeof(1.)) -> gen_potential_electric_field(; algo! = jacobi!, randomize, n, T)
problems["Electric field, SOR"] = (;n = 100, randomize = false, T = typeof(1.)) -> gen_potential_electric_field(; algo! = SOR!, randomize, n, T)

include("problems/power_iteration.jl")
problems["Power iter. for dom. eigenvalue"] = gen_power_iteration

include("problems/Wang_PH_interval_censoring.jl")
problems["Wang, PH interval censoring"] = gen_Wang_PH_interval_censoring