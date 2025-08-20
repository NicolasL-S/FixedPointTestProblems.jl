function val(x, ε, δ, β, next_value_function, budget, income)
    return -(ε * x^δ + β * SS.evaluate(next_value_function, budget - x + income))
end

function value_given_shock(budget, ε, next_value_function, dnext_value_function, δ, β, income, prev_x0)
#=
    opt = SM.speedmapping(prev_x0[] > budget ? budget/2 : prev_x0[]; g = x -> 
        -(ε * δ * x^(δ - 1) - β * SS.evaluate(dnext_value_function, budget - x + income)), 
        lower = 0., upper = budget, reltol_resid_grow = 1.5, maps_limit = 20)
=#
#    if opt.status != :first_order
        opt = Opt.optimize(x -> val(x, ε, δ, β, next_value_function, budget, income), 0.0, budget)
#    end

    prev_x0[] = opt.minimizer
    return -val(opt.minimizer, ε, δ, β, next_value_function, budget, income)
end

function expected_utility(budget, next_value_function, dnext_value_function, δ, β, income, shock_process)

    if budget > 0.00001
        prev_x0 = Ref(budget/2) # Will hold the previous optimum to hopefully serve as favorable starting point
        integ = hcubature(ε -> value_given_shock(budget, ε[1], next_value_function, 
            dnext_value_function, δ, β, income, prev_x0) * pdf(shock_process, ε[1]), 
            [quantile(shock_process,0.0001)], [quantile(shock_process, 0.9999)], maxevals = 1000)
        return integ[1]
    else
        return β * SS.evaluate(next_value_function, income)
    end
end

function one_iterate_budget_values!(new_budget_values, budget_values, δ, β, income, shock_process, budget_state_space)

    next_value_function = SS.Schumaker(budget_state_space, budget_values)
	dnext_value_function = SS.find_derivative_spline(next_value_function)

    new_budget_values .= expected_utility.(budget_state_space, next_value_function, 
        dnext_value_function, δ, β, income, shock_process)

    return new_budget_values
end

"""
A classic problem in economics is the optimal consumption/savings decision of a consumer who faces 
unpredictable fluctuation to consumption utility. If we assume that this 
consumer is infinitely lived, then today's optimal decisions should be the same as tomorrow's. 
A way to solve the problem is to postulate some value and some decision function in the far future, 
then find what would be the optimal decisions for one period before, then one period before, and so
on and so forth. This constitues a contraction mapping which can be accelerated.

This problem has explained in detail in the applications of FixedPointAcceleration 
(https://s-baumann.github.io/FixedPointAcceleration.jl/dev/4_Applications/). Below is a simplified
version of the code with a few changes, notably swapping Optim's minimization for speedmapping
which allows to reuse the last x in the next minimization for a significant speed-up.

Keyword arguments:
; T = typeof(1.), randomize = false
"""
function gen_consumption_smoothing(; T = typeof(1.), randomize = false)
    δ = 0.2      # Decreasing returns to consumption
    β = 0.95     # Future distounting
    income = 1.0 # Periodic income
    shock_process = LogNormal(0.0, 1.0)
    budget_state_space = [collect(0:0.015:income); collect(1.05:0.05:3income)]

    map! = (new_budget_values, budget_values) -> one_iterate_budget_values!(new_budget_values, 
        budget_values, δ, β, income, shock_process, budget_state_space)
    x0 = randomize ? rand(T, length(budget_state_space)) .* budget_state_space : sqrt.(budget_state_space)
    return (x0 = x0, map! = map!, obj = nothing)
end