function map_prices!(NewPrices, cache, prices, Endowments, TotalEndowmentsPerGood, Tastes, TotalTastesPerHousehold)
	cache .= prices'Endowments
	cache ./= TotalTastesPerHousehold
	NewPrices .= Tastes * cache'
	Numeraire = NewPrices[1] / TotalEndowmentsPerGood[1]
	for i in eachindex(NewPrices)
		NewPrices[i] = NewPrices[i] / (TotalEndowmentsPerGood[i] * Numeraire)
	end
	return NewPrices
end

"""
Finding equilibrium prices in a pure exchange economy
The code is adapted from the applications of FixedPointAcceleration
See https://s-baumann.github.io/FixedPointAcceleration.jl/dev/4_Applications/ for an in-depth 
explanation.

Keyword arguments:
; randomize = false, n = 5, g = 10, T = typeof(1.)
where n is the number of consumers and g is the numbber of goods
"""
function gen_exchange_economy(; randomize = false, n = 5, g = 10, T = typeof(1.))

	if randomize
		Endowments = rand(LogNormal(), g, n)
		Tastes     = rand(g, n)
	else
		init_stable_rand()
		Endowments = stable_rand(g, n; T, D = LogNormal())
		Tastes     = stable_rand(g, n; T)
	end

	TotalEndowmentsPerGood = sum(Endowments; dims = 2)
	TotalTastesPerHousehold = sum(Tastes; dims = 1)
	cache = similar(TotalTastesPerHousehold)

	return (
		x0 = randomize ? rand(T, g) : ones(T, g), 
		map! = (NewPrices, prices) -> map_prices!(NewPrices, cache, prices, Endowments, 
			TotalEndowmentsPerGood, Tastes, TotalTastesPerHousehold), obj = nothing)
end