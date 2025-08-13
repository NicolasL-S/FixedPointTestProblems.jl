# FixedPointTestProblems

[![Build Status](https://github.com/NicolasL-S/FixedPointTestProblems.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/NicolasL-S/FixedPointTestProblems.jl/actions/workflows/CI.yml?query=branch%3Amain)

FixedPointTestProblems is a library of fixed-point iterations problems to test the performance of acceleration algorithms. The problems from statistics, physics, economics, genetics are borrowed and adapted from research papers and different tutorials. No comparable repository of this type exists in Julia (or any programming language to my knowledge) with more than a few problems. 

The problems
1. Converge to a fixed point (which may not be unique) from reasonable starting values. (i.e. Let $F:\mathbb{R}^n\rightarrow\mathbb{R}^n$, the series $x, F(x), F(F(x)),...$ should converge to $x^*$ where $F(x^*) = x^*$);
2. Are smooth;
3. Have mechanisms to avoid or correct erroneous input values (e.g. negative probabilities or non positive-definite variance-covariance matrices);
4. Present interesting challenges to solvers (they will not converge in a few iterations).

To install:
```Julia
] add FixedPointTestProblems
```
## API
Functions stored in the dictionary ``problems`` generate problems as ``@NamedTuple``.  To see the full list of problems:
```Julia
using FixedPointTestProblems
keys(problems)
```
All problems provide a starting point and an in-place mapping function. To generate a problem:
```Julia
x0, map! = problems["Higham, correlation matrix mmb13"]()
```
To solve without acceleration:
```Julia
x_0 = copy(x0)
x_1 = similar(x0)
for i in 1:1000
    map!(x_1, x_0)
    norm_resid = norm(x_1 .- x_0)
    println(i,  "  ", norm_resid)
    norm_resid < 1e-8 && break
    x_0 .= x_1
end
```
For some problems like [EM algorithm applications](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm), an objective function is also available:
```Julia
	x0, map!, obj = problems["Mixture of 3 normals"]()
```
The problems can be generated with varying precisions, and sometimes varying specifications:
```Julia
	x0, map! = problems["Bratu"](;nx = 50, ny = 50, T = Float16)
```
To learn more about a problem and its available keyword arguments, it must first be extracted from the dictionary.
```Julia
p = problems["Bratu"]
? p
```
## Reproducibility and randomized problems
Many problems involve simulated random data. To make sure that they remain identical in all versions of Julia (and may potentially be reproduced in other languages), ``FixedPointTestProblems`` contains its own simple pseudo-random number generator. For data that must instead be loaded, it is housed in this repository.

For most problems, it is also possible to randomize the starting points and/or the problem data using the `randomize ` keyword argument. 
```Julia
	x0, map! = problems["Bratu"](;randomize = true)
```