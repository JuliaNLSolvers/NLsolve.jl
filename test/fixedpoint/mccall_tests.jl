# Dependencies 
using BenchmarkTools, Base.Test
include("mccall_utilities.jl")

# Benchmark the QuantEcon default solution
@btime default_quantecon()

# Benchmark the QuantEcon iteration scheme, with update_bellman! replaced with the state-space bellman_operator!
@btime stacked_quantecon()

# Benchmark the version using fixedpoint() with m = 0 (simple iteration). 
@btime fixedpoint_iteration()

# Benchmark the version using fixedpoint() and m = 1. 
@btime fixedpoint_anderson_m(1)

# Benchmark the version using fixedpoint() and m = 2. 
@btime fixedpoint_anderson_m(2)

# Benchmark the version using fixedpoint() and m = 7
@btime fixedpoint_anderson_m(7)

# Get solutions.  
default_quantecon_results = default_quantecon()
stacked_quantecon_results = stacked_quantecon()
default_fixedpoint_results = fixedpoint_iteration()
fixedpoint_results_1 = fixedpoint_anderson_m(1)
fixedpoint_results_2 = fixedpoint_anderson_m(2)
fixedpoint_results_7 = fixedpoint_anderson_m(7)

# Test for accuracy. 
@test default_quantecon_results == (stacked_quantecon_results[2:end], stacked_quantecon_results[1])
@show norm(default_fixedpoint_results.zero - fixedpoint_results_1.zero, Inf) 
@show norm(fixedpoint_results_1.zero - fixedpoint_results_2.zero, Inf)
@show norm(fixedpoint_results_2.zero - fixedpoint_results_7.zero, Inf)