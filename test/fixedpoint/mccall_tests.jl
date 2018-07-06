# Dependencies 
using BenchmarkTools
include("mccall_utilities.jl")

# Benchmark the QuantEcon default solution
@btime default_quantecon()

# Benchmark the QuantEcon iteration scheme, with update_bellman! replaced with the state-space bellman_operator!
@btime stacked_quantecon()

# Benchmark the version using both nlsolve() and the stacked representation. 
@btime nlsolve_quantecon()