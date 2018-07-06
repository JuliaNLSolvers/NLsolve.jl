# Dependencies 
using BenchmarkTools
include("mccall_utilities.jl")

# Benchmark the QuantEcon default solution
@btime default_quantecon()

# Benchmark the QuantEcon iteration scheme, with update_bellman! replaced with the state-space bellman_operator!
@btime stacked_quantecon()

# Benchmark the version using nlsolve() with m = 0 (simple iteration). 
@btime nlsolve_iteration()

# Benchmark the version using nlsolve() and m = 2. 
@btime nlsolve_anderson_m(2)

# Benchmark the version using nlsolve() and m = 7
@btime nlsolve_anderson_m(7)