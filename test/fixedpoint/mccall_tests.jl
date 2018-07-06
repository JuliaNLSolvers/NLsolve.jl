# Dependencies 
using BenchmarkTools
include("mccall_utilities.jl")

# Instantiate the model. 
mcm = McCallModel()

# Benchmark the QuantEcon default solution
@btime V, U = solve_mccall_model(mcm)

# Benchmark the QuantEcon iteration scheme, with update_bellman! replaced with the state-space bellman_operator!
@btime results = solve_mccall_model_inplace(mcm)
