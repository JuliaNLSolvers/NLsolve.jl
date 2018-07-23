# Dependencies 
using BenchmarkTools
include("mccall_utilities.jl")

# Benchmark the QuantEcon default solution
@btime default_quantecon()

# Benchmark the QuantEcon iteration scheme, with update_bellman! replaced with the stacked bellman_operator!
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

# Differences between Anderson results. 
@test default_quantecon_results == (stacked_quantecon_results[2:end], stacked_quantecon_results[1])
@show norm(default_fixedpoint_results.zero - fixedpoint_results_1.zero, Inf) 
@show norm(fixedpoint_results_1.zero - fixedpoint_results_2.zero, Inf)
@show norm(fixedpoint_results_2.zero - fixedpoint_results_7.zero, Inf)

# Differences between Anderson results and QuantEcon. 
@show norm(default_fixedpoint_results.zero - cat(1, default_quantecon_results[2], default_quantecon_results[1]), Inf)
@show norm(fixedpoint_results_1.zero - cat(1, default_quantecon_results[2], default_quantecon_results[1]), Inf)
@show norm(fixedpoint_results_2.zero - cat(1, default_quantecon_results[2], default_quantecon_results[1]), Inf)
@show norm(fixedpoint_results_7.zero - cat(1, default_quantecon_results[2], default_quantecon_results[1]), Inf)

# Test fixed-point-ness of each solution 
    # Utility function. 
    function u(c::Real, σ::Real)
        if c > 0
            return (c^(1 - σ) - 1) / (1 - σ)
        else
            return -10e6
        end
    end
    # Instantiate model object. 
    mcm = McCallModel()
    # QuantEcon Test 
        # Bellman function from QuantEcon 
        function update_bellman!(mcm::McCallModel,
                                V::AbstractVector,
                                V_new::AbstractVector,
                                U::Real)
            # Simplify notation
            α, β, σ, c, γ = mcm.α, mcm.β, mcm.σ, mcm.c, mcm.γ

            for (w_idx, w) in enumerate(mcm.w_vec)
                # w_idx indexes the vector of possible wages
                V_new[w_idx] = u(w, σ) + β * ((1 - α) * V[w_idx] + α * U)
            end

            U_new = u(c, σ) + β * (1 - γ) * U +
                            β * γ * dot(max.(U, V), mcm.p_vec)

            return U_new
        end
        # Difference 
        old = cat(1, default_quantecon_results[2], default_quantecon_results[1])
        V_new = similar(default_quantecon_results[1])
        U_new = update_bellman!(mcm, default_quantecon_results[1], V_new, default_quantecon_results[2]);
        @show norm(old - cat(1, U_new, V_new), Inf)
    # Other test 
        # New Bellman Operator Function 
        function bellman_operator!(newVec, oldVec, model = mcm) 
            # Unpack parameters. 
            α, β, σ, c, γ = mcm.α, mcm.β, mcm.σ, mcm.c, mcm.γ
            # Add new V(w) values to newVec.
            for (w_idx, w) in enumerate(mcm.w_vec)
                # w_idx indexes the vector of possible wages
                newVec[w_idx + 1] = u(w, σ) + β * ((1 - α) * oldVec[1 + w_idx] + α * oldVec[1])
            end
            # Add new U value to newVec. 
            newVec[1] = u(c, σ) + β * (1 - γ) * oldVec[1] + β * γ * dot(max.(oldVec[1], oldVec[2:end]), mcm.p_vec)
        end
        newVec = similar(default_fixedpoint_results.zero)
        bellman_operator!(newVec, default_fixedpoint_results.zero)
        @show norm(default_fixedpoint_results.zero - newVec, Inf)
        bellman_operator!(newVec, fixedpoint_results_1.zero)
        @show norm(fixedpoint_results_1.zero - newVec, Inf)
        bellman_operator!(newVec, fixedpoint_results_2.zero)
        @show norm(fixedpoint_results_2.zero - newVec, Inf)
        bellman_operator!(newVec, fixedpoint_results_7.zero)
        @show norm(fixedpoint_results_7.zero - newVec, Inf)


