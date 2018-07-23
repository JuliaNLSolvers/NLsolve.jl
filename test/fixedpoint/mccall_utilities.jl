# Benchmarking the NLsolve.fixedpoint() function in the use case of: https://lectures.quantecon.org/jl/mccall_model_with_separation.html

# Dependencies. 
using Distributions # So that some method overwritten error from NLsolve doesn't intermingle with the benchmarks. 

# Wages (global since used in constructor)
const n = 60                                   # n possible outcomes for wage
const default_w_vec = linspace(10, 20, n)      # wages between 10 and 20
const a, b = 600, 400                          # shape parameters
const dist = BetaBinomial(n-1, a, b)
const default_p_vec = pdf.(dist, support(dist))

# Model Struct (global since types can't be defined in local scope)
mutable struct McCallModel{TF <: AbstractFloat,
                        TAV <: AbstractVector{TF},
                        TAV2 <: AbstractVector{TF}}
    α::TF         # Job separation rate
    β::TF         # Discount rate
    γ::TF         # Job offer rate
    c::TF         # Unemployment compensation
    σ::TF         # Utility parameter
    w_vec::TAV    # Possible wage values
    p_vec::TAV2   # Probabilities over w_vec

    McCallModel(α::TF=0.2,
                β::TF=0.98,
                γ::TF=0.7,
                c::TF=6.0,
                σ::TF=2.0,
                w_vec::TAV=default_w_vec,
                p_vec::TAV2=default_p_vec) where {TF, TAV, TAV2} =
        new{TF, TAV, TAV2}(α, β, γ, c, σ, w_vec, p_vec)
end


# Implementation from McCall with separation lecture on QuantEcon 
function default_quantecon()
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
    # Define the function to be used in the iteration loop. 
    """
    A function to update the Bellman equations.  Note that V_new is modified in
    place (i.e, modified by this function).  The new value of U is returned.

    """
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
    # Define a function to carry out the iteration loop. 
    function solve_mccall_model(mcm::McCallModel;
                                tol::AbstractFloat=1e-5,
                                max_iter::Integer=2000)

        V = ones(length(mcm.w_vec))    # Initial guess of V
        V_new = similar(V)             # To store updates to V
        U = 1.0                        # Initial guess of U
        i = 0
        error = tol + 1

        while error > tol && i < max_iter
            U_new = update_bellman!(mcm, V, V_new, U)
            error_1 = maximum(abs, V_new - V)
            error_2 = abs(U_new - U)
            error = max(error_1, error_2)
            V[:] = V_new
            U = U_new
            i += 1
        end

        return V, U
    end
    # Get results. 
    return solve_mccall_model(mcm)
end 

# Implementation of the above, but with one vector that holds the state of the calculation [U V]
function stacked_quantecon()
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
    # Function for each loop of the iteration. 
    """
    An (in-place) stacked implementation of the Bellman operator for the McCall model with separation. By convention, U is at the top of the vector. 

    bellman_operator!(newVec::AbstractVector, oldVec::AbstractVector, model::McCallModel = mcm)
    """
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
    # Function to carry out the iteration. 
    """
    Modified version of solve_mccall_model(), which uses our bellman_operator! function. 

    solve_mccall_model_inplace(mcm::McCallModel; tol::AbstractFloat=1e-5, max_iter::Integer=2000)
    """    
    function solve_mccall_model(mcm; tol=1e-5, max_iter=2000)
        oldVec = ones(length(mcm.w_vec) + 1)  # Initial guess for states. Same as those in the QE implementation.     
        newVec = similar(oldVec)  # To store updates. 
        iter = 0    # Initialize the iteration counter. 
        error = tol + 1 # Initialize the error. 

        while error > tol && iter < max_iter
            bellman_operator!(newVec, oldVec, mcm)
            error = maximum(abs.(newVec - oldVec))
            iter += 1
            oldVec .= newVec 
        end 

        return newVec
    end
    # Get results. 
    return solve_mccall_model(mcm)
end

# Implementation of stacked_quantecon(), but using the NLsolve fixed point method. m = 0. 
function fixedpoint_iteration()
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
    # Function for each loop of the iteration. 
    """
    An (in-place) stacked implementation of the Bellman operator for the McCall model with separation. By convention, U is at the top of the state vector. 

    bellman_operator!(newVec::AbstractVector, oldVec::AbstractVector, model::McCallModel = mcm)
    """
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
    # Solve 
    init = ones(length(mcm.w_vec)+1)
    return fixedpoint(bellman_operator!, init; ftol = 1e-5, iterations = 2000)
end 

# Using fixedpoint() for variable m. 
function fixedpoint_anderson_m(m)
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
    # Function for each loop of the iteration. 
    """
    An (in-place) stacked implementation of the Bellman operator for the McCall model with separation. By convention, U is at the top of the state vector. 

    bellman_operator!(newVec::AbstractVector, oldVec::AbstractVector, model::McCallModel = mcm)
    """
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
    # Solve 
    init = ones(length(mcm.w_vec)+1)
    return fixedpoint(bellman_operator!, init; m = m, ftol = 1e-5, iterations = 2000)
end 

