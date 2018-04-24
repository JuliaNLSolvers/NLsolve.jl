# Notations from Walker & Ni, "Anderson acceleration for fixed-point iterations", SINUM 2011
# Attempts to accelerate the iteration xn+1 = xn + β f(x)

struct Anderson{Tm, Tb}
    m::Tm
    β::Tb
end
struct AndersonCache{Txs, Tx, Ta, Tf} <: AbstractSolverCache
    xs::Txs
    gs::Txs
    old_x::Tx
    residuals::Txs
    alphas::Ta
    fx::Tf
end
function AndersonCache(df, method::Anderson)
    m = method.m
    N = length(df.x_f)
    T = eltype(df.x_f)

    xs = zeros(T, N, m+1) #ring buffer storing the iterates, from newest to oldest
    gs = zeros(T, N, m+1) #ring buffer storing the g of the iterates, from newest to oldest
    old_x = xs[:,1]
    residuals = zeros(T, N, m) #matrix of residuals used for the least-squares problem
    alphas = zeros(T, m) #coefficients obtained by least-squares
    fx = similar(df.x_f, N) # temp variable to store f!

    AndersonCache(xs, gs, old_x, residuals, alphas, fx)
end

@views function anderson_(df::OnceDifferentiable,
                             x0::AbstractArray{T},
                             xtol::T,
                             ftol::T,
                             iterations::Integer,
                             store_trace::Bool,
                             show_trace::Bool,
                             extended_trace::Bool,
                             m::Integer,
                             β::Real,
                             cache = AndersonCache(df, Anderson(m, β))) where T


    copyto!(cache.xs[:,1], x0)
    n = 1
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    x_converged, f_converged, converged = false, false, false

    errs = zeros(iterations)

    for n = 1:iterations
        # fixed-point iteration
        value!!(df, cache.fx, cache.xs[:,1])

        cache.gs[:,1] .= cache.xs[:,1] .+ β.*cache.fx

        x_converged, f_converged, converged = assess_convergence(cache.gs[:,1], cache.old_x, cache.fx, xtol, ftol)

        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(cache.xs[:,1])
                dt["f(x)"] = copy(cache.fx)
            end
            update!(tr,
                    n,
                    maximum(abs,cache.fx),
                    n > 1 ? sqeuclidean(cache.xs[:,1],cache.old_x) : convert(T,NaN),
                    dt,
                    store_trace,
                    show_trace)
        end

        if converged
            break
        end

        #update of new_x
        m_eff = min(n-1,m)
        new_x = copy(cache.gs[:,1])
        if m_eff > 0
            cache.residuals[:, 1:m_eff] .= (cache.gs[:,2:m_eff+1] .- cache.xs[:,2:m_eff+1]) .- (cache.gs[:,1] .- cache.xs[:,1])
            cache.alphas[1:m_eff] .= cache.residuals[:,1:m_eff] \ (cache.xs[:,1] .- cache.gs[:,1])
            for i = 1:m_eff
                new_x .+= cache.alphas[i].*(cache.gs[:,i+1] .- cache.gs[:,1])
            end
        end

        cache.xs .= circshift(cache.xs,(0,1)) # no in-place circshift, unfortunately...
        cache.gs .= circshift(cache.gs,(0,1))
        if m > 1
            copyto!(cache.old_x, cache.xs[:,2])
        else
            copyto!(cache.old_x, cache.xs[:,1])
        end
        copyto!(cache.xs[:,1], new_x)
    end

    # returning gs[:,1] rather than xs[:,1] would be better here if
    # xn+1 = xn+beta*f(xn) is convergent, but the convergence
    # criterion is not guaranteed
    x = similar(x0)
    copyto!(x, cache.xs[:,1])
    return SolverResults("Anderson m=$m β=$β",
                         x0, x, maximum(abs,cache.fx),
                         n, x_converged, xtol, f_converged, ftol, tr,
                         first(df.f_calls), 0)
end

function anderson(df::OnceDifferentiable,
                     initial_x::AbstractArray{T},
                     xtol::Real,
                     ftol::Real,
                     iterations::Integer,
                     store_trace::Bool,
                     show_trace::Bool,
                     extended_trace::Bool,
                     m::Integer,
                     beta::Real,
                     cache = AndersonCache(df, Anderson(m, beta))) where T
    anderson_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace, m, beta)
end
