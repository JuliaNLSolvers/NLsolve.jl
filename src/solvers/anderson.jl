# Notations from Walker & Ni, "Anderson acceleration for fixed-point iterations", SINUM 2011
# Attempts to accelerate the iteration xn+1 = xn + β f(x)

@views function anderson_{T}(df::OnceDifferentiable,
                             x0::AbstractArray{T},
                             xtol::T,
                             ftol::T,
                             iterations::Integer,
                             store_trace::Bool,
                             show_trace::Bool,
                             extended_trace::Bool,
                             m::Integer,
                             β::Real)

    N = length(x0)
    xs = zeros(T,N,m+1) #ring buffer storing the iterates, from newest to oldest
    gs = zeros(T,N,m+1) #ring buffer storing the g of the iterates, from newest to oldest
    residuals = zeros(T, N, m) #matrix of residuals used for the least-squares problem
    alphas = zeros(T, m) #coefficients obtained by least-squares
    fx = similar(x0, N) # temp variable to store f!
    xs[:,1] = vec(x0)
    errs = zeros(iterations)
    n = 1
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    old_x = xs[:,1]
    x_converged, f_converged, converged = false, false, false

    for n = 1:iterations
        # fixed-point iteration
        value!(df, fx, xs[:,1])

        gs[:,1] .= xs[:,1] .+ β.*fx

        x_converged, f_converged, converged = assess_convergence(gs[:,1], old_x, fx, xtol, ftol)

        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(xs[:,1])
                dt["f(x)"] = copy(fx)
            end
            update!(tr,
                    n,
                    maximum(abs,fx),
                    n > 1 ? sqeuclidean(xs[:,1],old_x) : convert(T,NaN),
                    dt,
                    store_trace,
                    show_trace)
        end

        if converged
            break
        end

        #update of new_x
        m_eff = min(n-1,m)
        new_x = copy(gs[:,1])
        if m_eff > 0
            residuals[:, 1:m_eff] .= (gs[:,2:m_eff+1] .- xs[:,2:m_eff+1]) .- (gs[:,1] .- xs[:,1])
            alphas[1:m_eff] .= residuals[:,1:m_eff] \ (xs[:,1] .- gs[:,1])
            for i = 1:m_eff
                new_x .+= alphas[i].*(gs[:,i+1] .- gs[:,1])
            end
        end

        xs = circshift(xs,(0,1)) # no in-place circshift, unfortunately...
        gs = circshift(gs,(0,1))
        old_x = m > 1 ? xs[:,2] : copy(xs[:,1])
        xs[:,1] = new_x
    end

    # returning gs[:,1] rather than xs[:,1] would be better here if
    # xn+1 = xn+beta*f(xn) is convergent, but the convergence
    # criterion is not guaranteed
    x = similar(x0)
    copy!(x, xs[:,1])
    return SolverResults("Anderson m=$m β=$β",
                         x0, x, maximum(abs,fx),
                         n, x_converged, xtol, f_converged, ftol, tr,
                         first(df.f_calls), 0)
end

function anderson{T}(df::OnceDifferentiable,
                     initial_x::AbstractArray{T},
                     xtol::Real,
                     ftol::Real,
                     iterations::Integer,
                     store_trace::Bool,
                     show_trace::Bool,
                     extended_trace::Bool,
                     m::Integer,
                     beta::Real)
    anderson_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace, m, beta)
end
