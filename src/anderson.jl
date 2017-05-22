
function anderson_{T}(df::AbstractDifferentiableMultivariateFunction,
                      x0::Vector{T},
                      xtol::T,
                      ftol::T,
                      iterations::Integer,
                      store_trace::Bool,
                      show_trace::Bool,
                      extended_trace::Bool,
                      m::Integer,
                      β :: Real)

    f_calls = 0
    N = length(x0)
    xs = zeros(T,N,m+1) #ring buffer storing the iterates, from newest to oldest
    gs = zeros(T,N,m+1) #ring buffer storing the g of the iterates, from newest to oldest
    fx = similar(x0) # temp variable to store f!
    xs[:,1] = x0
    errs = zeros(iterations)
    n = 1
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    old_x = xs[:,1]

    for n = 1:iterations
        # fixed-point iteration
        df.f!(xs[:,1], fx)
        f_calls += 1
        gs[:,1] .= xs[:,1] .+ β.*fx
        x_converged, f_converged, converged = assess_convergence(gs[:,1], old_x, fx, xtol, ftol)

        # FIXME: How should this flag be set?
        mayterminate = false
        
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
        new_x = gs[:,1]
        if m_eff > 0
            mat = (gs[:,2:m_eff+1] .- xs[:,2:m_eff+1]) .- (gs[:,1] - xs[:,1])
            alphas = -mat \ (gs[:,1] .- xs[:,1])
            for i = 1:m_eff
                new_x .+= alphas[i].*(gs[:,i+1] .- gs[:,1])
            end
        end

        xs = circshift(xs,(0,1)) # no in-place circshift, unfortunately...
        gs = circshift(gs,(0,1))
        old_x = m > 1 ? xs[:,2] : copy(xs[:,1])
        xs[:,1] = new_x
    end

    return SolverResults("Anderson m=$m β=$β",
                         # returning gs[:,1] would be slightly better here, but the fnorm is not guaranteed
                         x0, xs[:,1], maximum(abs,fx),
                         n, x_converged, xtol, f_converged, ftol, tr,
                         f_calls, 0)
end

function anderson{T}(df::AbstractDifferentiableMultivariateFunction,
                     initial_x::Vector{T},
                     xtol::Real,
                     ftol::Real,
                     iterations::Integer,
                     store_trace::Bool,
                     show_trace::Bool,
                     extended_trace::Bool,
                     hist_size::Integer,
                     beta::Real)
    anderson_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace, hist_size, beta)
end
