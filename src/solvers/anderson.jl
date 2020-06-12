# Notations from Walker & Ni, "Anderson acceleration for fixed-point iterations", SINUM 2011
# Attempts to accelerate the iteration xₙ₊₁ = xₙ + beta*f(xₙ)

struct AndersonCache{Tx,To,Tdg,Tg,TQ,TR} <: AbstractSolverCache
    x::Tx
    g::Tx
    fxold::To
    gold::To
    Δgs::Tdg
    γs::Tg
    Q::TQ
    R::TR
end

function AndersonCache(df, m)
    x = similar(df.x_f)
    g = similar(x)

    if m > 0
        fxold = similar(x)
        gold = similar(x)

        # maximum size of history
        mmax = min(length(x), m)

        # buffer storing the differences between g of the iterates, from oldest to newest
        Δgs = [similar(x) for _ in 1:mmax]

        T = eltype(x)
        γs = Vector{T}(undef, mmax) # coefficients obtained from the least-squares problem

        # matrices for QR decomposition
        Q = Matrix{T}(undef, length(x), mmax)
        R = Matrix{T}(undef, mmax, mmax)
    else
        fxold = nothing
        gold = nothing
        Δgs = nothing
        γs = nothing
        Q = nothing
        R = nothing
    end

    AndersonCache(x, g, fxold, gold, Δgs, γs, Q, R)
end

@views function anderson_(df::Union{NonDifferentiable, OnceDifferentiable},
                             initial_x::AbstractArray,
                             xtol::Real,
                             ftol::Real,
                             iterations::Integer,
                             store_trace::Bool,
                             show_trace::Bool,
                             extended_trace::Bool,
                             beta::Real,
                             aa_start::Integer,
                             droptol::Real,
                             cache::AndersonCache)
    copyto!(cache.x, initial_x)
    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace

    aa_iteration = cache.γs !== nothing
    x_converged, f_converged, converged = false, false, false
    m = aa_iteration ? length(cache.γs) : 0
    aa_iteration && (m_eff = 0)

    iter = 0
    while iter < iterations
        iter += 1

        # evaluate function
        value!!(df, cache.x)
        fx = value(df)

        # check that all values are finite
        check_isfinite(fx)

        # compute next iterate of fixed-point iteration
        @. cache.g = cache.x + beta * fx

        # save trace
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(cache.x)
                dt["f(x)"] = copy(fx)
            end
            update!(tr,
                    iter,
                    maximum(abs, fx),
                    iter > 1 ? sqeuclidean(cache.g, cache.x) : convert(real(eltype(initial_x)), NaN),
                    dt,
                    store_trace,
                    show_trace)
        end

        # check convergence
        x_converged, f_converged = assess_convergence(cache.g, cache.x, fx, xtol, ftol)
        if any(isnan, cache.x) || any(isnan, value(df))
          break
        end
        converged = x_converged || f_converged
        converged && break

        # update current iterate
        copyto!(cache.x, cache.g)

        # perform Anderson acceleration
        if aa_iteration
            if iter == aa_start
                # initialize cache of residuals and g
                copyto!(cache.fxold, fx)
                copyto!(cache.gold, cache.g)
            elseif iter > aa_start
                # increase size of history
                m_eff += 1

                # remove oldest history if maximum size is exceeded
                if m_eff > m
                    # circularly shift differences of g
                    ptr = cache.Δgs[1]
                    for i in 1:(m-1)
                        cache.Δgs[i] = cache.Δgs[i + 1]
                    end
                    cache.Δgs[m] = ptr

                    # delete left-most column of QR decomposition
                    qrdelete!(cache.Q, cache.R, m)

                    # update size of history
                    m_eff = m
                end

                # update history of differences of g
                @. cache.Δgs[m_eff] = cache.g - cache.gold

                # replace/add difference of residuals as right-most column to QR decomposition
                @. cache.fxold = fx - cache.fxold
                qradd!(cache.Q, cache.R, vec(cache.fxold), m_eff)

                # update cached values
                copyto!(cache.fxold, fx)
                copyto!(cache.gold, cache.g)

                # define current Q and R matrices
                Q, R = view(cache.Q, :, 1:m_eff), UpperTriangular(view(cache.R, 1:m_eff, 1:m_eff))

                # check condition (TODO: incremental estimation)
                if droptol > 0
                    while cond(R) > droptol && m_eff > 1
                        qrdelete!(cache.Q, cache.R, m_eff)
                        m_eff -= 1
                        Q, R = view(cache.Q, :, 1:m_eff), UpperTriangular(view(cache.R, 1:m_eff, 1:m_eff))
                    end
                end

                # solve least squares problem
                γs = view(cache.γs, 1:m_eff)
                ldiv!(R, mul!(γs, Q', vec(fx)))

                # update next iterate
                for i in 1:m_eff
                    @. cache.x -= cache.γs[i] * cache.Δgs[i]
                end
            end
        end
    end

    return SolverResults("Anderson m=$m beta=$beta aa_start=$aa_start droptol=$droptol",
                         initial_x, copy(cache.x), norm(value(df), Inf),
                         iter, x_converged, xtol, f_converged, ftol, tr,
                         first(df.f_calls), 0)
end

function anderson(df::Union{NonDifferentiable, OnceDifferentiable},
                  initial_x::AbstractArray,
                  xtol::Real,
                  ftol::Real,
                  iterations::Integer,
                  store_trace::Bool,
                  show_trace::Bool,
                  extended_trace::Bool,
                  m::Integer,
                  beta::Real,
                  aa_start::Integer,
                  droptol::Real)
    anderson(df, initial_x, xtol, ftol, iterations, store_trace, show_trace, extended_trace, beta, aa_start, droptol, AndersonCache(df, m))
end

function anderson(df::Union{NonDifferentiable, OnceDifferentiable},
                  initial_x::AbstractArray,
                  xtol::Real,
                  ftol::Real,
                  iterations::Integer,
                  store_trace::Bool,
                  show_trace::Bool,
                  extended_trace::Bool,
                  beta::Real,
                  aa_start::Integer,
                  droptol::Real,
                  cache::AndersonCache)
    anderson_(df, initial_x, convert(real(eltype(initial_x)), xtol), convert(real(eltype(initial_x)), ftol), iterations, store_trace, show_trace, extended_trace, beta, aa_start, droptol, cache)
end
