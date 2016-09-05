# based on 
# A QUASI-NEWTON METHOD FOR SOLVING SMALL NONLINEAR SYSTEMS OF ALGEBRAIC EQUATIONS
# JEFFREY MARTIN
# http://web.mit.edu/jmartin3/Public/Project.pdf
macro broydentrace(stepnorm)
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
                dt["f(x)"] = copy(fvec)
                dt["g(x)"] = copy(fjac)
            end
            update!(tr,
                    it,
                    maximum(abs(fvec)),
                    $stepnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end


const τ = 0.5
const σ1 = 1e-8
const σ2 = 1e-8
const ρ = 1.0 - 1e-8
const η = 1e-8
const restol = 1e-6
const imax = 10
function broyden_{T}(df::AbstractDifferentiableMultivariateFunction,
                   initial_x::Vector{T},
                   xtol::T,
                   ftol::T,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool)

    x = copy(initial_x)
    nn = length(x)
    xold = fill(convert(T, NaN), nn)
    fvec = Array(T, nn)
    fvecold = similar(fvec)
    fjac = alloc_jacobian(df, T, nn)
    scaleinvfjac = alloc_jacobian(df, T, nn)
    oldinvfjac = alloc_jacobian(df, T, nn)
    minvjΔf = similar(xold)
    p = Array(T, nn)
    g = Array(T, nn)
    gr = Array(T, nn)

    # Count function calls
    f_calls::Int, g_calls::Int = 0, 0

    df.fg!(x, fvec, fjac)
    f_calls += 1
    g_calls += 1

    check_isfinite(fvec)
    normfvec = norm(fvec)
    normfvecold = normfvec
    normfvecoldold = normfvec

    it = 0
    x_converged, f_converged, converged = assess_convergence(x, xold, fvec, xtol, ftol)

    computejac = true

    tr = SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    @broydentrace convert(T, NaN)


    while !converged && it < iterations
        it += 1
        if computejac
            if it > 1
                df.fg!(x, fvec, fjac)
                g_calls += 1
            end
            try
                p = fjac \ fvec
                scale!(p, -1)
            catch e
                if isa(e, Base.LinAlg.LAPACKException)
                    fjac2 = Ac_mul_B(fjac, fjac)
                    lambda = convert(T, 1e6) * sqrt(nn * eps()) * norm(fjac2, 1)
                    g = Ac_mul_B(fjac, fvec)
                    p = (fjac2 + lambda * eye(nn)) \ g
                    scale!(p, -1)
                else
                    throw(e)
                end
            end
            invfjac = inv(fjac)
            computejac = false
        else
            # the following corresponds to the Sherman-Morrison formula
            # newJ^{-1} = J^{-1} + (Δx - J^{-1}Δf)Δx'J^{-1} / (Δx'J^{-1}Δf)
            #TODO : Use QR update
            Base.axpy!(-1.0, fvec, fvecold)
            A_mul_B!(minvjΔf, invfjac, fvecold)
            d = dot(p, minvjΔf)
            Base.axpy!(1.0, p, minvjΔf)
            scale!(minvjΔf, - 1 / d)
            A_mul_Bc!(scaleinvfjac, minvjΔf , p)
            for i in 1:size(scaleinvfjac, 2)
                scaleinvfjac[i, i] += 1.0
            end
            invfjac, oldinvfjac = oldinvfjac, invfjac
            A_mul_B!(invfjac, scaleinvfjac, oldinvfjac)
            A_mul_B!(p, invfjac,  fvec)
            scale!(p, -1.0)
        end
        copy!(xold, x)
        copy!(fvecold, fvec)
        Base.axpy!(1.0, p, x)
        df.f!(x, fvec)
        f_calls += 1
        normfvec = norm(fvec)
        normp = norm(p)
        # start line search
        i = 0
        if normfvec > (ρ * normfvecold - σ2 * normp^2)
            λ = τ
            while (i == 0) || ((i < imax) && (normfvec >= ((1 + 1 / it^2) * normfvecold - σ1 * λ^2 * normp^2)))
                i += 1
                λ *= τ
                copy!(x, xold)
                Base.axpy!(λ, p, x)
                df.f!(x, fvec)
                f_calls += 1
                normfvec = norm(fvec)
            end
        end
        if i == imax
            # Line search fails. Don't accept update and compute true jacobian
            computejac = true
            copy!(x, xold)
            copy!(fvec, fvecold)
        else
            # accept update
            if (abs(normfvec - normfvecold) < restol) && (abs(normfvecold - normfvecoldold) < restol)
            	# slow convergence. Compute true jacobian.
                computejac = true
            end
            normfvecold, normfvecoldold = normfvec, normfvecold
            x_converged, f_converged, converged = assess_convergence(x, xold, fvec, xtol, ftol)
        end
        @broydentrace sqeuclidean(x, xold)
    end
    SolverResults("broyden",
                         initial_x, x, norm(fvec, Inf),
                         it, x_converged, xtol, f_converged, ftol, tr,
                         f_calls, g_calls)
end

function broyden{T}(df::AbstractDifferentiableMultivariateFunction,
                   initial_x::Vector{T},
                   xtol::Real,
                   ftol::Real,
                   iterations::Integer,
                   store_trace::Bool,
                   show_trace::Bool,
                   extended_trace::Bool)
    broyden_(df, initial_x, convert(T, xtol), convert(T, ftol), iterations, store_trace, show_trace, extended_trace)
end
