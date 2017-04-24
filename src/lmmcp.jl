@with_kw immutable LMMCPOptions
    β::Float64 = 0.55
    Big::Float64 = 1e10
    δ::Float64 = 5.0
    δmin::Float64 = 1
    verbosity::Int = 0
    ϵ1::Float64 = 1e-6
    η::Float64 = 0.95
    m::Int = 10
    kwatch::Int = 20  # maximum number of steps
    λ1::Float64 = 0.1
    λ2::Float64 = 1 - λ1
    iterations::Int = 500
    null::Float64 = 1e-10
    preprocess::Bool = true
    presteps::Int = 20
    σ::Float64 = 1e-4
    σ1::Float64 = 0.5
    σ2::Float64 = 2.0
    tmin::Float64 = 1e-12  # safeguard stepsize
    ftol::Float64 = sqrt(eps())
    watchdog::Bool = true

    @assert kwatch >= m
    @assert λ2 == 1 - λ1
    @assert 0 < λ1 <= 1
end

function lmmcp{T}(df::AbstractDifferentiableMultivariateFunction, lb, ub, x0::AbstractVector{T};
               kwargs...)
    # process/unpack options
    opts = LMMCPOptions(;kwargs...)
    @unpack ϵ1, null, Big, iterations, β, σ, tmin, m, kwatch, η, ftol = opts
    @unpack watchdog, preprocess, presteps, δ, δmin, σ1, σ2, verbosity = opts

    # initalization
    k = 0
    k_main = 0

    # compute a feasible starting point by projection
    x = max.(lb, min.(x0, ub))
    n = length(x)

    @assert length(lb) == n
    @assert length(ub) == n

    # definition of index sets I_l, I_u, and I_lu
    indexset = zeros(UInt8, n)
    for i in 1:n
        if lb[i] > -Big
            indexset[i] = ifelse(ub[i] < Big, 3, 1)
        else
            ub[i] < Big && setindex!(indexset, 2, i)
        end
    end

    # function evaluations
    fx = similar(x)
    Dfx = alloc_jacobian(df, T, n)
    df.fg!(x, fx, Dfx); f_calls = 1; g_calls = 1

    # choice of NCP-function and corresponding evaluations
    Phix      = Phi(x, fx, lb, ub, n, indexset, opts)
    normPhix  = norm(Phix)
    Psix      = 0.5*dot(Phix, Phix)
    DPhix     = DPhi(x, fx, Dfx, lb, ub, n, indexset, opts)
    DPsix     = DPhix'*Phix
    normDPsix = norm(DPsix)

    # save initial values
    x0         = copy(x)
    Phix0      = copy(Phix)
    Psix0      = copy(Psix)
    DPhix0     = copy(DPhix)
    DPsix0     = copy(DPsix)
    normDPsix0 = copy(normDPsix)

    # watchdog strategy
    aux    = zeros(m)
    aux[1] = Psix
    MaxPsi = Psix

    if watchdog
        kbest        = copy(k)
        xbest        = copy(x)
        Phibest      = copy(Phix)
        Psibest      = copy(Psix)
        DPhibest     = copy(DPhix)
        DPsibest     = copy(DPsix)
        normDPsibest = copy(normDPsix)
    end

    # initial printout
    if verbosity > 1
        @printf "%4s%21s%29s%12s\n" "k" "Psi(x)" "|| DPsi(x) ||" "stepsize"
        println("="^68)
        println("*"^21, " Output at starting point ", "*"^21)
        @printf "%4.0f %24.5e %24.5e\n" k Psix normDPsix
    end

    if preprocess
        if verbosity > 1
            println("*"^26, " Preprocessor ", "*"^28)
        end

        normpLM = 1.0

        while (k < presteps) && (Psix > ftol) && (normpLM>null)
            k += 1

            # choice of Levenberg-Marquardt parameter, note that we do not use
            # the condition estimator for large-scale problems, although this
            # may cause numerical problems in some examples
            i = false
            mu = 0
            if n < 100
                i = true
                mu = 1e-16
                #= NOTE
                I really don't like doing `full` here, but for a simple case
                where

                    DPhix'*DPhix = sparse([15.431 -50.889; -50.889 177.489])

                I let `cond(DPhix'*DPhix, 1)` run for > 5 minutes with no
                answer. The dense version runs in <= 0.000024 seconds

                same note applies to use of `cond` below
                =#
                if cond(full(DPhix'*DPhix), 1) > 1e25
                    mu = 1e-6 / (k+1)
                end
            end

            if i
                pLM = [DPhix; sqrt(mu)*speye(n)] \ [-Phix; zeros(n)]
            else
                pLM = -DPhix\Phix
            end
            normpLM = norm(pLM)

            # compute the projected Levenberg-Marquard step onto box Xk
            lbnew = max.(min.(lb .- x, 0), -δ)
            ubnew = min.(max.(ub .- x, 0), δ)
            d     = max.(lbnew, min.(pLM, ubnew))
            x .+= d

            # function evaluations etc.
            df.fg!(x, fx, Dfx); f_calls += 1; g_calls += 1
            Phixnew = Phi(x, fx, lb, ub, n, indexset, opts)
            Psixnew = 0.5*dot(Phixnew, Phixnew)
            normPhixnew = norm(Phixnew)

            # update of δ
            if normPhixnew <= η*normPhix
                δ = max(δmin, σ2*δ)
            elseif normPhixnew > 5η*normPhix
                δ = max(δmin, σ1*δ)
            end

            # update
            Phix      = Phixnew
            Psix      = Psixnew
            normPhix  = normPhixnew
            DPhix     = DPhi(x, fx, Dfx, lb, ub, n, indexset, opts)
            DPsix     = DPhix'*Phix
            normDPsix = norm(DPsix, Inf)

            # output at each iteration
            t = 1
            if verbosity > 1
                @printf("%4.0f %24.5e %24.5e %11.7g\n", k, Psix, normDPsix, t)
            end
        end
    end

    # terminate program or redefine current iterate as original initial point
    if preprocess && Psix < ftol
        if verbosity > 0
            @printf("Psix = %1.4e\nnormDPsix = %1.4e\n", Psix, normDPsix)
            println("Approximate solution found.")
        end
        return SolverResults(
            "LMMCP", x0, x, norm(fx, Inf), k, false, 0.0, true, ftol, SolverTrace(),
            f_calls, g_calls
        )
    elseif preprocess && Psix >= ftol
        x = x0
        Phix = Phix0
        Psix = Psix0
        DPhix = DPhix0
        DPsix = DPsix0
        if verbosity > 1
            println("******************** Restart with initial point ********************")
            @printf("%4.0f %24.5e %24.5e\n", k_main, Psix0, normDPsix0)
        end
    end

    if verbosity > 1
        println("************************** Main program ****************************")
    end

    while (k < iterations) && (Psix > ftol)
        # choice of Levenberg-Marquardt parameter, note that we do not use
        # the condition estimator for large-scale problems, although this
        # may cause numerical problems in some examples

        i = false
        if n < 100
            i = true
            mu = 1e-16
            if cond(full(DPhix'*DPhix), 1) > 1e25
                mu = 1e-1/(k+1)
            end
        end

        # compute a Levenberg-Marquard direction

        if i
            d = [DPhix; sqrt(mu)*speye(n)] \ [-Phix; zeros(n)]
        else
            d = -DPhix\Phix
        end

        # computation of steplength t using the nonmonotone Armijo-rule
        # starting with the 6-th iteration

        # computation of steplength t using the monotone Armijo-rule if
        # d is a 'good' descent direction or k<=5

        t = 1
        x .+= d
        df.f!(x, fx); f_calls += 1
        Phi!(Phix, x, fx, lb, ub, n, indexset, opts)
        Psix = 0.5*dot(Phix, Phix)
        that_constant   = σ*dot(DPsix, d)

        while (Psix > MaxPsi + that_constant*t) && (t > tmin)
            t *= β
            x .+= t*d
            df.f!(x, fx); f_calls += 1
            Phi!(Phix, x, fx, lb, ub, n, indexset, opts)
            Psix = 0.5*dot(Phix, Phix)
        end

        # updating things
        df.g!(x, Dfx); g_calls += 1
        DPhix = DPhi(x, fx, Dfx, lb, ub, n, indexset, opts)
        DPsix = DPhix'*Phix
        normDPsix = norm(DPsix)
        k += 1
        k_main += 1

        if k_main <= 5
            aux[mod(k_main, m) + 1] = Psix
            MaxPsi = Psix
        else
            aux[mod(k_main, m) + 1] = Psix
            MaxPsi = maximum(aux)
        end

        # updatings for the watchdog strategy
        if watchdog
            if Psix < Psibest
                kbest = k
                xbest = copy(x)
                Phibest = copy(Phix)
                Psibest = copy(Psix)
                DPhibest = copy(DPhix)
                DPsibest = copy(DPsix)
                normDPsibest = normDPsix
            elseif k - kbest > kwatch
                x = copy(xbest)
                Phix = copy(Phibest)
                Psix = copy(Psibest)
                DPhix = copy(DPhibest)
                DPsix = copy(DPsibest)
                normDPsix = normDPsibest
                MaxPsi = Psix
            end
        end

        if verbosity > 1
            # output at each iteration
            @printf("%4.0f %24.5e %24.5e %11.7g\n", k, Psix, normDPsix, t)
        end
    end

    return SolverResults(
        "LMMCP", x0, x, norm(Phix, Inf), k, false, 0.0, k < iterations, ftol,
        SolverTrace(), f_calls, g_calls
    )

end

## Helper functions for this routine
ϕ(a, b) = sqrt(a*a + b*b) - a - b

function Phi!(
        Φ::AbstractVector, x::AbstractVector, fx::AbstractVector,
        lb::AbstractVector, ub::AbstractVector, n::Int,
        indexset::AbstractVector, opts::LMMCPOptions
    )
    @unpack λ1, λ2 = opts

    for i in 1:n
        if indexset[i] == 1 # i ∈ I_l
            Φ[i] = λ1 * ϕ(x[i] - lb[i], fx[i])
            Φ[i+n] = λ2 * max(0.0, x[i] - lb[i]) * max(0.0, fx[i])
        elseif indexset[i] == 2  # i ∈ I_u
            Φ[i] = -λ1 * ϕ(ub[i] - x[i], -fx[i])
            Φ[i+n] = λ2 * max(0.0, ub[i] - x[i]) * max(0.0, -fx[i])
        elseif indexset[i] == 3  # i ∈ I_lu
            ϕu = ϕ(ub[i] - x[i], -fx[i])
            Φ[i] = λ1 * ϕ(x[i] - lb[i], ϕu)
            Φ[i+n] = λ2 *(max(0, x[i] - lb[i]) * max(0, fx[i]) +
                          max(0, ub[i] - x[i]) * max(0, -fx[i]))
        else # indexset[i] == 0   => i ∈ I_f
            Φ[i] = -λ1 * fx[i]
            Φ[i+n] = -λ2 *fx[i]
        end
    end

    return Φ
end

function Phi(
        x::AbstractVector, fx::AbstractVector, lb::AbstractVector,
        ub::AbstractVector, n::Int, indexset::AbstractVector,
        opts::LMMCPOptions
    )
    Phi!(similar(x, 2n), x, fx, lb, ub, n, indexset, opts)
end

function DPhi(
        x::AbstractVector, fx::AbstractVector, Dfx::AbstractMatrix,
        lb::AbstractVector, ub::AbstractVector, n::Int,
        indexset::AbstractVector, opts::LMMCPOptions
    )
    @unpack λ1, λ2 = opts
    null = 1e-8
    ei = zeros(n)
    H1 = spzeros(n, n)
    H2 = spzeros(n, n)
    β_l = fill(false, n)
    β_u = fill(false, n)
    z = zeros(n)
    alpha_l = fill(false, n)
    alpha_u = fill(false, n)

    for i in 1:n
        # fill the β alpha, and z vectors
        if abs(x[i] - lb[i]) <= null && abs(fx[i]) <= null
            β_l[i] = true; z[i] = 1.0
        end
        if abs(ub[i] - x[i]) <= null && abs(fx[i]) <= null
            β_u[i] = true; z[i] = 1.0
        end
        if x[i] - lb[i] >= -null && fx[i] >= -null
            alpha_l[i] = true
        end
        if ub[i] - x[i] >= -null && fx[i] <= null
            alpha_u[i] = true
        end
    end

    for i in 1:n
        Da = 0.0
        Db = 0.0
        fill!(ei, 0)
        ei[i] = 1

        if indexset[i] == 0
            Da = 0.0
            Db = -1.0
            H2[i, :] = -Dfx[i, :]
        elseif indexset[i] == 1
            # lower bound only
            denom1 = max(null, sqrt((x[i]-lb[i])^2 + fx[i]^2))
            denom2 = max(null, sqrt(z[i]^2 + dot(Dfx[i, :], z)^2))

            if !β_l[i]
                Da = (x[i]-lb[i])/denom1 - 1
                Db = fx[i]/denom1 - 1
            else
                Da = z[i]/denom2 - 1
                Db = dot(Dfx[i, :], z)/denom2 - 1
            end

            if alpha_l[i]
                H2[i, :] = (x[i]-lb[i])*Dfx[i, :] + fx[i]*ei
            # else   This is already done when we call spzeros...
            #     H2[i, :] = 0
            end
        elseif indexset[i] == 2
            # upper bound only
            denom3 = max(null, sqrt((ub[i]-x[i])^2 + fx[i]^2))
            denom4 = max(null, sqrt(z[i]^2 + dot(Dfx[i, :], z)^2))

            if !β_u[i]
                Da = (ub[i]-x[i])/denom3 - 1
                Db = -fx[i]/denom3 - 1
            else
                Da = -z[i]/denom4 - 1
                Db = -dot(Dfx[i, :], z)/denom4 - 1
            end

            if alpha_u[i]
                H2[i, :] = (x[i]-ub[i])*Dfx[i,:] + fx[i]*ei
            # else   This is already done when we call spzeros...
            #     H2[i, :] = 0
            end
        elseif indexset[i] == 3
            # both upper and lower bounds
            ai = 0.0
            bi = 0.0
            ci = 0.0
            di = 0.0
            phi = ϕ(ub[i]-x[i], -fx[i])
            denom1 = max(null, sqrt((x[i]-lb[i])^2 + phi^2))
            denom2 = max(null, sqrt(z[i]^2 + dot(Dfx[i, :], z)^2))
            denom3 = max(null, sqrt((ub[i]-x[i])^2 + fx[i]^2))
            denom4 = max(null, sqrt(z[i]^2 + (ci*z[i] + di*dot(Dfx[i, :], z))^2))

            if !β_u[i]
                ci = (x[i]-ub[i])/denom3 + 1
                di = fx[i]/denom3 + 1
            else
                ci = 1 + z[i]/denom2
                di = 1 + dot(Dfx[i, :], z)/denom2
            end
            if !β_l[i]
                ai = (x[i]-lb[i])/denom1 - 1
                bi = phi/denom1 - 1
            else
                ai = z[i]/denom4 - 1
                bi = (ci*z[i] +di*dot(Dfx[i, :], z))/denom4 - 1
            end

            Da = ai + bi*ci
            Db = bi*di

            if alpha_l[i] && alpha_u[i]
                H2[i, :] = (-lb[i]-ub[i] + 2x[i])*Dfx[i, :] + 2fx[i]*ei
            else
                if alpha_l[i]
                    H2[i, :] = (x[i]-lb[i])*Dfx[i, :] + fx[i]*ei
                elseif alpha_u[i]
                    H2[i, :] = (x[i]-ub[i])*Dfx[i, :] + fx[i]*ei
                # else  # This is already done when we call spzeros
                #     H2[i, :] = 0
                end
            end
        end
        H1[i, :] = Da*ei + Db*Dfx[i, :]
    end
    H = [λ1*H1; λ2*H2]
end
