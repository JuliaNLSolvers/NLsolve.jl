#=

The routines in this file were created by referencing the implementation in
the scipy.optimize._spectral module.

The Scipy license is included below:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2017 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
=#

abstract NonMonLineSearch

type ChengLineSearch{T} <: NonMonLineSearch
    xp::Vector{T}
    Fp::Vector{T}
    Q::Float64
    C::Float64
    nx::Int
    gamma::Float64
    tau_min::Float64
    tau_max::Float64
    nu::Float64
end

_name(::Type{ChengLineSearch}) = "Cheng non-monotonic linesearch"

function ChengLineSearch{T}(xp::Vector{T}, Fp::Vector{T}=similar(xp),
                            n_prev::Int=10, Q::Float64=1.0, C::Float64=1.0,
                            gamma::Float64=1e-4, tau_min::Float64=0.1,
                            tau_max::Float64=0.5, nu::Float64=0.85)
    nx = length(xp)
    ChengLineSearch(xp, Fp, Q, C, nx, gamma, tau_min, tau_max, nu)
end

function (ls::ChengLineSearch)(func!, x_k, d, f_k, eta)
    alpha_p = 1.0
    alpha_m = 1.0
    alpha = 1.0

    fp = 1.0
    nfev = 0
    while true
        for i in 1:ls.nx
            ls.xp[i] = x_k[i] + alpha_p * d[i]
        end
        func!(ls.xp, ls.Fp); nfev += 1
        fp = _fmerit(ls.Fp)

        fp <= ls.C + eta - ls.gamma * alpha_p^2 * f_k && break

        alpha_tp = alpha_p^2 * f_k / (fp + (2*alpha_p - 1)*f_k)

        for i in 1:ls.nx
            ls.xp[i] = x_k[i] - alpha_m * d[i]
        end
        func!(ls.xp, ls.Fp); nfev += 1
        fp = _fmerit(ls.Fp)

        fp <= ls.C + eta - ls.gamma * alpha_m^2 * f_k && break

        alpha_tm = alpha_m^2 * f_k / (fp + (2*alpha_m - 1)*f_k)

        alpha_p = clamp(alpha_tp, ls.tau_min * alpha_p, ls.tau_max * alpha_p)
        alpha_m = clamp(alpha_tm, ls.tau_min * alpha_m, ls.tau_max * alpha_m)
    end
    # Update C and Q
    Q_next = ls.nu * ls.Q + 1
    ls.C = (ls.nu * ls.Q * (ls.C + eta) + fp) / Q_next
    ls.Q = Q_next

    return fp, nfev

end
Base.push!(ls::ChengLineSearch, f) = nothing  # no-op

type CruzLineSearch{T} <: NonMonLineSearch
    xp::Vector{T}
    Fp::Vector{T}
    n_prev::Int
    prev_f::CircularDeque{T}
    nx::Int
    gamma::Float64
    tau_min::Float64
    tau_max::Float64
    nu::Float64
end

function CruzLineSearch{T}(xp::Vector{T}, Fp::Vector{T}=similar(xp),
                           n_prev::Int=10, Q::Float64=1.0, C::Float64=1.0,
                           gamma::Float64=1e-4, tau_min::Float64=0.1,
                           tau_max::Float64=0.5, nu::Float64=0.85)
    nx = length(xp)
    prev_f = CircularDeque{T}(n_prev)
    CruzLineSearch(xp, Fp, n_prev, prev_f, nx, gamma, tau_min, tau_max, nu)
end

_name(::Type{CruzLineSearch}) = "Cruz non-monotonic linesearch"

@inline function Base.push!(ls::CruzLineSearch, f)
    if length(ls.prev_f) < ls.n_prev
        push!(ls.prev_f, f)
    else
        # throw away an old element
        shift!(ls.prev_f)

        # then add the newest one
        push!(ls.prev_f, f)
    end
end

function (ls::CruzLineSearch)(func!, x_k, d, f_k, eta)
    alpha_p = 1.0
    alpha_m = 1.0
    alpha = 1.0

    fp = 1.0
    nfev = 0
    f_k = back(ls.prev_f)
    f_bar = maximum(ls.prev_f)
    while true
        for i in 1:ls.nx
            ls.xp[i] = x_k[i] + alpha_p * d[i]
        end
        func!(ls.xp, ls.Fp); nfev += 1
        fp = _fmerit(ls.Fp)

        fp <= f_bar + eta - ls.gamma * alpha_p^2 * f_k && break

        alpha_tp = alpha_p^2 * f_k / (fp + (2*alpha_p - 1)*f_k)

        for i in 1:ls.nx
            ls.xp[i] = x_k[i] - alpha_m * d[i]
        end
        func!(ls.xp, ls.Fp); nfev += 1
        fp = _fmerit(ls.Fp)

        fp <= f_bar + eta - ls.gamma * alpha_m^2 * f_k && break

        alpha_tm = alpha_m^2 * f_k / (fp + (2*alpha_m - 1)*f_k)

        alpha_p = clamp(alpha_tp, ls.tau_min * alpha_p, ls.tau_max * alpha_p)
        alpha_m = clamp(alpha_tm, ls.tau_min * alpha_m, ls.tau_max * alpha_m)
    end

    return fp, nfev

end

# Define default merit, norm, and l2 norm squared
_fmerit(F) = dot(F, F)
_eta_strategy(k, x, F, f0) = f0 / ((1 + k)*(1 + k))

macro dfsanetrace(stepnorm)
    esc(quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x_k)     # current x
                dt["f(x)"] = copy(F_k)  # current f(x)
                dt["σ_k"] = σ_k         # current sigma
                dt["d"] = copy(d)       # current search direction
            end
            NLsolve.update!(tr,
                    k,
                    maximum(abs, F_k),
                    $stepnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end)
end

function _df_sane{T,lsT<:NonMonLineSearch}(func!, x0::AbstractVector{T},
                                           ::Type{lsT},
                                           σ_0::Float64,
                                           σ_eps::Float64,
                                           n_prev::Int,
                                           maxfev::Int,
                                           ftol::Float64,
                                           xtol::Float64,
                                           iterations::Int,
                                           store_trace::Bool,
                                           show_trace::Bool,
                                           extended_trace::Bool)
    # set up auxiliary variables
    k = 0             # iteration number
    nfev = 0          # number of function evaluations
    nx = length(x0)  # length of input vector

    # allocate memory for workspace arrays
    x_k = copy(x0)      # current x
    xp = similar(x_k)   # next x
    F_k = Array(T, nx)  # current residuals
    Fp = similar(F_k)   # next residuals
    d = similar(x_k)    # search direction

    # compute initial residuals and merit
    func!(x_k, F_k)
    f_k = _fmerit(F_k)
    f_0 = f_k
    σ_k = σ_0

    # set up line search
    ls! = lsT(xp, Fp, n_prev, 1.0, f_0)
    push!(ls!, f_0)

    # set up algorithm tracking variables
    xconverged = false
    fconverged = false
    converged = false
    tr = NLsolve.SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    if show_trace
        @printf "Iter     f(x) inf-norm    Step 2-norm \n"
        @printf "------   --------------   --------------\n"
    end
    @dfsanetrace convert(T, NaN)

    while k < iterations && !converged
        # Control spectral parameter, from [2]
        if abs(σ_k) > 1/σ_eps
            σ_k = 1/σ_eps * sign(σ_k)
        elseif abs(σ_k) < σ_eps
            σ_k = σ_eps
        end

        # Line search direction
        @inbounds for i in 1:nx
            d[i] = -σ_k * F_k[i]
        end

        # Nonmonotone line search
        eta = _eta_strategy(k, x_k, F_k, f_0)
        fp, _nfev = ls!(func!, x_k, d, f_k, eta)
        push!(ls!, fp)
        nfev += _nfev

        # Update spectral parameter
        s_k_norm = 0.0
        sy_k_norm = 0.0
        @inbounds for i in 1:nx
            s_k_norm += (xp[i] - x_k[i])*(xp[i] - x_k[i])
            sy_k_norm += (xp[i] - x_k[i])*(Fp[i] - F_k[i])
        end
        σ_k = s_k_norm / sy_k_norm

        # update the trace
        @dfsanetrace sqeuclidean(xp, x_k)

        # check convergence
        x_norm = chebyshev(xp, x_k)
        xconverged = x_norm < xtol
        f_norm = maximum(abs, Fp)
        fconverged = fp < ftol
        converged = xconverged || fconverged

        # Take step.
        copy!(x_k, xp)
        copy!(F_k, Fp)
        f_k = fp
        k += 1
    end

    NLsolve.SolverResults("df-sane with $(_name(lsT))",
                          x0, x_k, maximum(abs, Fp),
                          k, xconverged, xtol, fconverged, ftol,
                          tr, nfev, 0
                          )
end

function df_sane{T}(func!,
                    x0::AbstractVector{T};
                    linesearch::Symbol=:cheng,
                    n_prev::Int=10,
                    σ_0::Float64=1.0,
                    σ_eps::Float64=1e-10,
                    maxfev::Int=1_000,
                    ftol::Float64=convert(T, 1e-10),
                    xtol::Float64=1e-15,
                    iterations::Int=3_000,
                    store_trace::Bool=false,
                    show_trace::Bool=false,
                    extended_trace::Bool=false)

    lsT = linesearch == :cruz ? CruzLineSearch : ChengLineSearch
    _df_sane(func!, x0, lsT, σ_0, σ_eps, n_prev, maxfev, ftol, xtol, iterations,
             store_trace, show_trace, extended_trace)
end
