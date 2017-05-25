# Generate a smooth function whose roots are the solutions of the MCP.
# More precisely, this function is:
#
# x -> phiminus(phiplus(f(x), x-upper), x-lower)
#
# where
#  phiplus(u, v) = u + v + sqrt(u^2+v^2)
#  phiminus(u, v) = u + v - sqrt(u^2+v^2)
# are the Fischer functions.
#
# Note that Miranda and Fackler use the opposite sign convention for the MCP,
# hence the difference in the smooth function.
function mcp_smooth(df::AbstractDifferentiableMultivariateFunction,
                    lower::Vector, upper::Vector)

    function f!(x::Vector, fx::Vector)
        df.f!(x, fx)
        for i = 1:length(x)
            if  isfinite.(upper[i])
                fx[i] += (x[i]-upper[i]) + sqrt(fx[i]^2+(x[i]-upper[i])^2)
            end
            if  isfinite.(lower[i])
                fx[i] += (x[i]-lower[i]) - sqrt(fx[i]^2+(x[i]-lower[i])^2)
            end
        end
    end

    function g!(x::Vector, gx::AbstractArray)
        fx = similar(x)
        df.fg!(x, fx, gx)

        # Derivatives of phiplus
        sqplus = sqrt.(fx.^2 .+ (x .- upper).^2)

        dplus_du = 1 + fx./sqplus

        dplus_dv = similar(x)
        for i = 1:length(x)
            if isfinite.(upper[i])
                dplus_dv[i] = 1 + (x[i]-upper[i])/sqplus[i]
            else
                dplus_dv[i] = 0
            end
        end

        # Derivatives of phiminus
        phiplus = copy(fx)
        for i = 1:length(x)
            if isfinite(upper[i])
                phiplus[i] += (x[i]-upper[i]) + sqplus[i]
            end
        end

        sqminus = sqrt.(phiplus.^2 .+ (x .- lower).^2)

        dminus_du = 1-phiplus./sqminus

        dminus_dv = similar(x)
        for i = 1:length(x)
            if isfinite.(lower[i])
                dminus_dv[i] = 1 - (x[i]-lower[i])/sqminus[i]
            else
                dminus_dv[i] = 0
            end
        end

        # Final computations
        for i = 1:length(x)
            for j = 1:length(x)
                gx[i,j] *= dminus_du[i]*dplus_du[i]
            end
            gx[i,i] += dminus_dv[i] + dminus_du[i]*dplus_dv[i]
        end
    end

    dfT = typeof(df)

    return dfT(f!, g!)
end

# Generate a function whose roots are the solutions of the MCP.
# More precisely, this function is:
#
# x -> min(max(f(x), x-upper), x-lower)
#
# Note that Miranda and Fackler use the opposite sign convention for the MCP,
# hence the difference in the function.
function mcp_minmax(df::AbstractDifferentiableMultivariateFunction,
                    lower::Vector, upper::Vector)

    function f!(x::Vector, fx::Vector)
        df.f!(x, fx)
        for i = 1:length(x)
            if fx[i] < x[i]-upper[i]
                fx[i] = x[i]-upper[i]
            end
            if fx[i] > x[i]-lower[i]
                fx[i] = x[i]-lower[i]
            end
        end
    end

    function g!(x::Vector, gx::AbstractArray)
        fx = similar(x)
        df.fg!(x, fx, gx)
        for i = 1:length(x)
            if fx[i] < x[i]-upper[i] || fx[i] > x[i]-lower[i]
                for j = 1:length(x)
                    gx[i,j] = (i == j ? 1.0 : 0.0)
                end
            end
        end
    end

    dfT = typeof(df)

    return dfT(f!, g!)
end
