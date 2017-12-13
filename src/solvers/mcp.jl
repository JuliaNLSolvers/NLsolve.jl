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
struct MCP
end
function mcp_smooth(df::OnceDifferentiable,
                    lower::Vector, upper::Vector)

    function f!(F, x)
        value!(df, F, x)
        for i = 1:length(x)
            if  isfinite.(upper[i])
                F[i] += (x[i]-upper[i]) + sqrt(F[i]^2+(x[i]-upper[i])^2)
            end
            if  isfinite.(lower[i])
                F[i] += (x[i]-lower[i]) - sqrt(F[i]^2+(x[i]-lower[i])^2)
            end
        end
    end

    function j!(J, x)
        F = similar(x)
        value_jacobian!(df, F, J, x)

        # Derivatives of phiplus
        sqplus = sqrt.(F.^2 .+ (x .- upper).^2)

        dplus_du = 1 .+ F./sqplus

        dplus_dv = similar(x)
        for i = 1:length(x)
            if isfinite.(upper[i])
                dplus_dv[i] = 1 + (x[i]-upper[i])/sqplus[i]
            else
                dplus_dv[i] = 0
            end
        end

        # Derivatives of phiminus
        phiplus = copy(F)
        for i = 1:length(x)
            if isfinite(upper[i])
                phiplus[i] += (x[i]-upper[i]) + sqplus[i]
            end
        end

        sqminus = sqrt.(phiplus.^2 .+ (x .- lower).^2)

        dminus_du = 1.-phiplus./sqminus

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
                J[i,j] *= dminus_du[i]*dplus_du[i]
            end
            J[i,i] += dminus_dv[i] + dminus_du[i]*dplus_dv[i]
        end
    end
    return OnceDifferentiable(f!, j!, similar(df.F), similar(df.x_f))
end

# Generate a function whose roots are the solutions of the MCP.
# More precisely, this function is:
#
# x -> min(max(f(x), x-upper), x-lower)
#
# Note that Miranda and Fackler use the opposite sign convention for the MCP,
# hence the difference in the function.
function mcp_minmax(df::OnceDifferentiable,
                    lower::Vector, upper::Vector)
    function f!(F, x)
        value!(df, F, x)
        for i = 1:length(x)
            if F[i] < x[i]-upper[i]
                F[i] = x[i]-upper[i]
            end
            if F[i] > x[i]-lower[i]
                F[i] = x[i]-lower[i]
            end
        end
    end

    function j!(J, x)
        F = similar(x)
        value_jacobian!(df, F, J, x)
        for i = 1:length(x)
            if F[i] < x[i]-upper[i] || F[i] > x[i]-lower[i]
                for j = 1:length(x)
                    J[i,j] = (i == j ? 1.0 : 0.0)
                end
            end
        end
    end
    return OnceDifferentiable(f!, j!, similar(df.F), similar(df.x_f))
end
