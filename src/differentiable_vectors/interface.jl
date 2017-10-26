value_jacobian!(df, x) = value_jacobian!(df, df.F, df.J, x)
function value_jacobian!(df, fvec, fjac, x)
    df.fj!(fvec, fjac, x)
    df.f_calls .+= 1
    df.j_calls .+= 1
end

jacobian!(df, x) = jacobian!(df, df.J, x)
function jacobian!(df, fjac, x)
    df.j!(fjac, x)
    df.j_calls .+= 1
end
jacobian(df) = df.J

value!(df, x) = value!(df, df.F, x)
function value!(df, fvec, x)
    df.f!(fvec, x)
    df.f_calls .+= 1
end
value(df) = df.F
