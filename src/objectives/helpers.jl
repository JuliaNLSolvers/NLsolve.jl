

# Helpers for functions that do not modify arguments in place but return
function copy_value!(F, y)
    length(y) == length(F) || throw(DimensionMismatch("the function returned an output of length $(length(y)), but length $(length(F)) was expected. NLsolve solves square systems, with one residual per unknown. For least squares, see LsqFit.jl."))
    copyto!(F, y)
end

function not_in_place(f)
    function f!(F, x)
        copy_value!(F, f(x))
    end
end

function not_in_place(f, j)
    not_in_place(f), not_in_place(j)
end

function not_in_place(f, j, fj)
    function fj!(F, J, x)
        f, j = fj(x)
        copy_value!(F, f)
        copy_value!(J, j)
    end
    not_in_place(f, j)..., fj!
end


# Helper for functions that take several scalar arguments and return a tuple
function n_ary(f)
    f!(fx, x) = copyto!(fx, [f(x...)... ])
end
