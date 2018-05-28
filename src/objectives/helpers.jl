

# Helpers for functions that do not modify arguments in place but return
function not_in_place(f)
    function f!(F, x)
        copyto!(F, f(x))
    end
end

function not_in_place(f, j)
    not_in_place(f), not_in_place(j)
end

function not_in_place(f, j, fj)
    function fj!(F, J, x)
        f, j = fj(x)
        copyto!(F, f)
        copyto!(J, j)
    end
    not_in_place(f, j)..., fj!
end


# Helper for functions that take several scalar arguments and return a tuple
function n_ary(f)
    f!(fx, x) = copyto!(fx, [f(x...)... ])
end
