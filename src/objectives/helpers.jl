

# Helpers for functions that do not modify arguments in place but return
function not_in_place(f)
    function f!(F, x)
        copy!(F, f(x))
    end
end

function not_in_place(f, j)
    not_in_place(f), not_in_place(j)
end

function not_in_place(f, j, fj)
    function fj!(F, J, x)
        f, j = fj(x)
        copy!(F, f)
        copy!(J, j)
    end
    not_in_place(f, j)..., fj!
end


# Helper for functions that take several scalar arguments and return a tuple
function n_ary(f)
    f!(fx, x) = copy!(fx, [f(x...)... ])
end
