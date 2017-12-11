# Helper for functions that take several scalar arguments and return a tuple
function n_ary(f)
    f!(fx, x) = copy!(fx, [f(x...)... ])
end
