# Helpers for the case where only f! and/or fj! are available...
# ... and no initial x is present, but a Jacobian cache might be
only_f!_and_fj!(f!, fj!, J = nothing) = UninitializedDifferentiableVector(f!, nothing, fj!, J)
# ... an initial vector shaped x is present and a Jacobian cache might be
function only_f!_and_fj!(f!, fj!, initial_x::AbstractVector, J = nothing)
    function j!(jx, x)
        fx = similar(x)
        fj!(fx, jx, x)
    end
    return DifferentiableVector(f!, j!, fj!, initial_x; J = J)
end
# ... an initial array shaped x is present, and a Jacobian cache might be
function only_f!_and_fj!(f!, fj!, initial_x::AbstractArray; J = nothing)
    return only_f!_and_fj!(reshape_f(f!, initial_x),
                           reshape_fj(fj!, initial_x); J = J)
end
# Helpers for the case where only fj! is available...
# ... and no initial x is present, but a Jacobian cache might be
only_fj!(fj!; J = nothing) = UninitializedDifferentiableVector(nothing, nothing, fj!, J)
# ... and an initial vector shaped x is present and a Jacobian cache might be
function only_fj!(fj!, initial_x::AbstractVector{T}; J = nothing) where T
    function f!(fx, x)
        jx = Matrix{T}(length(x), length(x))
        fj!(fx, jx, x)
    end
    function j!(jx, x)
        fx = similar(x)
        fj!(fx, jx, x)
    end
    return DifferentiableVector(f!, j!, fj!, initial_x; J = J)
end
# ... and an initial array shaped x is present, and a Jacobian cache might be
function only_fj!(fj!, initial_x::AbstractArray, J = nothing)
    return only_fj!(reshape_fj(fj!, initial_x), initial_x; J = J)
end

# Helpers for reshaping functions on arbitrary arrays to functions on vectors
function reshape_f(f!, initial_x::AbstractArray)
    function fvec!(fx, x)
        f!(reshape(fx, size(initial_x)...), reshape(x, size(initial_x)...))
    end
end

function reshape_j(j!, initial_x::AbstractArray)
    function gvec!(jx, x)
        j!(jx, reshape(x, size(initial_x)...))
    end
end

function reshape_fj(fj!, initial_x::AbstractArray)
    function fjvec!(fx, jx, x)
        fj!(reshape(fx, size(initial_x)...), jx, reshape(x, size(initial_x)...))
    end
end
# make sure that reshaping is cost-less in the "normal" case of vectors
# and matrices
reshape_f(f!, storage::AbstractVector) = f!
reshape_j(j!, storage::AbstractMatrix) = j!
reshape_fj(fj!, storage::AbstractVector) = fj!

# Helpers for functions that do not modify arguments in place
# if x is a vector shaped array, pass only the function
function not_in_place(f)
    function f!(fx, x)
        copy!(fx, f(x))
    end
end
# if x is not a vector shaped array, pass the array so we can reshape
function not_in_place(f, initial_x::AbstractArray; J = nothing)
    function f!(fx, x)
        copy!(reshape(fx, size(initial_x)...), f(reshape(x, size(initial_x)...)))
    end
end
# if x is a vector shaped array, pass only the functions
function not_in_place(f, g; J = nothing)
    DifferentiableVector(not_in_place(f), not_in_place_g(g); J = J)
end
# if x is not a vector shaped array, pass the array so we can reshape
function not_in_place(f, g, initial_x::AbstractArray; J = nothing)
    DifferentiableVector(not_in_place(f, initial_x),
                         not_in_place_g(g, initial_x); J = J)
end
# if x is a vector shaped array, pass only the functions
function not_in_place(f, g, fj; J = nothing)
    DifferentiableVector(not_in_place(f),
                         not_in_place_g(g),
                         not_in_place_fj(fj);
                         J = J)
end
# if x is not a vector shaped array, pass the array so we can reshape
function not_in_place(f, g, fj, initial_x::AbstractArray; J = nothing)
    DifferentiableVector(not_in_place(f, initial_x),
                         not_in_place_g(g, initial_x),
                         not_in_place_fj(fj, initial_x); J = J)
end
# if x is a vector shaped array, pass only the function
function not_in_place_g(g)
    function j!(jx, x)
        copy!(jx, g(x))
    end
end
# if x is not a vector shaped array, pass the array so we can reshape
function not_in_place_g(g, initial_x::AbstractArray)
    function j!(jx, x)
        copy!(jx, g(reshape(x, size(initial_x)...)))
    end
end
# if x is a vector shaped array, pass only the function
function not_in_place_fj(fj)
    function fj!(fx, jx, x)
        (F, J) = fj(x)
        copy!(fx, F)
        copy!(jx, J)
    end
end
# if x is not a vector shaped array, pass the array so we can reshape
function not_in_place_fj(fj, initial_x::AbstractArray)
    function fj!(fx, jx, x)
        (fvec, fjac) = fj(reshape(x, size(initial_x)...))
        copy!(reshape(fx, size(initial_x)...), fvec)
        copy!(jx, fjac)
    end
end

# Helper for functions that take several scalar arguments and return a tuple
function n_ary(f)
    f!(fx, x) = copy!(fx, [f(x...)... ])
end
