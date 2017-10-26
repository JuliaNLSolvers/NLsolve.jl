# AbstractDifferentiableVector interface
# ---
# The idea of using an encapsulating type that holds both f!, j!, and fj! as well
# as cache variables for the function values (fvec), Jacobians (J) and more.
#
# There ends up being quite a few combinations, but it is not the ambition that
# users should use these directly. There's an abstract version to allow further
# use of the abstraction later on.
#
# For reference, we're trying to solve
#    F(x) = 0
# where we might need the Jacobian
#    J(x) ≡ ∇F(x).
# Below we use f!, j!, fj! as it is usual to lower case function names.
#
# We allow for different types for the x's and J-storage, but x must be of a type
# that is a subtype of AbstractArray. It's understood that if no J-storage is
# provided, a Matrix instance will be passed to j! or fj!. If x is not a vector
# type (subtyping AbstractVector), we will use reshape(...) to make sure every-
# thing matches what is expected in the algorithms.
abstract type AbstractDifferentiableVector end

# Uninitialized
# This is here to ease the use of nlsolve(...) to the user, no need to set up
# all the caches up front unless it's by an advanced user.
abstract type Uninitialized <: AbstractDifferentiableVector end
struct UninitializedDifferentiableVector{TF, TG, TFG, TJ} <: Uninitialized
    f!::TF # in-place function to calculate and store F(x)
    j!::TG # in-place function to calculate and store J(x)
    fj!::TFG # in-place function to calculate and store F(x) and J(x)
    J::TJ # cache variable to hold J(x)
end
# Uninitialized constructor with Jacobian keyword
UninitializedDifferentiableVector(f!, j!, fj!; J = nothing) =
    UninitializedDifferentiableVector(f!, j!, fj!, J)

# Initialized
# This is the actual type that we're using internally. Holds cache variables,
# and all three objective callables.
struct DifferentiableVector{TF, TG, TFG, FV, TJ} <: AbstractDifferentiableVector
    f!::TF # in-place function to calculate and store F(x)
    j!::TG # in-place function to calculate and store J(x)
    fj!::TFG # in-place function to calculate and store F(x) and J(x)
    F::FV # cache variable to hold F(x)
    J::TJ # cache variable to hold J(x)
    f_calls::Vector{Int} # number of calls that calculated F(x)
    j_calls::Vector{Int} # number of calls that calculated J(x)
end
alloc_J(J, x) = J
function alloc_J(J::Void, x)
    n = length(x)
    if J == nothing
        # Initialize an n-by-n Array{T, 2}
        J = Array{eltype(x)}(n, n)
    end
end
# Generic function to allocate the full DifferentiableVector given all functions,
# an initial point, and possibly a custom Jacobian cache variable (that j!, fj!)
# expect as first positional argument.
function DifferentiableVector(f!, j!, fj!, x_seed::AbstractVector; J = nothing)
    DifferentiableVector(f!, j!, fj!, similar(x_seed), alloc_J(J, x_seed), [0,], [0,])
end
# Uninitialized to DifferentiableVector conversion by overloading the call...
# ... when all functions are present
(udf::UninitializedDifferentiableVector)(x) =
    DifferentiableVector(udf.f!, udf.j!, udf.fj!, x; J = udf.J)
# ... when fj! is not present
(udf::UninitializedDifferentiableVector{TF, TG, TFG, TJ})(x) where {TF, TG, TFG<:Void, TJ} =
    DifferentiableVector(udf.f!, udf.j!, x; J = udf.J)
# ... when only fj! is present
(udf::UninitializedDifferentiableVector{TF, TG, TFG, TJ})(x) where {TF<:Void, TG<:Void, TFG, TJ} =
    only_fj!(udf.fj!, x)
# ... when f! and fj! is present
function (udf::UninitializedDifferentiableVector{TF, TG, TFG, TJ})(x) where {TF, TG<:Void, TFG, TJ}
    only_f!_and_fj!(udf.f!, udf.fj!, x)
end
# ... when only f! is present
function (udf::UninitializedDifferentiableVector{TF, TG, TFG, TJ})(x) where {TF, TG<:Void, TFG<:Void, TJ}
    DifferentiableVector(udf.f!, x; J = udf.J)
end
# When only f! is given, create an uninitialized objective without j!, fj! and possibly J
DifferentiableVector(f!; J = nothing) = UninitializedDifferentiableVector(f!, nothing, nothing, J)
# When f!, j! is given but no x, create uninitialized without fj!
DifferentiableVector(f!, j!; J = nothing) = UninitializedDifferentiableVector(f!, j!, nothing, J)
# When f!, j!, fj! is given without but no x, create a fully specified uninitialized
DifferentiableVector(f!, j!, fj!; J = nothing) = UninitializedDifferentiableVector(f!, j!, fj!, J)
# If an initial x is available, skip the Uinitialized state
# if f!, j!, and x_seed is given but not a vector, reshape f!, j! to accept
# vector shaped inputs. Jacobian can be omitted if it is a dense matrix.
function DifferentiableVector(f!, j!, x_seed::AbstractArray; J = nothing)
    rf! = reshape_f(f!, x_seed)
    rj! = reshape_j(j!, x_seed)
    function rfj!(fx, jx, x)
        rf!(fx, x)
        rj!(jx, x)
    end
    return DifferentiableVector(rf!, rj!, rfj!, x_seed; J = alloc_J(J, x_seed))
end
# if everything is provided but x_seed is not a vector, reshape the functions
function DifferentiableVector(f!, j!, fj!, x_seed::AbstractArray; J = nothing)
    DifferentiableVector(reshape_f(f!, x_seed),
                         reshape_j(j!, x_seed),
                         reshape_fj(fj!, x_seed),
                         copy(vec(x_seed)),
                         alloc_J(J, x_seed),
                         [0,], [0,])
end
# if f!, j!, and x_seed as a vector is given, create fj! and construct the type
function DifferentiableVector(f!, j!, x_seed::AbstractVector; J = nothing)
    function fj!(fx, jx, x)
        f!(fx, x)
        j!(jx, x)
    end
    return DifferentiableVector(f!, j!, fj!, x_seed; J = alloc_J(J, x_seed))
end
