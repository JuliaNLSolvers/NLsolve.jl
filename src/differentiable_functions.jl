abstract type AbstractDifferentiableMultivariateFunction end

struct DifferentiableMultivariateFunction{F1,F2,F3} <: AbstractDifferentiableMultivariateFunction
    f!::F1
    g!::F2
    fg!::F3
end

alloc_jacobian(df::DifferentiableMultivariateFunction, T::Type, n::Integer) = Array{T}(n, n)

function DifferentiableMultivariateFunction(f!::F1, g!::F2,
                                            initial_x::AbstractArray) where {F1,F2}
    return DifferentiableMultivariateFunction(reshape_f(f!, initial_x),
                                              reshape_g(g!, initial_x))
end

function DifferentiableMultivariateFunction(f!, g!, fg!, initial_x::AbstractArray)
    return DifferentiableMultivariateFunction(reshape_f(f!, initial_x),
                                              reshape_g(g!, initial_x),
                                              reshape_fg(fg!, initial_x))
end

function DifferentiableMultivariateFunction(f!, g!)
    function fg!(x, fx, gx)
        f!(x, fx)
        g!(x, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function DifferentiableMultivariateFunction(f!; dtype::Symbol=:central)
    function fg!(x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix)
        f!(x, fx)
        function f(x::AbstractVector)
            fx = similar(x)
            f!(x, fx)
            return fx
        end
        finite_difference_jacobian!(f, x, fx, gx, dtype)
    end
    function g!(x, gx)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function DifferentiableMultivariateFunction(f!, initial_x::AbstractArray;
                                            dtype::Symbol=:central)
    return DifferentiableMultivariateFunction(reshape_f(f!, initial_x); dtype=dtype)
end

# Helper for the case where only f! and fg! are available
function only_f!_and_fg!(f!, fg!)
    function g!(x, gx)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function only_f!_and_fg!(f!, fg!, initial_x::AbstractArray)
    return only_f!_and_fg!(reshape_f(f!, initial_x),
                           reshape_fg(fg!, initial_x))
end

# Helper for the case where only fg! is available
function only_fg!(fg!)
    function f!(x, fx)
        gx = Array{eltype(x)}(length(x), length(x))
        fg!(x, fx, gx)
    end
    function g!(x, gx)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function only_fg!(fg!, initial_x::AbstractArray)
    return only_fg!(reshape_fg(fg!, initial_x))
end

# For sparse Jacobians
struct DifferentiableSparseMultivariateFunction{F1,F2,F3} <: AbstractDifferentiableMultivariateFunction
    f!::F1
    g!::F2
    fg!::F3
end

alloc_jacobian(df::DifferentiableSparseMultivariateFunction, T::Type, n::Integer) = spzeros(T, n, n)

function DifferentiableSparseMultivariateFunction(f!::F1, g!::F2,
                                                  initial_x::AbstractArray) where {F1,F2}
    return DifferentiableSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                    reshape_g_sparse(g!, initial_x))
end

function DifferentiableSparseMultivariateFunction(f!, g!, fg!, initial_x::AbstractArray)
    return DifferentiableSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                    reshape_g_sparse(g!, initial_x),
                                                    reshape_fg_sparse(fg!, initial_x))
end

function DifferentiableSparseMultivariateFunction(f!, g!)
    function fg!(x::AbstractVector, fx::AbstractVector, gx::SparseMatrixCSC)
        f!(x, fx)
        g!(x, gx)
    end
    return DifferentiableSparseMultivariateFunction(f!, g!, fg!)
end

struct DifferentiableGivenSparseMultivariateFunction{F1,F2,F3,Tv,Ti} <: AbstractDifferentiableMultivariateFunction
    f!::F1
    g!::F2
    fg!::F3
    J::SparseMatrixCSC{Tv, Ti}
end

alloc_jacobian(df::DifferentiableGivenSparseMultivariateFunction, args...) = deepcopy(df.J)

function DifferentiableGivenSparseMultivariateFunction(f!, g!, fg!, J::SparseMatrixCSC,
                                                       initial_x::AbstractArray)
    return DifferentiableGivenSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                         reshape_g_sparse(g!, initial_x),
                                                         reshape_fg_sparse(fg!, initial_x),
                                                         J)
end

function DifferentiableGivenSparseMultivariateFunction(f!::F1, g!::F2,
                                                       J::SparseMatrixCSC) where {F1,F2}
    function fg!(x::AbstractVector, fx::AbstractVector, gx::SparseMatrixCSC)
        f!(x, fx)
        g!(x, gx)
    end
    DifferentiableGivenSparseMultivariateFunction(f!, g!, fg!, J)
end

function DifferentiableGivenSparseMultivariateFunction(f!, g!, J::SparseMatrixCSC,
                                                       initial_x::AbstractArray)
    return DifferentiableGivenSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                         reshape_g_sparse(g!, initial_x),
                                                         J)
end
