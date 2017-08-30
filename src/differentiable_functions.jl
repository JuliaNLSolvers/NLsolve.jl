abstract type AbstractDifferentiableMultivariateFunction end

struct DifferentiableMultivariateFunction{F1,F2,F3} <: AbstractDifferentiableMultivariateFunction
    f!::F1
    g!::F2
    fg!::F3

    DifferentiableMultivariateFunction{F1,F2,F3}(f!, g!, fg!) where {F1,F2,F3} =
        new{F1,F2,F3}(f!, g!, fg!)
end

alloc_jacobian(df::DifferentiableMultivariateFunction, T::Type, n::Integer) = Array{T}(n, n)

DifferentiableMultivariateFunction(f!, g!, fg!) =
    DifferentiableMultivariateFunction{typeof(f!),typeof(g!),typeof(fg!)}(f!, g!, fg!)

function DifferentiableMultivariateFunction(f!, g!, initial_x::AbstractArray)
    return DifferentiableMultivariateFunction(reshape_f(f!, initial_x),
                                              reshape_g(g!, initial_x))
end

function DifferentiableMultivariateFunction(f!, g!, fg!, initial_x::AbstractArray)
    return DifferentiableMultivariateFunction(reshape_f(f!, initial_x),
                                              reshape_g(g!, initial_x),
                                              reshape_fg(fg!, initial_x))
end

function DifferentiableMultivariateFunction(f!, g!)
    function fg!(x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix)
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
    function g!(x::AbstractVector, gx::AbstractMatrix)
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
    function g!(x::AbstractVector, gx::AbstractMatrix)
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
    function f!(x::AbstractVector, fx::AbstractVector)
        gx = Array{eltype(x)}(length(x), length(x))
        fg!(x, fx, gx)
    end
    function g!(x::AbstractVector, gx::AbstractMatrix)
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

    DifferentiableSparseMultivariateFunction{F1,F2,F3}(f!, g!, fg!) where {F1,F2,F3} =
        new{F1,F2,F3}(f!, g!, fg!)
end

alloc_jacobian(df::DifferentiableSparseMultivariateFunction, T::Type, n::Integer) = spzeros(T, n, n)

DifferentiableSparseMultivariateFunction(f!, g!, fg!) =
    DifferentiableSparseMultivariateFunction{typeof(f!),typeof(g!),typeof(fg!)}(f!, g!, fg!)

function DifferentiableSparseMultivariateFunction(f!, g!, initial_x::AbstractArray)
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

    DifferentiableGivenSparseMultivariateFunction{F1,F2,F3,Tv,Ti}(f!, g!, fg!, J) where {F1,F2,F3,Tv,Ti} =
        new{F1,F2,F3,Tv,Ti}(f!, g!, fg!, J)
end

alloc_jacobian(df::DifferentiableGivenSparseMultivariateFunction, args...) = deepcopy(df.J)

DifferentiableGivenSparseMultivariateFunction(f!, g!, fg!, J::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} =
    DifferentiableGivenSparseMultivariateFunction{typeof(f!),typeof(g!),typeof(fg!),Tv,Ti}(f!, g!, fg!, J)

function DifferentiableGivenSparseMultivariateFunction(f!, g!, fg!, J::SparseMatrixCSC,
                                                       initial_x::AbstractArray)
    return DifferentiableGivenSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                         reshape_g_sparse(g!, initial_x),
                                                         reshape_fg_sparse(fg!, initial_x),
                                                         J)
end

function DifferentiableGivenSparseMultivariateFunction(f!, g!, J::SparseMatrixCSC)
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
