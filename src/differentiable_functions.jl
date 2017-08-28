abstract type AbstractDifferentiableMultivariateFunction end

struct DifferentiableMultivariateFunction <: AbstractDifferentiableMultivariateFunction
    f!::Function
    g!::Function
    fg!::Function
end

alloc_jacobian(df::DifferentiableMultivariateFunction, T::Type, n::Integer) = Array{T}(n, n)

function DifferentiableMultivariateFunction(f!::Function, g!::Function, fg!::Function,
                                            initial_x::AbstractArray)
    return DifferentiableMultivariateFunction(reshape_f(f!, initial_x),
                                              reshape_g(g!, initial_x),
                                              reshape_fg(fg!, initial_x))
end

function DifferentiableMultivariateFunction(f!::Function, g!::Function)
    function fg!(x::AbstractVector, fx::AbstractVector, gx::AbstractMatrix)
        f!(x, fx)
        g!(x, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function DifferentiableMultivariateFunction(f!::Function, g!::Function,
                                            initial_x::AbstractArray)
    return DifferentiableMultivariateFunction(reshape_f(f!, initial_x),
                                              reshape_g(g!, initial_x))
end

function DifferentiableMultivariateFunction(f!::Function; dtype::Symbol=:central)
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

function DifferentiableMultivariateFunction(f!::Function, initial_x::AbstractArray;
                                            dtype::Symbol=:central)
    return DifferentiableMultivariateFunction(reshape_f(f!, initial_x); dtype=dtype)
end

# Helper for the case where only f! and fg! are available
function only_f!_and_fg!(f!::Function, fg!::Function)
    function g!(x::AbstractVector, gx::AbstractMatrix)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function only_f!_and_fg!(f!::Function, fg!::Function, initial_x::AbstractArray)
    return only_f!_and_fg!(reshape_f(f!, initial_x),
                           reshape_fg(fg!, initial_x))
end

# Helper for the case where only fg! is available
function only_fg!(fg!::Function)
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

function only_fg!(fg!::Function, initial_x::AbstractArray)
    return only_fg!(reshape_fg(fg!, initial_x))
end

# For sparse Jacobians
struct DifferentiableSparseMultivariateFunction <: AbstractDifferentiableMultivariateFunction
    f!::Function
    g!::Function
    fg!::Function
end

alloc_jacobian(df::DifferentiableSparseMultivariateFunction, T::Type, n::Integer) = spzeros(T, n, n)

function DifferentiableSparseMultivariateFunction(f!::Function, g!::Function, fg!::Function,
                                                  initial_x::AbstractArray)
    return DifferentiableSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                    reshape_g_sparse(g!, initial_x),
                                                    reshape_fg_sparse(fg!, initial_x))
end

function DifferentiableSparseMultivariateFunction(f!::Function, g!::Function)
    function fg!(x::AbstractVector, fx::AbstractVector, gx::SparseMatrixCSC)
        f!(x, fx)
        g!(x, gx)
    end
    return DifferentiableSparseMultivariateFunction(f!, g!, fg!)
end

function DifferentiableSparseMultivariateFunction(f!::Function, g!::Function,
                                                  initial_x::AbstractArray)
    return DifferentiableSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                    reshape_g_sparse(g!, initial_x))
end

struct DifferentiableGivenSparseMultivariateFunction{Tv, Ti} <: AbstractDifferentiableMultivariateFunction
    f!::Function
    g!::Function
    fg!::Function
    J::SparseMatrixCSC{Tv, Ti}
end

alloc_jacobian(df::DifferentiableGivenSparseMultivariateFunction, args...) = deepcopy(df.J)

function DifferentiableGivenSparseMultivariateFunction(f!::Function, g!::Function,
                                                       fg!::Function, J::SparseMatrixCSC,
                                                       initial_x::AbstractArray)
    return DifferentiableGivenSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                         reshape_g_sparse(g!, initial_x),
                                                         reshape_fg_sparse(fg!, initial_x),
                                                         J)
end

function DifferentiableGivenSparseMultivariateFunction(f!::Function, g!::Function,
                                                       J::SparseMatrixCSC)
    function fg!(x::AbstractVector, fx::AbstractVector, gx::SparseMatrixCSC)
        f!(x, fx)
        g!(x, gx)
    end
    DifferentiableGivenSparseMultivariateFunction(f!, g!, fg!, J)
end

function DifferentiableGivenSparseMultivariateFunction(f!::Function, g!::Function,
                                                       J::SparseMatrixCSC,
                                                       initial_x::AbstractArray)
    return DifferentiableGivenSparseMultivariateFunction(reshape_f(f!, initial_x),
                                                         reshape_g_sparse(g!, initial_x),
                                                         J)
end
