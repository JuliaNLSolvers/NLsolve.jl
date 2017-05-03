@compat abstract type AbstractDifferentiableMultivariateFunction end

immutable DifferentiableMultivariateFunction <: AbstractDifferentiableMultivariateFunction
    f!::Function
    g!::Function
    fg!::Function
end

alloc_jacobian(df::DifferentiableMultivariateFunction, T::Type, n::Integer) = Array{T}(n, n)

function DifferentiableMultivariateFunction(f!::Function, g!::Function)
    function fg!(x::Vector, fx::Vector, gx::Array)
        f!(x, fx)
        g!(x, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

function DifferentiableMultivariateFunction(f!::Function; dtype::Symbol=:central)
    function fg!(x::Vector, fx::Vector, gx::Array)
        f!(x, fx)
        function f(x::Vector)
            fx = similar(x)
            f!(x, fx)
            return fx
        end
        finite_difference_jacobian!(f, x, fx, gx, dtype)
    end
    function g!(x::Vector, gx::Array)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

# Helper for the case where only f! and fg! are available
function only_f!_and_fg!(f!::Function, fg!::Function)
    function g!(x::Vector, gx::Array)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

# Helper for the case where only fg! is available
function only_fg!(fg!::Function)
    function f!(x::Vector, fx::Vector)
        gx = Array{eltype(x)}(length(x), length(x))
        fg!(x, fx, gx)
    end
    function g!(x::Vector, gx::Array)
        fx = similar(x)
        fg!(x, fx, gx)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end

# For sparse Jacobians
immutable DifferentiableSparseMultivariateFunction <: AbstractDifferentiableMultivariateFunction
    f!::Function
    g!::Function
    fg!::Function
end

alloc_jacobian(df::DifferentiableSparseMultivariateFunction, T::Type, n::Integer) = spzeros(T, n, n)

function DifferentiableSparseMultivariateFunction(f!::Function, g!::Function)
    function fg!(x::Vector, fx::Vector, gx::SparseMatrixCSC)
        f!(x, fx)
        g!(x, gx)
    end
    return DifferentiableSparseMultivariateFunction(f!, g!, fg!)
end


immutable DifferentiableGivenSparseMultivariateFunction{Tv, Ti} <: AbstractDifferentiableMultivariateFunction
    f!::Function
    g!::Function
    fg!::Function
    J::SparseMatrixCSC{Tv, Ti}
end

alloc_jacobian(df::DifferentiableGivenSparseMultivariateFunction, args...) = deepcopy(df.J)

function DifferentiableGivenSparseMultivariateFunction(f!::Function, g!::Function, J::SparseMatrixCSC)
    function fg!(x::Vector, fx::Vector, gx::SparseMatrixCSC)
        f!(x, fx)
        g!(x, gx)
    end
    DifferentiableGivenSparseMultivariateFunction(f!, g!, fg!, J)
end
