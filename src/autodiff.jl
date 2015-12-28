# generates a function that computes the jacobian of f!(x,fx)
# assuming that f takes a Vector{T} of length n
# and writes the result to a Vector{T} of length m

function autodiff{T <: Real}(f!,::Type{T}, m)
    permf!(yp, xp) = f!(xp, yp)
    permg! = jacobian(permf!, mutates = true, output_length = m)
    g!(x, fx) = permg!(fx, x)

    return DifferentiableMultivariateFunction(f!, g!)
end
