# generates a function that computes the jacobian of f!(x,fx)
# assuming that f takes a Vector{T} of length n
# and writes the result to a Vector{T} of length m

# TODO: Update chunk_size to constant ~10 when https://github.com/JuliaDiff/ForwardDiff.jl/issues/36
# is resolved

# Compute the chunk size so that the chunk size is smaller than
# ForwardDiff.tuple_usage_threshold and evenly divides the input length
function compute_chunk_size(length_x0)
    chunk_size = ForwardDiff.tuple_usage_threshold
    while chunk_size > 1
        if isinteger(length_x0 / chunk_size)
            break
        else
            chunk_size -= 1
        end
    end
    return chunk_size
end

function autodiff{T <: Real}(f!, ::Type{T}, length_x0)

    cache = ForwardDiffCache()
    nl_chunk_size = compute_chunk_size(length_x0)

    permf!(yp, xp) = f!(xp, yp)
    permg! = jacobian(permf!, mutates = true, output_length = length_x0,
                      chunk_size = nl_chunk_size, cache = cache)
    permg_allres! = jacobian(permf!, ForwardDiff.AllResults, mutates = true,
                             output_length = length_x0, chunk_size = nl_chunk_size, cache = cache)

    g!(x, gx) = permg!(gx, x)

    function fg!(x, fx, gx)
        _, all_results = permg_allres!(gx, x)
        ForwardDiff.value!(fx, all_results)
    end
    return DifferentiableMultivariateFunction(f!, g!, fg!)
end
