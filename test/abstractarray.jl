# test use of arbitrary arrays with Wood example from MINPACK

@testset "abstractarray" begin

using MappedArrays

const c3 = 2e2
const c4 = 2.02e1
const c5 = 1.98e1
const c6 = 1.8e2

function f!(x, fvec)
    temp1 = x[2] - x[1]^2
    temp2 = x[4] - x[3]^2
    fvec[1] = -c3*x[1]*temp1 - (1 - x[1])
    fvec[2] = c3*temp1 + c4*(x[2] - 1) + c5*(x[4] - 1)
    fvec[3] = -c6*x[3]*temp2 - (1 - x[3])
    fvec[4] = c6*temp2 + c4*(x[4] - 1) + c5*(x[2] - 1)
end

function g!(x, fjac)
    fill!(fjac, 0)
    temp1 = x[2] - 3x[1]^2
    temp2 = x[4] - 3x[3]^2
    fjac[1,1] = -c3*temp1 + 1
    fjac[1,2] = -c3*x[1]
    fjac[2,1] = -2*c3*x[1]
    fjac[2,2] = c3 + c4
    fjac[2,4] = c5
    fjac[3,3] = -c6*temp2 + 1
    fjac[3,4] = -c6*x[3]
    fjac[4,2] = c5
    fjac[4,3] = -2*c6*x[3]
    fjac[4,4] = c6 + c4
end

initial_x_matrix = [-3. -3; -1 -1]
initial_x = vec(initial_x_matrix)
initial_x_mapped_matrix = mappedarray((x -> -2*x, x -> -x/2), [1.5 1.5; 0.5 0.5])

for method in (:trust_region, :newton, :anderson)
    r = nlsolve(f!, g!, initial_x, method = method)
    r_matrix = nlsolve(f!, g!, initial_x_matrix, method = method)
    r_mapped_matrix = nlsolve(f!, g!, initial_x_mapped_matrix, method = method)

    @test r.zero == vec(r_matrix.zero)
    @test r_matrix.zero == r_mapped_matrix.zero
    @test r.residual_norm == r_matrix.residual_norm
    @test r_matrix.residual_norm == r_mapped_matrix.residual_norm
    @test size(r.zero) == size(initial_x)
    @test size(r_matrix.zero) == size(initial_x_matrix)
    @test size(r_mapped_matrix.zero) == size(initial_x_mapped_matrix)

    r_AD = nlsolve(f!, initial_x, method = method, autodiff = true)
    r_matrix_AD = nlsolve(f!, initial_x_matrix, method = method, autodiff = true)
    r_mapped_matrix_AD = nlsolve(f!, initial_x_mapped_matrix, method = method,
                                 autodiff = true)

    @test r_AD.zero == vec(r_matrix_AD.zero)
    @test r_matrix_AD.zero == r_mapped_matrix_AD.zero
    @test r_AD.residual_norm == r_matrix_AD.residual_norm
    @test r_matrix_AD.residual_norm == r_mapped_matrix_AD.residual_norm
    @test size(r_AD.zero) == size(initial_x)
    @test size(r_matrix_AD.zero) == size(initial_x_matrix)
    @test size(r_mapped_matrix_AD.zero) == size(initial_x_mapped_matrix)
end

end
