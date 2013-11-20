# Translation of Dynare's solve1.m

function newton(func::Function, x::Vector{Float64})
    tolf = eps()^(1/3)
    tolx = eps()^(2/3)
    maxit = 500

    nn = length(x)
    fjac = zeros(nn,nn) 
    g = zeros(nn,1) 

    tolmin = tolx 

    stpmx = 100 

    check = 0 

    (fvec,) = func(x)

    i = find(!isfinite(fvec))

    if !isempty(i)
        error("During the resolution of the non-linear system, the evaluation of the following equation(s) resulted in a non-finite number: $(j1[i])")
    end

    f = 0.5*dot(fvec, fvec)

    if max(abs(fvec)) < tolf
        return x
    end

    stpmax = stpmx*max(sqrt(dot(x, x)), nn) 
    first_time = 1
    for its = 1:maxit
        println("Iteration $its")
        (fvec,fjac) = func(x)

        g = fjac'*fvec

        p = -fjac\fvec

        xold = x 
        fold = f 

        (x,f,fvec,check)=lnsrch(xold,fold,g,p,stpmax,func)

        if check == 1
            den = max(f, 0.5*nn)
            if max(abs(g).*max([abs(x[j2]) ones(nn,1)], (), 2))/den >= tolmin
                error("Spurious convergence")
            end
            return(x)
        end

        if max(abs(fvec)) < tolf
            return(x)
        end
    end

    error("Maximum iterations reached")
end

function lnsrch(xold::Vector{Float64},fold::Float64,g::Vector{Float64},p::Vector{Float64},stpmax::Float64,func::Function)
    local f, fvec, check
    
    alf = 1e-4 
    tolx = eps()^(2/3)
    alam = 1

    x = xold
    nn = length(x)
    summ = sqrt(sum(p.*p))
    if !isfinite(summ)
        error("Some element of Newton direction isn't finite. Jacobian maybe singular or there is a problem with initial values")
    end

    if summ > stpmax
        p=p.*stpmax/summ
    end

    slope = dot(g,p)

    test = max(abs(p)'./max([abs(xold)';ones(1,nn)],(),1))
    alamin = tolx/test

    if alamin > 0.1
        alamin = 0.1
    end

    while true
        if alam < alamin
            check = 1
            break
        end
    
        x = xold + (alam*p)
        (fvec,) = func(x)
        f = 0.5*dot(fvec,fvec)

        if any(isnan(fvec))
            alam = alam/2 
            alam2 = alam 
            f2 = f 
            fold2 = fold 
        else
            if f <= fold+alf*alam*slope
                check = 0
                break 
            else
                if alam == 1
                    tmplam = -slope/(2*(f-fold-slope)) 
                else
                    rhs1 = f-fold-alam*slope 
                    rhs2 = f2-fold2-alam2*slope 
                    a = (rhs1/(alam^2)-rhs2/(alam2^2))/(alam-alam2) 
                    b = (-alam2*rhs1/(alam^2)+alam*rhs2/(alam2^2))/(alam-alam2) 
                    if a == 0
                        tmplam = -slope/(2*b) 
                    else
                        disc = (b^2)-3*a*slope 

                        if disc < 0
                            error ("Roundoff problem")
                        else
                            tmplam = (-b+sqrt(disc))/(3*a) 
                        end

                    end

                    if tmplam > 0.5*alam
                        tmplam = 0.5*alam
                    end

                end

                alam2 = alam 
                f2 = f 
                fold2 = fold 
                alam = max(tmplam, 0.1*alam) 
            end
        end
    end
    return (x,f,fvec,check)
end
