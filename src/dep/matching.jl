#experiment with different matching function
function solveModel2(modd; max_iter1 = 50, max_iter2 = 1000, max_iter3 = 1000,
    tol1 = 10^-7, tol2 = 10^-8, tol3 =  10^-8, noisy = true, q_lb_0 =  0.0, q_ub_0 = 1.0)

    @unpack β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, procyclical, N_z, z0, z0_idx = modd  

    # set tolerance parameters for outermost loop
    err1  = 10
    iter1 = 1
    # initialize tolerance parameters for inner loops (for export)
    err2  = 10
    iter2 = 1
    err3  = 10
    iter3 = 1

    # Initialize default values and search parameters
    ω_0    = procyclical ? ω[z0_idx] : ω # unemployment value at z0
    q_lb   = q_lb_0          # lower search bound for q(θ)
    q_ub   = q_ub_0          # upper search bound for q(θ)
    q_0    = (q_lb + q_ub)/2 # initial guess for q(θ)
    α      = 0               # dampening parameter
    Y_0    = 0               # initalize Y_0 for export
    U      = 0               # initalize worker's EU from contract for export
    w_0    = 0               # initialize initial wage constant for export
    W_0    = 0
    
    # Initialize series
    az     = zeros(N_z)   # a(z|z_0)                         
    yz     = zeros(N_z)   # y(z|z_0)                         
    a_flag = zeros(N_z)   # flag for a(z|z_0)                         

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 < max_iter1  

        if noisy 
            println(q_0)
        end

        # Look for a fixed point in Y(z | z_0), ∀ z
        err2   = 10
        iter2  = 1      
        Y_0    = ones(N_z)*(30*κ/q_0)   # initial guess for Y(z | z_0)
        
        @inbounds while err2 > tol2 && iter2 < max_iter2   
            w_0  = ψ*(Y_0[z0_idx] - κ/q_0) # constant for wage difference equation
            # Solve for optimal effort a(z | z_0)
            @inbounds for (iz,z) in enumerate(zgrid)
                az[iz], yz[iz], a_flag[iz] = optA(z, modd, w_0)
            end
            Y_1    = yz + β*(1-s)*P_z*Y_0    
            err2   = maximum(abs.(Y_0 - Y_1))  # Error       
            #α     = iter2 > 100 ? 0.75 : α 
            if (err2 > tol2) && (iter2 < max_iter2) 
                Y_0    = α*Y_0 + (1-α)*Y_1 
                iter2 += 1
            end
            #println(Y_0[z0_idx])
        end

        # Solve recursively for the PV utility from the contract
        err3  = 10
        iter3 = 1  
        W_0   = copy(ω) # initial guess
        flow  = -(1/2ψ)*(ψ*hp.(az)σ_η).^2 - h.(az) + β*s*P_z*ω
        @inbounds while err3 > tol3 && iter3 < max_iter3
            W_1  = flow + β*(1-s)*P_z*W_0
            err3 = maximum(abs.(W_1 - W_0))
            #α   = iter3 > 100 ? 0.75 : α 
            W_0  = α*W_0 + (1-α)*W_1
            #println(W_0[z0_idx])
            iter3 +=1
        end

        # Check the IR constraint (must bind)
        U      = (1/ψ)*log(w_0) + W_0[z0_idx] 
        err1   = abs(U - ω_0)

        # Upate θ accordingly
        if U > ω_0
            q_ub  = copy(q_0)
        elseif U < ω_0
            q_lb  = copy(q_0)
        end

        if (err1 > tol1) && (iter1 < max_iter1) 
            q_0    = (q_lb + q_ub)/2
            iter1 += 1
        end

        # stop if we get stuck near the bounds on θ
        if (abs(q_0 - q_lb_0) < 10^-8) || (abs(q_0 - q_ub_0) < 10^-8 )
            iter1 = max_iter1
            break
        end
    end


    θ_0 = (1.355/q_0)^(1/0.72)

    return (θ = θ_0, Y = Y_0[z0_idx], U = U, W_0 = W_0, ω_0 = ω_0, w_0 = w_0, mod = modd, 
    az = az, yz = yz, err1 = err1, err2 = err2, err3 = err3, iter1 = iter1, iter2 = iter2, iter3 = iter3, wage_flag = (w_0 < 0),
    effort_flag = maximum(a_flag), exit_flag1 = (iter1 >= max_iter1), exit_flag2 = (iter2 >= max_iter2), exit_flag3 = (iter3 >= max_iter3))
end
