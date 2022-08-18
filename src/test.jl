"""
Solve for the optimal effort a(z,t).
"""
function optA(z, t, modd, Y, θ)
    @unpack ψ, ε, q, κ, hp, σ_η = modd
    ψ_t = ψ[t]
    w0  = ψ[1]*(Y - κ/q(θ))    # time-0 earnings (constant)
    if ε == 1 # can solve analytically
        aa = (z/w0 + sqrt((z/w0)^2))/2(1 + ψ_t*σ_η^2)
    else # exclude the choice of zero effort
        aa = find_zeros(x -> x - max(z*x/w0 -  (ψ_t/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0, 20) 
    end
    a    = ~isempty(aa) ? aa[1] : 0
    y    = a*z # Expectation of y_t over η_t (given z_t)
    flag = ~isempty(aa) ? ((z*aa[1]/w0 + (ψ_t/ε)*(hp(aa[1])*σ_η)^2) < 0) : isempty(aa) 
    flag += (w0 < 0)
    return a, y, flag
end

# simulate 
function simulateProd(P_z, zgrid, T; N = 10000, seed = 211, set_seed = true, z0 = median(1:length(zgrid)))
    if set_seed == true
        Random.seed!(seed)
    end
    sim      = rand(N, T)            # N X T - draw uniform numbers in (0,1)
    zt       = zeros(Int64, N, T)    # N X T - index on productivity grid
    zt[:,1] .= floor(Int64, z0)      # Set z0 with valid index
    CDF      = cumsum(P_z, dims = 2) # CDF for transition probs. given initial state
    probs    = ones(N)               # N x 1 - probability of a given sequence

    @inbounds for t = 2:T
        @inbounds for i = 1:N
            zt[i, t]  = findfirst(x-> x >=  sim[i,t], CDF[zt[i,t-1], :]) 
            probs[i]  =  P_z[zt[i,t-1], zt[i,t]]*probs[i]
        end
    end
    return zgrid[zt], probs, zt
end


function solveModel2(m; N_sim = 20000, max_iter1 = 100, max_iter2 = 100, tol1 = 10^-7, tol2 = 10^-8, noisy = true)

    @unpack T, β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, savings, procyclical, N_z = m   
    if (savings) error("Use solveModelSavings") end 

    # set tolerance parameters for inner and outer loops
    err1  = 10
    err2  = 10
    iter1 = 1
    iter2 = 1

    # Initialize default values and search parameters
    θ_lb  = 0.0             # lower search bound
    θ_ub  = 10.0            # upper search bound
    θ_0   = (θ_lb + θ_ub)/2 # initial guess for θ
    α     = 0.25            # dampening parameter
    Y_0   = 0               # initalize Y_0 for export
    IR    = 0               # initalize IR for export
    w0    = 0               # initialize initial wage constant for export

    #  simulate productivity paths for computing expectations
    ZZ, probs, IZ  = simulateProd(P_z, zgrid, T; N = N_sim) # N x T
    az             = zeros(N_z, T)    # N_z x T                         
    yz             = zeros(N_z, T)    # N_z x T                        
    flag           = zeros(N_z, T)    # N_z x T                          
    AA             = zeros(N_sim, T)  # N x T                   
    YY             = zeros(N_sim, T)  # N x T                   
    @views ω0      = procyclical ? ω[IZ[1,1]] : ω # unemployment value at z0

    # look for a fixed point in θ
    @inbounds while err1 > tol1 && iter1 < max_iter1
        err2   = 10
        iter2  = 1
        Y_lb   = κ/q(θ_0)         # lower search bound
        Y_ub   = 100*κ/q(θ_0)      # upper search bound
        Y_0    = (Y_lb + Y_ub)/2  # initial guess for Y
        # look for a fixed point in Y0
        @inbounds while err2 > tol2 && iter2 < max_iter2
            # solve for optimal effort a_t, and implied (expected) y_t
            @inbounds for t = 1:T
                @inbounds for (iz,z) in enumerate(zgrid)
                    az[iz,t], yz[iz,t], flag[iz,t] = optA(z, t, m, Y_0, θ_0)
                end 
            end
            @views @inbounds for t=1:T
                AA[:,t] .= az[IZ[:,t],t] 
                YY[:,t] .= yz[IZ[:,t],t] 
            end
            Y    = mapreduce(t -> YY[:,t] *(β*(1-s))^(t-1), +, 1:T)  # compute PV of Y_i
            Y_1  = mean(Y)         # Numerical approximation of E_0[Y]
            err2 = abs(Y_0 - Y_1)  # Error       

            #= if doing bisection search on Y_0 
            if Y_1 < Y_0 
                Y_ub  = copy(Y_0)
            elseif Y_1 > Y_0 || w0 < 0
                Y_lb  = copy(Y_0)
            end
            Y_0  = 0.5(Y_lb + Y_ub) 
            # Note: delivers ≈ Y_0, but converges more slowly. =#

            α      = iter2 > 50 ? 0.75 : α
            Y_0    = max(α*Y_0 + (1-α)*Y_1, κ/q(θ_0))
            iter2 += 1
            #println(Y_0)
        end
        # Numerical approximation of expected lifetime utility
        V0 = zeros(N_sim)
        w0 = ψ[1]*(Y_0 - κ/q(θ_0)) # wage at t_0
        # compute LHS of IR constraint
        @inbounds for n = 1:N_sim
            t1 = 0
            @views @inbounds for t = 1:T
                t1   += 0.5*(ψ[t]*hp(AA[n,t])*σ_η)^2 # utility from consumption
                t2    = h(AA[n,t]) # disutility of effort
                # continuation value upon separation
                if procyclical == false
                    t3    = (t == T) ? ω*β : ω*β*s 
                elseif procyclical == true
                    t3    = (t == T) ? β*dot(P_z[IZ[n,t],:],ω) : β*s*dot(P_z[IZ[n,t],:],ω) 
                end                    
                V0[n] += (log(w0) - t1 - t2 + t3)*(β*(1-s))^(t-1) # LHS at time t 
            end
        end

        # check IR constraint (must bind)
        IR     = mean(V0)
        err1   = abs(IR - ω0)

        if IR > ω0
            θ_lb  = copy(θ_0)
        elseif IR < ω0
            θ_ub  = copy(θ_0)
        end

        if noisy 
            println(θ_0)
        end
        θ_0 = (θ_lb + θ_ub)/2
        iter1 += 1
    end

    return (θ = θ_0, Y = Y_0, V = IR, ω0 = ω0, w0 = w0, mod = m,
    ZZ = ZZ, IZ = IZ, az = az, yz = yz, 
    err1 = err1, err2 = err2, iter1 = iter1, iter2 = iter2, 
    effort_flag = flag, exit_flag1 = (iter1 >= max_iter1), exit_flag2 = (iter2 >= max_iter2))
end