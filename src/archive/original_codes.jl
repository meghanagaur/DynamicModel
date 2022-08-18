"""
Solve the model WITHOUT savings using a bisection search on θ.
"""
function solveModel2(m; max_iter1 = 25, max_iter2 = 100, tol1 = 10^-7, tol2 = 10^-8, noisy = true)

    @unpack T, β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, savings, procyclical = m   
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
    ZZ, probs, IZ  = simulateProd2(P_z, zgrid, T) # T X N
    YY             = zeros(size(ZZ,2))           # T X N
    AZ             = zeros(size(ZZ))             # T X N

    # reduce computation time by considering only for unique z_t paths 
    zz    = unique(ZZ, dims=2)'     # n X T
    iz    = unique(IZ, dims=2)'     # n x T
    az    = zeros(size(zz))         # n x T
    yy    = zeros(size(zz))         # n x T
    flag  = zeros(Int64, size(zz))  # n x T
    idx   = Dict{Int64, Vector{Int64}}()
    ω0    = procyclical ? ω[iz[1,1]] : ω # initialize unemployment value at z0

    @inbounds for n = 1:size(zz,1)
        push!(idx, n => findall(isequal(T), vec(sum(ZZ .== zz[n,:], dims=1))) )
    end   

    # look for a fixed point in θ
    @inbounds while err1 > tol1 && iter1 < max_iter1
        err2   = 10
        iter2  = 1

        Y_lb = κ/q(θ_0)         # lower search bound
        Y_ub = 50*κ/q(θ_0)      # upper search bound
        Y_0  = (Y_lb + Y_ub)/2  # initial guess for Y

        # look for a fixed point in Y0
        @inbounds while err2 > tol2 && iter2 < max_iter2
            w0 = ψ[1]*(Y_0 - κ/q(θ_0)) 
            @inbounds for t = 1:T
                #zt          = unique(zz[:, t])  
                ψ_t         = ψ[t]
                @inbounds for (iz,z) in enumerate(unique(zz[:, t]))   #n = 1:length(zt)
                    #z = zt[n]
                    if ε == 1 # can solve analytically
                        aa = (z/w0 + sqrt((z/w0)^2))/2(1 + ψ_t*σ_η^2)
                    else # exclude the choice of zero effort
                        aa = find_zeros(x -> x - max(z*x/w0 -  (ψ_t/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0, 10) 
                    end
                    # create a flag matrix -- if neccessary, will need to handle violations
                    idx1           = findall(isequal(z), zz[:, t])
                    az[idx1,t]    .= isempty(aa) ? 0 : aa[1]
                    flag[idx1,t]  .= ~isempty(aa) ? ((z*aa[1]/w0 + (ψ_t/ε)*(hp(aa[1])*σ_η)^2) < 0) : isempty(aa) 
                    #flag[idx1,t]  .+= (w0 < 0)
                    yy[idx1,t]    .= ((β*(1-s))^(t-1))*az[idx1,t]*z
                end  
            end
            # Expand
            y = sum(yy,dims=2)
            @inbounds for n = 1:size(zz,1)
                YY[idx[n]]    .= y[n]
            end   
            # Numerical approximation of E_0[Y]
            Y_1  = mean(YY)
            err2 = abs(Y_0 - Y_1)
            #= if doing bisection search on Y_0 
            if Y_1 < Y_0 
                Y_ub  = copy(Y_0)
            elseif Y_1 > Y_0 || w0 < 0
                Y_lb  = copy(Y_0)
            end
            Y_0  = 0.5(Y_lb + Y_ub) 
            # Note: delivers ≈ Y_0, but converges more slowly. =#
            α = iter2 > 50 ? 0.75 : α
            Y_0  = max(α*Y_0 + (1-α)*Y_1, κ/q(θ_0))
            #println(Y_0)
            iter2 += 1
        end

        # Numerical approximation of expected lifetime utility
        V0 = zeros(size(ZZ,2))
        v0 = zeros(size(zz,1))
        w0 = ψ[1]*(Y_0 - κ/q(θ_0)) # wage at t_0

        # compute LHS of IR constraint
        @inbounds for n = 1:size(zz,1)
            t1 = 0
            @inbounds for t = 1:T
                t1   += 0.5*(ψ[t]*hp(az[n,t])*σ_η)^2 # utility from consumption
                t2    = h(az[n,t]) # disutility of effort
                # continuation value upon separation
                if procyclical == false
                    t3    = (t == T) ? ω*β : ω*β*s 
                elseif procyclical == true
                    t3    = (t == T) ? β*dot(P_z[iz[n,t],:],ω) : β*s*dot(P_z[iz[n,t],:],ω) #ω[iz[n,t+1]]*β*s <- could directly use next period's draw
                end                    
                v0[n] += (log(w0) - t1 - t2 + t3)*(β*(1-s))^(t-1) # LHS at time t 
            end
            V0[idx[n]] .= v0[n]
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

    @inbounds for n = 1:size(zz,1)
        AZ[:, idx[n]] .= az[n,:]
    end

    return (θ = θ_0, Y = Y_0, V = IR, ω0 = ω0, w0 = w0, mod = m,
    idx = idx, zz = zz, ZZ = ZZ, iz = iz, IZ = IZ, az = az, AZ = AZ, 
    err1 = err1, err2 = err2, iter1 = iter1, iter2 = iter2, 
    effort_flag = flag, exit_flag1 = (iter1 >= max_iter1), exit_flag2 = (iter2 >= max_iter2))
end

"""
Simulate N {z_t} paths, given transition prob. matrix P_z and 
productivity grid, and return probability of each path.
"""
function simulateProd2(P_z, zgrid, T; N = 10000, seed = 211, set_seed = true, z0 = median(1:length(zgrid)))
    if set_seed == true
        Random.seed!(seed)
    end
    sim      = rand(T, N)            # T X N - draw uniform numbers in (0,1)
    zt       = zeros(Int64, T, N)    # T x N - index on productivity grid
    zt[1,:] .= floor(Int64, z0)      # T x N - set z0 with valid index
    CDF      = cumsum(P_z, dims = 2) # CDF for transition probs. given initial state
    probs    = ones(N)               # N x 1 - probability of a given sequence

    @inbounds for i = 1:N
        @inbounds for t = 2:T
            zt[t, i]  = findfirst(x-> x >=  sim[t,i], CDF[zt[t-1,i],:]) 
            probs[i]  =  P_z[zt[t-1,i], zt[t,i]]*probs[i]
        end
    end
    return zgrid[zt], probs, zt
end