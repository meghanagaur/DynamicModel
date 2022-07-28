"""
Solve the model WITH savings using a bisection search on θ.
"""
function solveModelSavings(m; max_iter1 = 100, max_iter2 = 500, tol1 = 10^-8, tol2 = 10^-8, noisy = true)

    @unpack T, β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, savings, procyclical, bgrid, b0 = m   
    if (~savings) error("Use solveModel to solve model without savings") end 
    if (T!=2) error("set T=2 to solve model with savings") end 

    if procyclical
        nodes  = (zgrid, bgrid)
        ω_bz    = extrapolate(interpolate(nodes, ω, Gridded(Linear())),Line())
    end

    # set tolerance parameters for inner and outer loops
    err1  = 10
    err2  = 10
    iter1 = 1
    iter2 = 1

    # Initialize default values and search parameters
    θ_lb  = 0.0             # lower search bound
    θ_ub  = 2.0             # upper search bound
    θ_0   = (θ_lb + θ_ub)/2 # initial guess for θ
    α     = 0.25            # dampening parameter
    Y_0   = 0               # initalize Y_0 for export
    IR    = 0               # initalize IR for export
    w0    = 0               # initialize expected wage
    ω0    = 0               # initilize PV of unemp 

    #  simulate productivity paths
    ZZ, probs, IZ  = simulateProd(P_z, zgrid, T)
    AZ             = zeros(size(ZZ))
    YY             = zeros(size(ZZ,2))
    L              = zeros(size(ZZ,2))
    Λ              = zeros(size(ZZ))

    # reduce computation time of expectations by computing values only for unique z_t paths 
    zz    = unique(ZZ, dims=2)'
    iz    = unique(IZ, dims=2)'
    az    = zeros(size(zz))
    yy    = zeros(size(zz))
    λ     = zeros(size(zz))

    flag  = zeros(Int64, size(zz))
    idx   =  Dict{Int64, Vector{Int64}}()
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
            term1 = ψ[1]*(Y_0 - κ/q(θ_0)) 
            @inbounds for n = 1:size(zz,1)
                term2 = 0
                @inbounds for t = 1:T
                    z     = zz[n,t]
                    ψ_t   = ψ[t]
                    if t==1 
                        aa          = find_zeros(x -> x - max(z*x/term1 - (ψ_t/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0, 10) 
                        #aa          = find_zeros(x -> x - max(z*x/term1 - (2/ε)*(ψ_t*hp(x)*σ_η)^2 + (ψ_t/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0, 10) 

                        flag[n,t]   = ~isempty(aa) ? ((z*aa[1]/term1 + (ψ_t/ε)*(hp(aa[1])*σ_η)^2) < 0) : isempty(aa) 

                    elseif t == 2 # exclude the choice of zero effort
                        aa          = find_zeros(x -> x - max(z*x/term1 + (ψ_t/ε)*(hp(x)*σ_η)^2  - 
                                    ((2/ε)*(exp((ψ_t*hp(x)*σ_η)^2))*(ψ_t*hp(x)*σ_η)^2)/ ( exp((ψ[1]*hp(az[n,t-1])*σ_η)^2) 
                                    + exp((ψ_t*hp(x)*σ_η)^2)), 0)^(ε/(1+ε))+ Inf*(x==0), 0, 10)  

                        flag[n,t]   = ~isempty(aa) ? (z*aa[1]/term1 + (ψ_t/ε)*(hp(aa[1])*σ_η)^2  - 
                                    ((2/ε)*(exp((ψ_t*hp(aa[1])*σ_η)^2))*(ψ_t*hp(aa[1])*σ_η)^2)/ ( exp((ψ[1]*hp(az[n,t-1])*σ_η)^2) +
                                    exp((ψ_t*hp(aa[1])*σ_η)^2)) <0) : isempty(aa) 
                    end
                    flag[n,t] += (term1 < 0)
                    az[n,t]    = isempty(aa) ? 0 : aa[1]
                    term2     += (ψ_t*hp(az[n,t])*σ_η)^2
                    λ[n,t]     = ((β*(1-s))^(t-1))*exp(term2)
                    yy[n,t]    = ((β*(1-s))^(t-1))*az[n,t]*z
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

            Y_0  = α*Y_0 + (1-α)*Y_1 
            #println(Y_0)
            iter2 += 1
        end

        # Numerical approximation of expected lifetime utility
        l = sum(λ,dims=2)
        @inbounds for n = 1:size(zz,1)
            L[idx[n]]    .= l[n]
        end   
        Λ       = mean(L)
        V0      = zeros(size(ZZ,2))
        v0      = zeros(size(zz,1))
        b       = zeros(size(az,1),T+1)
        b[:,1] .= b0           
        w0      = ψ[1]*(Y_0 - κ/q(θ_0))/Λ 

        # compute LHS of IR constraint
        @inbounds for n = 1:size(zz,1)
            t1 = 0
            @inbounds for t = 1:T
                b[n,t+1] = (1+r)*b[n,t]                 # c_t = w_t, so invest all of assets
                t1      += 0.5*(ψ[t]*hp(az[n,t])*σ_η)^2 # utility from consumption = log wages
                t2      = h(az[n,t])                    # disutility of effort
                # continuation value upon separation
                if procyclical == false
                    t3    = (t == T) ? ω(b[n,t+1])*β : ω(b[n,t+1])*β*s 
                elseif procyclical == true
                    t3    = (t == T) ? β*dot(P_z[iz[n,t],:], ω_bz.(zgrid, b[n,t+1])) : ω_bz(zz[n,t+1], b[n,t+1])*β*s
                end                    
                v0[n] += (log(w0) + t1 - t2 + t3)*(β*(1-s))^(t-1)
            end
            V0[idx[n]] .= v0[n]
        end

        if procyclical == true
            ω0 = ω_bz(zz[1,1],b0)
        elseif procyclical == false
            ω0 = ω(b0)
        end

        # check IR constraint
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

    return (θ = θ_0, Y = Y_0, V = IR, ω0 = ω0, w0 = w0, mod = m, Λ = Λ,
    idx = idx, zz = zz, ZZ = ZZ, az = az, AZ = AZ, flags = flag, 
    err1 = err1, err2 = err2, iter1 = iter1, iter2 = iter2) 
end

"""
Solve for the (infinite horizon) value of unemployment, given analytically
initial asset position and a procyclical unemployment benefit
via value function iteration.
"""
function unemploymentValueSavings(β, r, ξ, u, zgrid, P_z; N_b = 250, tol = 10^-7, max_iter = 2000)
    bmin  = -10
    bmax  = 50
    bgrid = LinRange(bmin, bmax, N_b)
    N_z   = length(zgrid)

    val  = zeros(N_z, N_b, N_b)
    @inbounds for ib2 = 1:N_b
        @inbounds for ib1 = 1:N_b
            @inbounds for iz1 = 1:N_z
                c = ξ(zgrid[iz1]) + (1+r)*bgrid[ib1] - bgrid[ib2] 
                val[iz1,ib1,ib2] = u(c)*(c>0) - Inf*(c<=0)  # c>0 w/ log utility & NBL
            end
        end
    end

    v0_new = zeros(N_z, N_b)
    v0     = zeros(N_z, N_b)
    @inbounds for iz = 1:length(zgrid)
        v0[iz,:] = diag(val[iz,:,:])
    end

    bb   = zeros(Int64, N_z, N_b)
    iter = 1
    err  = 10

    # solve via simple value function iteration
    @inbounds while err > tol && iter < max_iter
        @inbounds for ib1 = 1:N_b
            @inbounds for (iz,z) in enumerate(zgrid)
                 V = val[iz,ib1,:] .+ β*vec(P_z[iz,:]'*v0)
                 v0_new[iz, ib1], bb[iz,ib1] = findmax(V)
            end
        end
         err = maximum(abs.(v0_new - v0))
         v0  = copy(v0_new)
        iter +=1
    end

    return (v0 = v0, bgrid = collect(bgrid), bb = bb, err = err, iter = iter) 
end
