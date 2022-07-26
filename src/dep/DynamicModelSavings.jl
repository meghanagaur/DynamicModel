"""
Solve the model WITH savings using a bisection search on θ.
"""
function solveModelSavings(m; max_iter1 = 50, max_iter2 = 500, tol1 = 10^-6, tol2 = 10^-8, noisy = true)

    @unpack T, β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, savings, procyclical, bgrid, b0 = m   
    if (~savings) error("Use solveModelSavings") end 

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
    YY             = zeros(size(ZZ,2))
    AZ             = zeros(size(ZZ))

    # reduce computation time of expectations by computing values only for unique z_t paths 
    zz    = unique(ZZ, dims=2)'
    iz    = unique(IZ, dims=2)'
    az    = zeros(size(zz))
    yy    = zeros(size(zz))
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
            w0 = ψ[1]*(Y_0 - κ/q(θ_0)) 
            @inbounds for t = 1:T
                zt          = unique(zz[:, t])  
                ψ_t         = ψ[t]
                @inbounds for n = 1:length(zt)
                    z = zt[n]
                    if ε == 1 # can solve analytically
                        aa = (z/w0 + sqrt((z/w0)^2))/2(1 + ψ_t*σ_η^2)
                    else # exclude the choice of zero effort
                        aa = find_zeros(x -> x - max(z*x/w0 -  (ψ_t/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0, 10) 
                    end
                    # create a flag matrix -- if neccessary, will need to handle violations
                    idx1           = findall(isequal(z), zz[:, t])
                    az[idx1,t]    .= isempty(aa) ? 0 : aa[1]
                    flag[idx1,t]  .= ((z*aa[1]/w0 + (ψ_t/ε)*(hp(aa[1])*σ_η)^2) < 0) + isempty(aa) + (w0 < 0)
                    yy[idx1,t]    .= ((β*(1-s))^(t-1))*az[n,t]*z
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

            #= if Y_1 < Y_0 
                Y_ub  = copy(Y_0)
            elseif Y_1 > Y_0 || w0 < 0
                Y_lb  = copy(Y_0)
            end
            Y_0  = 0.5(Y_lb + Y_ub) =#
            Y_0  = α*Y_0 + (1-α)*Y_1 
            #println(Y_0)
            iter2 += 1
        end

        # Numerical approximation of expected lifetime utility
        V0 = zeros(size(ZZ,2))
        v0 = zeros(size(zz,1))
        b  = zeros(size(az,1),T+1)
        b[:,1] .= b0
        w0 = ψ[1]*(Y_0 - κ/q(θ_0)) # wage at t_0

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
            ω0 = ω[iz[1,1]]
        elseif procyclical == false
            ω0 = ω
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

    return (θ = θ_0, Y = Y_0, V = IR, ω0 = ω0, w0 = w0, mod = m,
    idx = idx, zz = zz, ZZ = ZZ, az = az, AZ = AZ, flags = flag, 
    err1 = err1, err2 = err2, iter1 = iter1, iter2 = iter2) 
end

"""
Simulate wage paths given simulated z, a(z) paths
WITH savings.
"""
function simulateWagesSavings(model, w0, AZ, ZZ; seed = 145)
    #Random.seed!(seed)
    @unpack β,s,ψ,T,zgrid,P_z,ρ,σ_ϵ,hp,σ_η  = model
    N        = size(AZ,2)

    lw       = zeros(N,T+1)    # log wages
    lw[:,1] .= log(w0)         # initial wages
  
    @inbounds for  t=2:T+1, n=1:size(AZ,2)
       lw[n,t] = lw[n,t-1] + ψ[t]*hp(AZ[t,n])*rand(Normal(0,σ_η)) + 0.5(ψ[t]*hp(AZ[t,n])*σ_η)^2
    end
    ww = exp.(lw)
    return ww[2:end]
end

"""
Solve for the (infinite horizon) value of unemployment, given analytically
initial asset position and a procyclical unemployment benefit
via value function iteration.
"""
function unemploymentValueSavings(β, r, ξ, u, zgrid, P_z; N_b = 250, tol = 10^-6, max_iter = 2000)
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
