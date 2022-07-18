module DynamicModel

#= Solve the dynamic EGSS model (first pass).
Focus on the case without savings and where the
the unemployment benefit is constant. =#

export model, solveModel, simulateProd, simulateWages

using DataStructures, Distributions, ForwardDiff, Interpolations,
 LinearAlgebra, Parameters, Random, Roots, StatsBase

include("dep/rouwenhorst.jl")

"""
Simulate N {z_t} paths, given transition probs and 
productivity grid, and return probability of each path.
"""
function simulateProd(P_z, zgrid, TT; N = 25000, seed = 111, z0 = median(1:length(zgrid)))
    Random.seed!(seed)
    sim      = rand(TT, N)  # draw uniform numbers in (0,1)
    zt       = zeros(Int64, TT, N)
    zt[1,:] .= floor(Int64, z0)
    CDF      = cumsum(P_z, dims = 2)
    probs    = ones(N) # probably of a given sequence

    @inbounds for i = 1:N
        @inbounds for t = 2:TT
            zt[t, i]  = findfirst(x-> x >=  sim[t,i], CDF[zt[t-1,i],:]) 
            probs[i]  =  P_z[zt[t-1], zt[t]]*probs[i]
        end
    end
    return zgrid[zt], probs
end

"""
Setup dynamic EGSS model, where m(u,v) = (uv)/(u^ι + v^⟦)^(1/ι),
η ∼ N(0, σ_η^2), log(z_t) = ρ*log(z_t-1) + u_t, u_t ∼ N(0, σ_z^2),
y = z(a + η).
β   = discount factor
s   = exogenous separation rate
ι   = matching elasticity
κ   = vacancy-posting cost
T   = final period (including t = 0)
ω   = worker's present value from unemployment at time t
b   = unemployment benefit (flow)
σ_η = st dev of η distribution
μ_z = unconditional mean of log productivity
z0  = initial log productivity
ρ   = persistence of AR1 logz process
σ_ϵ = variance of error in logz process
ε   = Frisch elasticity, disutility of effort
ψ   = pass-through parameters
"""
function model(; T = 20, β = 0.99, s = 0.2, κ = 0.213, ι = 1.27, ε = 0.5, σ_η = 0.05, 
    ρ = 0.999, σ_ϵ = 0.01, b = 0.68, z0 = 0.0, μ_z = z0, N_z = 11, savings = false)

    q(θ)    = 1/(1 + θ^ι)^(1/ι)                     # vacancy-filling rate
    v(c)    = log(c)                                # utility from consumption                
    h(a)    = (a^(1 + 1/ε))/(1 + 1/ε)               # disutility from effort  
    u(c, a) = u(c) - h(a)                           # utility function
    hp(a)   = a^(1/ε)                               # h'(a)

    if (iseven(N_z)) error("N_z must be odd") end 

    logz, P_z = rouwenhorst(μ_z, ρ, σ_ϵ, N_z)       # discretized logz grid & transition probabilties
    zgrid     = exp.(logz)                          # actual productivity grid

    # pass-through parameters. Note: there is a 0th period
    ψ    = Dict{Int64, Float64}()
    temp = reverse(1 ./[mapreduce(t-> (β*(1-s))^t, +, [0:xx;]) for xx = 0:T])
    for t = 0:T
        ψ[t] = temp[t+1]
    end
    # PV of unemp if you consume unemployment benefit forever
    ω     = log(b)/(1-β)

    return (T = T, β = β, s = s, κ = κ, ι = ι, ε = ε, σ_η = σ_η, ρ = ρ,
    σ_ϵ = σ_ϵ, ω = ω, μ_z = μ_z, N_z = N_z, q = q, v = v, ψ = ψ, z0 = z0,
    h = h, u = u, hp = hp, zgrid = zgrid, P_z = P_z, savings = savings)
end

"""
Solve the model using a bisection search on θ 
"""
function solveModel(m; max_iter1 = 50, max_iter2 = 500, tol1 = 10^-6, tol2 = 10^-8, noisy = true)
    @unpack T, β, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, savings = m   
    
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

    #  simulate the productivity paths
    ZZ, probs   = simulateProd(P_z, zgrid, T+1)
    YY          = similar(ZZ, size(ZZ,2))
    AZ          = similar(ZZ, size(ZZ))

    # reduce computation time of expectations by computing values only for unique z_t paths 
    zz    = unique(ZZ, dims=2)
    az    = similar(zz')
    yy    = similar(zz')
    flag  = zeros(Int64, size(zz'))
    idx   =  Dict{Int64, Vector{Int64}}()
    @inbounds for n = 1:size(zz,2)
        push!(idx, n => findall(isequal(T+1), vec(sum(ZZ .== zz[:,n], dims=1))) )
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
            w0 = ψ[0]*(Y_0 - κ/q(θ_0)) # wages at t = 0

            @inbounds for t = 0:T
                zt          = unique(zz[t+1, :])  
                ψ_t         = ψ[t]
                @inbounds for n = 1:length(zt)
                    z = zt[n]
                    if ε == 1 # can solve analytically
                        aa = (z/w0 + sqrt((z/w0)^2))/2(1 + ψ_t*σ_η^2)
                    else # exclude the choice of zero effort
                        aa = find_zeros(x -> x - max(z*x/w0 -  (ψ_t/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0, 10) 
                    end

                    # create a flag matrix -- if neccessary, will need to handle violations
                    idx1             = findall(isequal(z), zz[t+1, :])
                    az[idx1,t+1]    .= isempty(aa) ? 0 : aa[1]
                    flag[idx1,t+1]  .= ((z*az[n, t+1]/w0 + (ψ_t/ε)*(hp(az[n,t+1])*σ_η)^2) < 0) + isempty(aa) + (w0 < 0)
                    yy[idx1,t+1]    .= ((β*(1-s))^(t))*az[n,t+1]*z
                end  
            end

            # Expand
            y = sum(yy,dims=2)
            @inbounds for n = 1:size(zz,2)
                YY[idx[n]]    .= y[n]
            end   

            # Numerical approximation of E_0[Y]
            Y_1  = mean(YY)
            err2 = abs(Y_0 - Y_1)

            #=
            if Y_1 < Y_0 
                Y_ub  = copy(Y_0)
            elseif Y_1 > Y_0 || w0 < 0
                Y_lb  = copy(Y_0)
            end
            Y_0  = 0.5(Y_lb + Y_ub) 
            =#
            
            Y_0  = α*Y_0 + (1-α)*Y_1 # faster convergence
            #println(Y_0)
            iter2 += 1
        end

        # Numerical approximation of expected lifetime utility
        V0 = zeros(size(ZZ,2))
        v0 = zeros(size(zz,2))

        if savings == false
            w0 = ψ[0]*(Y_0 - κ/q(θ_0)) # wages at t = 0, from martingale property (w/o savings)
        end

        # compute LHS of IR constraint
        @inbounds for n = 1:size(zz,2)
            t1 = 0
            @inbounds for t = 0:T
                t1    += (t == 0) ? 0 : 0.5*(ψ[t]*hp(az[n,t+1])*σ_η)^2 
                t2    = h(az[n,t+1])
                t3    = (t == T) ? ω*β : ω*β*s # continuation value upon separation
                if savings == false
                    v0[n] += (log(w0) - t1 - t2 + t3)*(β*(1-s))^t 
                else
                    v0[n] += (log(w0) + t1 - t2 + t3)*(β*(1-s))^t
                end
            end
            V0[idx[n]] .= v0[n]
        end

        # check IR constraint
        IR     = mean(V0)
        err1   = abs(IR - ω)

        if IR > ω
            θ_lb  = copy(θ_0)
        elseif IR < ω
            θ_ub  = copy(θ_0)
        end

        if noisy 
            println(θ_0)
        end
        θ_0 = (θ_lb + θ_ub)/2
        iter1 += 1
    end

    @inbounds for n = 1:size(zz,2)
        AZ[:, idx[n]]   .= az[n,:]
    end

    return (θ = θ_0, Y = Y_0, V = IR, ω0 = ω, w0 = w0, mod = m,
    idx = idx, zz = zz, ZZ = ZZ, az = az, AZ = AZ, flags = flag, 
    err1 = err1, err2 = err2, iter1 = iter1, iter2 = iter2) 
end

"""
Simulate wage paths given simulated z, a(z) paths. 
"""
function simulateWages(model, w0, AZ, ZZ; seed = 145)
    #Random.seed!(seed)
    @unpack β,s,ψ,T,zgrid,P_z,ρ,σ_ϵ,hp,σ_η  = model

    lw       = zeros(size(AZ')) # log wages
    lw[:,1] .= log(w0)          # initial wages
  
    @inbounds for  t=2:T+1, n=1:size(AZ,2)
       lw[n,t] = lw[n,t-1] + ψ[t-1]*hp(AZ[t,n])*rand(Normal(0,σ_η)) - 0.5(ψ[t-1]*hp(AZ[t,n])*σ_η)^2
    end

    ww = exp.(lw)

    return ww
end

end # module
