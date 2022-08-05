#= Solve the dynamic EGSS model (first pass). =#
module DynamicModel # begin module

export model, solveModel, simulateProd, simulateWages, unemploymentValue,
solveModelSavings, simulateWagesSavings,unemploymentValueSavings, rouwenhorst

using DataStructures, Distributions, ForwardDiff, Interpolations,
 LinearAlgebra, Parameters, Random, Roots, StatsBase

include("dep/rouwenhorst.jl")
include("dep/DynamicModelSavings.jl")

"""
Simulate N {z_t} paths, given transition probs and 
productivity grid, and return probability of each path.
"""
function simulateProd(P_z, zgrid, TT; N = 30000, seed = 211, z0 = median(1:length(zgrid)))
    Random.seed!(seed)
    sim      = rand(TT, N)           # T X N - draw uniform numbers in (0,1)
    zt       = zeros(Int64, TT, N)   # T x N - index on productivity grid
    zt[1,:] .= floor(Int64, z0)      # T x N
    CDF      = cumsum(P_z, dims = 2)
    probs    = ones(N)               # N x 1 - probability of a given sequence

    @inbounds for i = 1:N
        @inbounds for t = 2:TT
            zt[t, i]  = findfirst(x-> x >=  sim[t,i], CDF[zt[t-1,i],:]) 
            probs[i]  =  P_z[zt[t-1,i], zt[t,i]]*probs[i]
        end
    end
    return zgrid[zt], probs, zt
end

"""
Setup dynamic EGSS model, where m(u,v) = (uv)/(u^ι + v^⟦)^(1/ι),
η ∼ N(0, σ_η^2), log(z_t) = ρ*log(z_t-1) + u_t, u_t ∼ N(0, σ_z^2),
y = z(a + η).
β   = discount factor
r   = interest rate
s   = exogenous separation rate
ι   = matching elasticity
κ   = vacancy-posting cost
T   = final period 
ω   = worker's PV from unemployment 
χ   = prop. of unemp benefit to z / actual unemp benefit
γ   = intercept term of unemp benefit w/ procylical benefit
σ_η = st dev of η distribution
μ_z = unconditional mean of log productivity
z0  = initial log productivity
ρ   = persistence of AR1 logz process
σ_ϵ = variance of error in logz process
ε   = Frisch elasticity, disutility of effort
ψ   = pass-through parameters
b0  = initial assets

savings     == (EGSS with savings)
procyclical == (EGSS with procyclical unemployment benefit)
"""
function model(; T = 20, β = 0.99, s = 0.2, κ = 0.213, ι = 1.27, ε = 0.5, σ_η = 0.05, 
    ρ = 0.999, σ_ϵ = 0.01, γ = 0, χ = 0.66, z0 = 0.0, μ_z = z0, N_z = 17, savings = false,
    procyclical = false, b0 = 0)

    q(θ)    = 1/(1 + θ^ι)^(1/ι)                     # vacancy-filling rate
    u(c)    = log(max(c,eps()))                     # utility from consumption                
    h(a)    = (a^(1 + 1/ε))/(1 + 1/ε)               # disutility from effort  
    u(c, a) = u(c) - h(a)                           # utility function
    hp(a)   = a^(1/ε)                               # h'(a)
    r       = 1/β -1                                # interest rate        

    if (iseven(N_z)) error("N_z must be odd") end 
    logz, P_z = rouwenhorst(μ_z, ρ, σ_ϵ, N_z)       # discretized logz grid & transition probabilties
    zgrid     = exp.(logz)                          # actual productivity grid

    # pass-through parameters
    ψ    = Dict{Int64, Float64}()
    temp = reverse(1 ./[mapreduce(t-> (β*(1-s))^t, +, [0:xx;]) for xx = 0:T])
    for t = 0:T
        ψ[t] = temp[t+1]
    end

    if ((T!=2) & (savings == true)) error("set T=2 for model with savings") end 
    # unemployment benefit given current state: (z) or (z,b)
    if procyclical == true
        ξ(z) = γ + χ*z 
    elseif procyclical == false
        ξ    = χ
    end

    # PV of unemp if you receive unemployment benefit forever
    bgrid = NaN
    if savings == false
        if procyclical == false
            ω = log(ξ)/(1-β) # scalar
        elseif procyclical == true
            println("Solving for value of unemployment...")
            ω = unemploymentValue(β, ξ, u, zgrid, P_z).v0 # N_z x 1
        end
    elseif savings == true
        if procyclical == false
            ω(b)   = u(ξ + r*b)/(1-β) # scalar function; b = initial asset position
        elseif procyclical == true
            println("Solving for value of unemployment...")
            modd   = unemploymentValueSavings(β, r, ξ, u, zgrid, P_z)
            bgrid  = modd.bgrid
            ω      = modd.v0 # N_z x N_b
        end
    end
    
    return (T = T, β = β, r = r, s = s, κ = κ, ι = ι, ε = ε, σ_η = σ_η, ρ = ρ,
    σ_ϵ = σ_ϵ, ω = ω, μ_z = μ_z, N_z = N_z, q = q, ψ = ψ, z0 = z0, bgrid = bgrid,
    h = h, u = u, hp = hp, zgrid = zgrid, P_z = P_z, savings = savings, b0 = b0, ξ = ξ,
    procyclical = procyclical)
end

"""
Solve the model WITHOUT savings using a bisection search on θ.
"""
function solveModel(m; max_iter1 = 50, max_iter2 = 500, tol1 = 10^-8, tol2 = 10^-8, noisy = true)

    @unpack T, β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, savings, procyclical = m   
    if (savings) error("Use solveModelSavings") end 

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
    w0    = 0               # initialize initial wage constant for export
    ω0    = 0               # initilize PV of unemp for export

    #  simulate productivity paths
    ZZ, probs, IZ  = simulateProd(P_z, zgrid, T) # T X N
    YY             = zeros(size(ZZ,2))           # T X N
    AZ             = zeros(size(ZZ))             # T X N

    # reduce computation time of expectations by computing values only for unique z_t paths 
    zz    = unique(ZZ, dims=2)'     # n X T
    iz    = unique(IZ, dims=2)'     # n x T
    az    = zeros(size(zz))         # n x T
    yy    = zeros(size(zz))         # n x T
    flag  = zeros(Int64, size(zz))  # n x T
    idx   =  Dict{Int64, Vector{Int64}}()
    @inbounds for n = 1:size(zz,1)
        push!(idx, n => findall(isequal(T), vec(sum(ZZ .== zz[n,:], dims=1))) )
    end   

    # look for a fixed point in θ
    @inbounds while err1 > tol1 && iter1 < max_iter1
        err2   = 10
        iter2  = 1

        Y_lb = κ/q(θ_0)         # lower search bound
        Y_ub = 100*κ/q(θ_0)     # upper search bound
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
                    flag[idx1,t]  .+= (w0 < 0)
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
                    t3    = (t == T) ? β*dot(P_z[iz[n,t],:],ω) : β*s*dot(P_z[iz[n,t],:],ω) #ω[iz[n,t+1]]*β*s
                end                    
                v0[n] += (log(w0) - t1 - t2 + t3)*(β*(1-s))^(t-1) # LHS at time t 
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
Simulate wage paths given already simulated {z, a(z)} paths 
(i.e. simulate η shocks) WITH or WITHOUT savings.
"""
function simulateWages(sol; savings = false, seed = 123)
    Random.seed!(seed)

    @unpack AZ, ZZ, mod, w0 = sol
    @unpack β,s,ψ,T,zgrid,P_z,ρ,σ_ϵ,hp,σ_η  = mod

    N        = size(AZ,2)    # number of simulations
    lw       = zeros(N,T+1)  # log wages
    lw[:,1] .= log(w0)       # earnings @ t=0 (constant)
  
    @inbounds for t=2:T+1, n=1:N
        if savings
            lw[n,t] = lw[n,t-1] + ψ[t-1]*hp(AZ[t-1,n])*rand(Normal(0,σ_η)) + 0.5(ψ[t-1]*hp(AZ[t-1,n])*σ_η)^2
         else
            lw[n,t] = lw[n,t-1] + ψ[t-1]*hp(AZ[t-1,n])*rand(Normal(0,σ_η)) - 0.5(ψ[t-1]*hp(AZ[t-1,n])*σ_η)^2
        end
    end
    return exp.(lw[:,2:end])
end

"""
Solve for the (infinite horizon) value of unemployment, 
WITHOUT savings and a procyclical unemployment benefit 
via value function iteration.
"""
function unemploymentValue(β, ξ, u, zgrid, P_z; tol = 10^-8, max_iter = 5000)
    N_z    = length(zgrid)
    v0_new = zeros(N_z)
    v0     = u.(ξ.(zgrid))
    iter   = 1
    err    = 10

    # solve via simple value function iteration
    @inbounds while err > tol && iter < max_iter
        @inbounds for (iz,z) in enumerate(zgrid)
            v0_new[iz] = u(ξ.(z)) + β*dot(P_z[iz,:],v0)
        end
        err = maximum(abs.(v0_new - v0))
        v0  = copy(v0_new)
        iter +=1
    end
    return (v0 = v0, err = err, iter = iter) 
end

end # module
