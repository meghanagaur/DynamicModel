using DataStructures, Distributions, ForwardDiff, Interpolations,
 LinearAlgebra, Parameters, Random, Roots, StatsBase

"""
Setup dynamic EGSS model, where m(u,v) = (uv)/(u^ι + v^⟦)^(1/ι),
η ∼ N(0, σ_η^2), log(z_t) = μ_z + ρ*log(z_t-1) + u_t, u_t ∼ N(0, σ_z^2),
and y_t = z_t(a_t + η_t).
β    = discount factor
r    = interest rate
s    = exogenous separation rate
ι    = matching elasticity
κ    = vacancy-posting cost
T    = maximal contract duration 
ω    = worker's PV from unemployment (infinite horizon)
χ    = prop. of unemp benefit to z / actual unemp benefit
γ    = intercept for unemp benefit w/ procyclical benefit
z_ss = steady state productivity
σ_η  = st dev of η distribution
μ_z  = unconditional mean of log prod.
z0   = initial (log) prod.
ρ    = persistence of log prod. process
σ_ϵ  = variance of error in log prod. process
ε    = Frisch elasticity: disutility of effort
ψ    = pass-through parameters
b0   = initial assets

savings     == (EGSS with savings)
procyclical == (EGSS with procyclical unemployment benefit)
"""
function model(; T = 20, β = 0.96, s = 0.1, κ = 0.213, ι = 1.27, ε = 0.5, σ_η = 0.05, z_ss = 1,
    ρ = 0.92, σ_ϵ = 0.01, χ = 0.1, γ = 0.63, z0 = 0.0, μ_z = z0, N_z = 11, savings = false,
    procyclical = true, b0 = 0)

    q(θ)    = 1/(1 + θ^ι)^(1/ι)                     # vacancy-filling rate
    f(θ)    = 1/(1 + θ^-ι)^(1/ι)                    # job-filling rate
    u(c)    = log(max(c, eps()))                    # utility from consumption                
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
        ξ(z) = γ + χ*(z - z_ss) 
    elseif procyclical == false
        ξ    = γ
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
    
    return (T = T, β = β, r = r, s = s, κ = κ, ι = ι, ε = ε, σ_η = σ_η, ρ = ρ, σ_ϵ = σ_ϵ, 
    ω = ω, μ_z = μ_z, N_z = N_z, q = q, f = f, ψ = ψ, z0 = z0, h = h, u = u, hp = hp, 
    zgrid = zgrid, P_z = P_z, savings = savings, bgrid = bgrid, b0 = b0, ξ = ξ, χ = χ,
    γ = γ, procyclical = procyclical)
end

"""
Solve for the optimal effort a(z,t), given Y_0, θ_0, zt, and t.
"""
function optA(z, t, modd, Y, θ)
    @unpack ψ, ε, q, κ, hp, σ_η = modd
    ψ_t = ψ[t]
    w0  = ψ[1]*(Y - κ/q(θ)) # time-0 earnings (constant)
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

"""
Solve the model WITHOUT savings using a bisection search on θ.
"""
function solveModel(m; N_sim = 10000, max_iter1 = 100, max_iter2 = 100, tol1 = 10^-7, tol2 = 10^-8, noisy = true)
    @unpack T, β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, savings, procyclical, N_z = m   
    if (savings) error("Use solveModelSavings") end 

    # set tolerance parameters for outermost loops
    err1  = 10
    iter1 = 1

    # Initialize default values and search parameters
    z0_idx = findfirst(isequal(z0), log.(zgrid)) # index of z0 on zgrid
    ω_0    = procyclical ? ω[z0_idx] : ω         # unemployment value at z0
    θ_lb   = 0.0             # lower search bound
    θ_ub   = 10.0            # upper search bound
    θ_0    = (θ_lb + θ_ub)/2 # initial guess for θ
    α      = 0.25            # dampening parameter
    Y_0    = 0               # initalize Y_0 for export
    IR     = 0               # initalize IR for export
    w0     = 0               # initialize initial wage constant for export

    #  simulate productivity paths for computing expectations
    ZZ, probs, IZ  = simulateProd(P_z, zgrid, T; N = N_sim) # N x T
    az             = zeros(N_z, T)    # N_z x T                         
    yz             = zeros(N_z, T)    # N_z x T                        
    flag           = zeros(N_z, T)    # N_z x T                          
    AA             = zeros(N_sim, T)  # N x T                   
    YY             = zeros(N_sim, T)  # N x T                   

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 < max_iter1
        err2   = 10
        iter2  = 1
        Y_lb   = κ/q(θ_0)         # lower search bound
        Y_ub   = 100*κ/q(θ_0)      # upper search bound
        Y_0    = (Y_lb + Y_ub)/2  # initial guess for Y
        # Look for a fixed point in Y_0
        @inbounds while err2 > tol2 && iter2 < max_iter2
            # Solve for optimal effort a_t, and implied (expected) y_t
            @inbounds for t = 1:T
                @inbounds for (iz,z) in enumerate(zgrid)
                    az[iz,t], yz[iz,t], flag[iz,t] = optA(z, t, m, Y_0, θ_0)
                end 
            end
            @views @inbounds for t=1:T
                AA[:,t] .= az[IZ[:,t], t] 
                YY[:,t] .= yz[IZ[:,t], t] 
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
            # increase dampening parameter if not converging
            α      = iter2 > 50 ? 0.75 : α 
            Y_0    = max(α*Y_0 + (1-α)*Y_1, κ/q(θ_0))
            iter2 += 1
            #println(Y_0)
        end
        # Numerical approximation of expected lifetime utility
        V0 = zeros(N_sim)
        w0 = ψ[1]*(Y_0 - κ/q(θ_0)) # wage at t_0
        # compute LHS of IR constraint
        Threads.@threads for n = 1:N_sim
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
        err1   = abs(IR - ω_0)

        if IR > ω_0
            θ_lb  = copy(θ_0)
        elseif IR < ω_0
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

"""
NEEDS UPDATING.
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
