module DynamicModel

# Solve the dynamic EGSS model (first pass)

export model, solveModel

using LinearAlgebra, Distributions, Random, Interpolations, ForwardDiff,
BenchmarkTools, Parameters, StatsBase, Roots, Distributed

include("dep/rouwenhorst.jl")

"""
Simulate N {z_t} paths, given transition probs and 
productivity grid, and return probability of each path.
"""
function simulate(P_z, zgrid, T; N = 10000, seed = 111)

    Random.seed!(seed)
    sim      = rand(T, N)  # draw uniform numbers in (0,1)
    zt       = zeros(Int64, T, N)
    zt[1,:] .= floor(Int64, median(1:length(zgrid)))
    CDF      = cumsum(P_z, dims = 2)
    probs    = ones(N)

    @inbounds for i = 1:N
        @inbounds for t = 2:T
            zt[t, i]  = findfirst(x-> x >= sim[t,i], CDF[zt[t-1,i],:] ) 
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
z0  = initial productivity
ρ   = persistence of AR1 logz process
σ_ϵ = variance of error in logz process
ε   = Frisch elasticity, disutility of effort
ψ   = pass-through parameters
"""
function model(; T = 5, β = 0.99, s = 0.2, κ = 0.213, ι = 1.25, ε = 0.5,
    σ_η = 0.1, ρ = 0.99, σ_ϵ = 0.1, b = 0.73, z0 = 1.0, N_z = 17, savings = false)

    q(θ)    = 1/(1 + θ^ι)^(1/ι)                     # vacancy-filling rate
    v(c)    = log(c)                                # utility from consumption                
    h(a)    = (a^(1 + 1/ε))/(1 + 1/ε)               # disutility from effort  
    u(c, a) = u(c) - h(a)                           # utility function
    hp(a)   = a^(1/ε)                               # h'(a)

    logz, P_z = rouwenhorst(log(z0), ρ, σ_ϵ, N_z)   # discretized logz grid & transition probabilties
    zgrid     = exp.(logz)                          # actual productivity grid

    # pass-through parameters. Note: there is a 0th period
    ψ = reverse(1 ./[mapreduce(t-> (β*(1-s))^t, +, [0:xx;]) for xx = 0:T])

    # value of unemployment independent of z_t for now
    ω(t) = log(b)*(1 - β^(T-t+1))/(1-β)
    
    return (T = T, β = β, s = s, κ = κ, ι = ι, ε = ε, σ_η = σ_η, ρ = ρ,
    σ_ϵ = σ_ϵ, ω = ω, z0 = z0, N_z = N_z, q = q, v = v, ψ = ψ,
    h = h, u = u, hp = hp, zgrid = zgrid, P_z = P_z, savings = savings)
end

"""
Solve the model using a bisection search on θ 
"""
function solveModel(m; max_iter1 = 25, max_iter2 = 200, tol1 = 10^-5, tol2 =  10^-6)
    
    @unpack T, β, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, savings = m   
    
    # set tolerance parameters for outer loop
    err1   = 10
    err2   = 10
    iter1  = 1
    iter2  = 1

    θ_lb = 0.0             # lower search bound
    θ_ub = 2.0             # upper search bound
    θ_0  = (θ_lb + θ_ub)/2 # initial guess for θ
    Y_0  = 1.0             # initalize Y_0 for export
    IR   = 1.0             # initalize IR for export

    #  simulate productivity draws
    ZZ, probs   = simulate(P_z, zgrid, T+1)
    AZ          = similar(ZZ, size(ZZ))
    YY          = similar(ZZ, size(ZZ,2))

    # reduce computation time by looking at unique draws
    zz    = unique(ZZ, dims=2)
    az    = similar(zz')
    yy    = similar(zz')
    flag  = zeros(Int64, size(zz'))

    # Dampening parameter
    α     = 0.25

    # look for a fixed point in θ
    @inbounds while err1 > tol1 && iter1 < max_iter1

        err2   = 10
        iter2  = 1

        Y_lb = κ/q(θ_0)         # lower search bound
        Y_ub = 100*κ/q(θ_0)     # upper search bound
        Y_0  = (Y_lb + Y_ub)/2  # initial guess for Y

        # look for a fixed point in Y0
        @inbounds while err2 > tol2 && iter2 < max_iter2

            w0 = ψ[1+1]*(Y_0 - κ/q(θ_0)) # wages at t = 0

            @inbounds for t = 0:T
                zt          = unique(zz[t+1, :])  
                ψ_t         = ψ[t + 1]
                @inbounds for n = 1:length(zt)
                    z = zt[n]
                    if ε == 1 # can solve analytically
                        aa = (z/w0 + sqrt((z/w0)^2))/2(1 + ψ_t*σ_η^2)
                    else # exclude the choice of zero effort
                        aa = find_zeros(x -> x - max(z*x/w0 -  (ψ_t/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0.0, 10.0) 
                    end

                    # create a flag matrix -- decide how to handle any violations
                    idx1            = findall(isequal(z), zz[t+1, :])
                    az[idx1,t+1]    .= isempty(aa) ? 0 : aa[1]
                    flag[idx1,t+1]  .= ((z*az[n, t+1]/w0 +  (ψ_t/ε)*(hp(az[n,t+1])*σ_η)^2) < 0) + isempty(aa) + (w0 < 0)
                    yy[idx1,t+1]    .= ((β*(1-s))^(t))*az[n,t+1]*z
                end  
            end

            # Expand
            y = sum(yy,dims=2)
            @inbounds for n = 1:size(zz,2)
                idx        = findall(isequal(T+1), vec(sum(ZZ .== zz[:,n], dims=1)))
                YY[idx]    .= y[n]
                AZ[:,idx]  .= az[n,:]
            end   

            # Numerical approximation of E_0[Y]
            Y_1  = mean(YY)
            err2 = abs(Y_0 - Y_1)

            if Y_1 < Y_0 
                Y_ub  = copy(Y_0)
            elseif Y_1 > Y_0 || w0 < 0
                Y_lb  = copy(Y_0)
            end

            #Y_0  = (Y_lb + Y_ub)/2  # converges slowly
            Y_0  = α*Y_0 + (1-α)*Y_1 # converges faster
            #println(Y_0)
            iter2 += 1
        end

        # Numerical approximation of expected lifetime utility
        V0 = zeros(size(ZZ,2))
        v0 = zeros(size(zz,2))
        w0 = ψ[1+1]*(Y_0 - κ/q(θ_0)) # wages at t = 0

        @inbounds for n = 1:size(zz,2)
            t1 = 0
            @inbounds for t = 0:T
                t1    += (t == 0) ? 0 : 0.5*(ψ[t + 1]*hp(az[n,t+1])*σ_η)^2 
                t2    = h(az[n,t+1])
                t3    = ω(t+1)
                if savings == false
                    v0[n] += (log(w0) - t1 - t2 + t3*β*s)*(β*(1-s))^t 
                else
                    v0[n] += (log(w0) + t1 - t2 + t3*β*s)*(β*(1-s))^t
                end
            end
            idx        = findall(isequal(T+1), vec(sum(ZZ .== zz[:,n], dims=1)))
            V0[idx]   .= v0[n]
        end

        # check IR constraint
        IR     = mean(V0)
        err1   = abs(IR - ω(0))

        if IR >  ω(0) 
            θ_lb  = copy(θ_0)
        elseif IR <  ω(0) 
            θ_ub  = copy(θ_0)
        end

        println(θ_0)
        θ_0 = (θ_lb + θ_ub)/2
        iter1 += 1
    end

    return (θ = θ_0, flags = flag, Y = Y_0, V = IR, ω0 = ω(0), err1 = err1, 
    err2 = err2, iter1 = iter1, iter2 = iter2) 
end

end # module
