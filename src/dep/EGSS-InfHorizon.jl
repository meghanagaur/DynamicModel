#= Solve the infinite horizon dynamic EGSS model. 
Monthly calibration, no savings. =#

"""
Set up the dynamic EGSS model, where m(u,v) = (uv)/(u^ι + v^⟦)^(1/ι),
η ∼ N(0, σ_η^2), log(z_t) = μ_z + ρ*log(z_t-1) + u_t, u_t ∼ N(0, σ_z^2),
and y_t = z_t(a_t + η_t).
β    = discount factor
s    = exogenous separation rate
α    = elasticity of matching function 
μ    = matching function efficiency
κ    = vacancy-posting cost
ω    = worker's PV from unemployment (infinite horizon)
χ    = prop. of unemp benefit to z / actual unemp benefit
γ    = intercept for unemp benefit w / procyclical benefit
z_ss = mean productivity (this is just a definition)
z_0  = value of initial z; MUST BE ON ZGRID.
σ_η  = st dev of η distribution
z_1  = initial prod. (= log(z_ss) by default)
ρ    = persistence of log prod. process
σ_ϵ  = variance of noise in log prod. process
ε    = Frisch elasticity: disutility of effort
ψ    = pass-through parameter

procyclical == (procyclical unemployment benefit)
""" 
#=
Quarterly:
function model(; β = 0.99, s = 0.088, κ = 0.474, ι = 1.67, ε = 0.5, σ_η = 0.05, z_ss = 1.0,
    ρ =  0.87, σ_ϵ = 0.008, χ = 0.1, γ = 0.66, z_1 = z_ss, μ_z = log(z_ss), N_z = 11, procyclical = true)
Quarterly->monthly
ρ = 0.87^(1/3)
sqrt(0.017^2 / mapreduce(j-> ρ^(2j), +, [0:2;])) = 0.01
=#
function model(; β = 0.99^(1/3), s = 0.035, κ = 0.45, ε = 0.3, σ_η = 0.5, z_ss = 1.0, μ = 0.42,
    α = 0.72, hbar = 1, ρ =  0.95^(1/3), σ_ϵ = 0.0065, χ = 0.0, γ = 0.7, z_1 = z_ss, N_z = 11)

    # Basic parameterization
    q(θ)    = μ*θ^(-α)                      # vacancy-filling rate
    f(θ)    = μ*θ^(1-α)                     # job-filling rate
    u(c)    = log(max(c, eps()))            # utility from consumption                
    h(a)    = hbar*(a^(1 + 1/ε))/(1 + 1/ε)  # disutility from effort  
    hp(a)   = hbar*a^(1/ε)                  # h'(a)
    u(c, a) = u(c) - h(a)                   # overall utility function

    # Define productivity grid
    if (iseven(N_z)) error("N_z must be odd") end 
    μ_z       = log(z_ss) - ((1-ρ)*σ_ϵ^2)/(2*(1-ρ^2))            # normalize E[z_t] = 1
    logz, P_z = rouwenhorst(μ_z, ρ, σ_ϵ, N_z)                    # discretized logz grid & transition probabilties
    zgrid     = exp.(logz)                                       # actual productivity grid
    z_1_idx   = findfirst(isapprox(z_1, atol = 0.0001), zgrid)   # index of z0 on zgrid

    # Pass-through parameter
    ψ    = 1 - β*(1-s)

    # Unemployment benefit given aggregate state: (z) 
    if isapprox(χ, 0) 
        procyclical = false
    else
        procyclical = true
        ξ(z) = (γ)*(z/z_ss)^χ 
    end

    # PV of unemp = PV of utility from consuming unemployment benefit forever
    if procyclical == false
        ω = log(γ)/(1-β) # scalar
    elseif procyclical == true
        #println("Solving for value of unemployment...")
        ω = unemploymentValue(β, ξ, u, zgrid, P_z).v0 # N_z x 1
    end
    
    return (β = β, s = s, κ = κ, ε = ε, σ_η = σ_η, ρ = ρ, σ_ϵ = σ_ϵ, z_ss = z_ss, μ = μ,
     α = α, hbar = hbar, ω = ω, N_z = N_z, q = q, f = f, ψ = ψ, z_1 = z_1, h = h, u = u, hp = hp, 
    z_1_idx = z_1_idx, zgrid = zgrid, P_z = P_z, χ = χ, γ = γ, procyclical = procyclical)
end

"""
Solve for the optimal effort a(z | z_0), given Y(z_0), θ(z_0), and z.
Note: a_min > 0 to allow for numerical error.
"""
function optA(z, modd, w_0; a_min = 10^-10, a_max = 10)
    @unpack ψ, ε, q, κ, hp, σ_η = modd
    if ε == 1 # can solve analytically for positive root
        a      = (z/w_0)/(1 + ψ*σ_η^2)
        a_flag = 0
        #aa2 = find_zeros(x -> x - max(z*x/w_0 -  (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), a_min, a_max) 
        #@assert(isapprox(aa,aa2[1]))
    else 
        # solve for positive root.
        aa         = find_zeros(x -> x - max(z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), a_min, a_max) 
        if ~isempty(aa)
            a      = aa[1] 
            a_flag = max( ((z*a/w_0 - (ψ/ε)*(hp(a)*σ_η)^2) < 0), (length(aa) > 1) ) 
        elseif isempty(aa)
            a       = 0
            a_flag  = 1
        end
    end
    y      = a*z # Expectation of y_t = z_t*(a_t+ η_t) over η_t (given z_t)
    return a, y, a_flag
end

"""
Solve the infinite horizon EGSS model using a bisection search on θ.
"""
function solveModel(modd; max_iter1 = 50, max_iter2 = 1000, max_iter3 = 1000,
    tol1 = 10^-6, tol2 = 10^-8, tol3 =  10^-8, noisy = true, q_lb_0 =  0.0, q_ub_0 = 2.0)

    @unpack β, s, κ, μ, α, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, procyclical, N_z, z_1_idx = modd  

    # set tolerance parameters for outermost loop
    err1  = 10
    iter1 = 1
    # initialize tolerance parameters for inner loops (for export)
    err2  = 10
    iter2 = 1
    err3  = 10
    iter3 = 1

    # Initialize default values and search parameters
    ω_0    = procyclical ? ω[z_1_idx] : ω # unemployment value at z_1
    q_lb   = q_lb_0          # lower search bound for q
    q_ub   = q_ub_0          # upper search bound for q
    q_0    = (q_lb + q_ub)/2 # initial guess for q
    l      = 0               # dampening parameter
    Y_0    = 0               # initalize Y for export
    U      = 0               # initalize worker's EU from contract for export
    w_0    = 0               # initialize initial wage constant for export

    # Initialize series
    az     = zeros(N_z)   # a(z|z_1)                         
    yz     = zeros(N_z)   # y(z|z_1)                         
    a_flag = zeros(N_z)   # flag for a(z|z_1)                         

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 <= max_iter1  

        if noisy 
            println(q_0)
        end

        # Look for a fixed point in Y(z | z_1), ∀ z
        err2   = 10
        iter2  = 1      
        Y_0    = ones(N_z)*(50*κ/q_0)   # initial guess for Y(z | z_1)
        
        @inbounds while err2 > tol2 && iter2 <= max_iter2   
            w_0  = ψ*(Y_0[z_1_idx] - κ/q_0) # constant for wage difference equation
            # Solve for optimal effort a(z | z_1)
            @inbounds for (iz,z) in enumerate(zgrid)
                az[iz], yz[iz], a_flag[iz] = optA(z, modd, w_0)
            end
            Y_1    = yz + β*(1-s)*P_z*Y_0    
            err2   = maximum(abs.(Y_0 - Y_1))  # Error       
            if (err2 > tol2) 
                iter2 += 1
                if (iter2 < max_iter2) 
                    Y_0    = l*Y_0 + (1-l)*Y_1 
                end
            end
            #println(Y_0[z_1_idx])
        end

        # Solve recursively for the PV utility from the contract
        err3  = 10
        iter3 = 1  
        ω_vec = procyclical ?  copy(ω) : ω*ones(N_z)
        W_0   = copy(ω_vec) # initial guess
        flow  = -(1/(2ψ))*(ψ*hp.(az)*σ_η).^2 - h.(az) + β*s*(P_z*ω_vec)
        @inbounds while err3 > tol3 && iter3 <= max_iter3
            W_1  = flow + β*(1-s)*(P_z*W_0)
            err3 = maximum(abs.(W_1 - W_0))
            W_0  = l*W_0 + (1-l)*W_1
            #println(W_0[z_1_idx])
            iter3 +=1
        end
        
        # Check the IR constraint (must bind)
        U      = (1/ψ)*log(max(eps(), w_0)) + W_0[z_1_idx] # nudge w_0 to avoid runtime error
        err1   = abs(U - ω_0)
        
        # Upate θ accordingly: note U is decreasing in θ (=> increasing in q)
        if U < ω_0 # increase q (decrease θ)
            q_lb  = copy(q_0)
        elseif U > ω_0 # decrease q (increase θ)
            q_ub  = copy(q_0)
        end

        # export the accurate iter & q value
        if (err1 > tol1)
            iter1 += 1
            if (iter1 < max_iter1) 
                q_0    = (q_lb + q_ub)/2
                # exit loop if q is stuck near the bounds 
                if min(abs(q_0 - q_ub_0), abs(q_0 - q_lb_0)) <= 10^-6
                    # check if the  IR constraint is satisfied
                    if U > ω_0
                        break
                    else  
                        iter1 = max_iter1 + 1              
                    end 
                    #iter1 = max_iter1 + 1              
                end
            end
        end
    end

    return (θ = (μ/q_0)^(1/α), Y = Y_0[z_1_idx], U = U, ω_0 = ω_0, w_0 = w_0, mod = modd, 
    az = az, yz = yz, err1 = err1, err2 = err2, err3 = err3, iter1 = iter1, iter2 = iter2, iter3 = iter3, wage_flag = (w_0 <= 0),
    effort_flag = maximum(a_flag), exit_flag1 = (iter1 > max_iter1), exit_flag2 = (iter2 > max_iter2), exit_flag3 = (iter3 > max_iter3))
end

"""
Solve for the value of unemployment, with a
procyclical unemployment benefit via value function iteration.
"""
function unemploymentValue(β, ξ, u, zgrid, P_z; tol = 10^-8, max_iter = 5000)
    N_z    = length(zgrid)
    v0_new = zeros(N_z)
    v0     = u.(ξ.(zgrid))
    iter   = 1
    err    = 10

    # solve via simple value function iteration
    @inbounds while err > tol && iter < max_iter
        v0_new = u.(ξ.(zgrid)) + β*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter   +=1
    end
    return (v0 = v0, err = err, iter = iter) 
end