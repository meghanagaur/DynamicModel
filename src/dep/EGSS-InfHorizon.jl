#= Solve the infinite horizon dynamic EGSS model (first pass). 
Quarterly calibration, no savings. =#

"""
Set up the dynamic EGSS model, where m(u,v) = (uv)/(u^ι + v^⟦)^(1/ι),
η ∼ N(0, σ_η^2), log(z_t) = μ_z + ρ*log(z_t-1) + u_t, u_t ∼ N(0, σ_z^2),
and y_t = z_t(a_t + η_t).
β    = discount factor
r    = interest rate
s    = exogenous separation rate
ι    = matching elasticity
κ    = vacancy-posting cost
ω    = worker's PV from unemployment (infinite horizon)
χ    = prop. of unemp benefit to z / actual unemp benefit
γ    = intercept for unemp benefit w/ procyclical benefit
z_ss = steady state productivity (definition)
σ_η  = st dev of η distribution
μ_z  = unconditional mean of log prod. process (= log(z_ss) by default)
z0   = initial prod. (= z_ss by default)
ρ    = persistence of log prod. process
σ_ϵ  = variance of innovation in log prod. process
ε    = Frisch elasticity: disutility of effort
ψ    = pass-through parameters

procyclical == (procyclical unemployment benefit)
""" 
#ρ =  0.97   # ι = 1.25 (monthly)
#σ_ϵ = 0.008 # κ =  0.213 (quarterly)
function model(; β = 0.99, s = 0.1, κ = 0.45, ι = 1.5, ε = 0.5, σ_η = 0.05, z_ss = 1.0,
    ρ =  0.87, σ_ϵ = 0.008, χ = 0.1, γ = 0.66, z0 = z_ss, μ_z = log(z_ss), N_z = 11, procyclical = true)

    # Basic parameterization
    q(θ)    = 1/(1 + θ^ι)^(1/ι)                     # vacancy-filling rate
    f(θ)    = 1/(1 + θ^-ι)^(1/ι)                    # job-filling rate
    u(c)    = log(max(c, eps()))                    # utility from consumption                
    h(a)    = (a^(1 + 1/ε))/(1 + 1/ε)               # disutility from effort  
    u(c, a) = u(c) - h(a)                           # utility function
    hp(a)   = a^(1/ε)                               # h'(a)
    r       = 1/β -1                                # interest rate        

    # Define productivity grid
    if (iseven(N_z)) error("N_z must be odd") end 
    logz, P_z = rouwenhorst(μ_z, ρ, σ_ϵ, N_z)        # discretized logz grid & transition probabilties
    zgrid     = exp.(logz)                           # actual productivity grid
    z0_idx    = findfirst(isapprox(z0), zgrid)       # index of z0 on zgrid

    # Pass-through parameter
    ψ    = 1 - β*(1-s)

    # Unemployment benefit given aggregate state: (z) 
    if procyclical == true
        ξ(z) = γ + χ*(z - z_ss) 
    elseif procyclical == false
        ξ    = γ
    end

    # PV of unemp = PV of utility from consuming unemployment benefit forever
    if procyclical == false
        ω = log(ξ)/(1-β) # scalar
    elseif procyclical == true
        println("Solving for value of unemployment...")
        ω = unemploymentValue(β, ξ, u, zgrid, P_z).v0 # N_z x 1
    end
    
    return (β = β, r = r, s = s, κ = κ, ι = ι, ε = ε, σ_η = σ_η, ρ = ρ, σ_ϵ = σ_ϵ, z_ss = z_ss,
    ω = ω, μ_z = μ_z, N_z = N_z, q = q, f = f, ψ = ψ, z0 = z0, h = h, u = u, hp = hp, 
    z0_idx = z0_idx, zgrid = zgrid, P_z = P_z, ξ = ξ, χ = χ, γ = γ, procyclical = procyclical)
end

"""
Solve for the optimal effort a(z | z_0), given Y(z_0), θ(z_0), and z.
"""
function optA(z, modd, w_0)
    @unpack ψ, ε, q, κ, hp, σ_η = modd
    if ε == 1 # can solve analytically for positive root
        aa = (z/w_0)/(1 + ψ*σ_η^2)
    else # solve for the positive root 
        aa = find_zeros(x -> x - max(z*x/w_0 -  (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0, 20) 
    end
    if ~isempty(aa)
        a      = ~isempty(aa) ? aa[1] : 0
        a_flag = max( ((z*aa[1]/w_0 + (ψ/ε)*(hp(aa[1])*σ_η)^2) < 0), (length(aa) > 1) )
    elseif isempty(aa)
        a       = 0
        a_flag  = 1
    end
    y      = a*z # Expectation of y_t over η_t (given z_t)
    return a, y, a_flag
end

"""
Solve the infinite horizon EGSS model using a bisection search on θ.
"""
function solveModel(modd; max_iter1 = 50, max_iter2 = 1000, max_iter3 = 1000,
    tol1 = 10^-7, tol2 = 10^-8, tol3 =  10^-8, noisy = true, θ_lb_0 =  0.0, θ_ub_0 = 5.0)

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
    θ_lb   = θ_lb_0          # lower search bound for θ
    θ_ub   = θ_ub_0          # upper search bound for θ
    θ_0    = (θ_lb + θ_ub)/2 # initial guess for θ
    α      = 0               # dampening parameter
    Y_0    = 0               # initalize Y_0 for export
    U      = 0               # initalize worker's EU from contract for export
    w_0    = 0               # initialize initial wage constant for export

    # Initialize series
    az     = zeros(N_z)   # a(z|z_0)                         
    yz     = zeros(N_z)   # y(z|z_0)                         
    a_flag = zeros(N_z)   # flag for a(z|z_0)                         

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 < max_iter1  

        if noisy 
            println(θ_0)
        end

        # Look for a fixed point in Y(z | z_0), ∀ z
        err2   = 10
        iter2  = 1      
        Y_0    = ones(N_z)*(30*κ/q(θ_0))   # initial guess for Y(z | z_0)
        
        @inbounds while err2 > tol2 && iter2 < max_iter2   
            w_0  = ψ*(Y_0[z0_idx] - κ/q(θ_0)) # constant for wage difference equation
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
            θ_lb  = copy(θ_0)
        elseif U < ω_0
            θ_ub  = copy(θ_0)
        end

        if (err1 > tol1) && (iter1 < max_iter1) 
            θ_0    = (θ_lb + θ_ub)/2
            iter1 += 1
        end

        # stop if we get stuck near the bounds on θ
        if (abs(θ_0 - θ_lb_0) < 10^-8) || (abs(θ_0 - θ_ub_0) < 10^-8 )
            iter1 = max_iter1
            break
        end
    end

    return (θ = θ_0, Y = Y_0[z0_idx], U = U, ω_0 = ω_0, w_0 = w_0, mod = modd, 
    az = az, yz = yz, err1 = err1, err2 = err2, err3 = err3, iter1 = iter1, iter2 = iter2, iter3 = iter3, wage_flag = (w_0 < 0),
    effort_flag = maximum(a_flag), exit_flag1 = (iter1 >= max_iter1), exit_flag2 = (iter2 >= max_iter2), exit_flag3 = (iter3 >= max_iter3))
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