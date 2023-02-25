#= Solve the infinite horizon dynamic EGSS model. 
Monthly calibration, no savings. =#

"""
Set up the dynamic EGSS model:

m(u,v) = (uv)/(u^ι + v^⟦)^(1/ι)
log(z_t) = (1 - ρ)μ_z + ρ*log(z_t - 1 ) + ϵ_t, where ϵ_t ∼ N(0, σ_ϵ^2),
y_t = z_t(a_t + η_t), where η ∼ N(0, σ_η^2).

β    = discount factor
s    = exogenous separation rate
ι    = controls elasticity of matching function 
κ    = vacancy-posting cost
ω    = worker's PV from unemployment (infinite horizon)
χ    = elasticity of unemp benefit to z 
γ    = level of unemp benefit 
zbar = mean productivity
σ_η  = st dev of η distribution
μ_z  = unconditional mean of log productivity 
ρ    = persistence of log productivity
σ_ϵ  = conditional variance of log productivity
ε    = Frisch elasticity: disutility of effort
hbar = disutility of effort - level
ψ    = pass-through parameter

procyclical == (procyclical unemployment benefit)
""" 
function model(; β = 0.99^(1/3), s = 0.031, κ = 0.45, ε = 0.5, σ_η = 0.5, zbar = 1.0, ι = 0.8,
    hbar = 1.0, ρ =  0.95^(1/3), σ_ϵ = 0.003, χ = 0.0, γ = 0.6, N_z = 11)

    # Basic parameterization
    q(θ)    = (1 + θ^ι)^(-1/ι)                          # job-filling rate
    f(θ)    = (1 + θ^-ι)^(-1/ι)                         # job-finding rate
    u(c)    = log(max(c, eps()))                        # utility from consumption                
    h(a)    = hbar*(max(a, eps())^(1 + 1/ε))/(1 + 1/ε)  # disutility from effort  
    hp(a)   = hbar*max(eps(), a)^(1/ε)                  # h'(a)
    u(c, a) = u(c) - h(a)                               # overall utility function

    # Define productivity grid
    if (iseven(N_z)) error("N_z must be odd") end 
    μ_z             = log(zbar) - (σ_ϵ^2)/(2*(1-ρ^2))   # normalize E[logz], so that E[z_t] = 1
    logz, P_z, p_z  = rouwenhorst(μ_z, ρ, σ_ϵ, N_z)     # log z grid, transition matrix, invariant distribution
    zgrid           = exp.(logz)                        # z grid in levels
    z_ss_idx        = findfirst(isapprox(μ_z, atol = 1e-6), logz )

    # Pass-through 
    ψ    = 1 - β*(1-s)

    # Unemployment benefit given aggregate state: (z) 
    if isapprox(χ, 0) 
        procyclical = false
    else
        procyclical = true
        ξ(z) = (γ)*z^χ 
    end

    # PV of unemp = PV of utility from consuming unemployment benefit forever
    if procyclical == false
        ω = log(γ)/(1 - β) # scalar
    elseif procyclical == true
        ω = unemploymentValue(β, ξ, u, zgrid, P_z).v0 # N_z x 1
    end
    
    return (β = β, s = s, κ = κ, ε = ε, σ_η = σ_η, ρ = ρ, σ_ϵ = σ_ϵ, zbar = zbar, μ_z = μ_z,
    ι = ι, hbar = hbar, ω = ω, N_z = N_z, q = q, f = f, ψ = ψ, h = h, u = u, hp = hp, z_ss_idx = z_ss_idx,
    logz = logz, zgrid = zgrid, P_z = P_z, p_z = p_z, χ = χ, γ = γ, procyclical = procyclical)
end

"""
Solve for the optimal effort a(z | z_0), given Y(z_0), θ(z_0), and z.
Note: a_min > 0 to allow for numerical error. 
If check_min == true, then root-finding checks for multiple roots (slow).
"""
function optA(z, modd, w_0; a_min = 10^-6, a_max = 100.0, check_mult = false)
   
    @unpack ψ, ε, q, κ, hp, σ_η, hbar = modd
    
    if ε == 1 # can solve analytically for positive root
        a      = (z/w_0)/(1 + ψ*σ_η^2)
        a_flag = 0
    else 

        # solve for the positive root. nudge to avoid any runtime errors.
        if check_mult == false 
            aa          = solve(ZeroProblem( x -> (x > a_min)*(x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0))
            #aa         = solve(ZeroProblem( x -> (x > a_min)*(x - max( z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0))
            #aa         = fzero(x -> (x > a_min)*(x - max( z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0)
            #aa         = find_zero(x -> x - max(z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), (a_min, a_max)) # requires bracketing
        
        elseif check_mult == true
            #aa         = find_zeros(x -> x - max(z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), a_min, a_max)  
            aa          = find_zeros( x -> (x > a_min)*(x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10,  a_min, a_max)
        end

        if ~isempty(aa) 
            if (maximum(isnan.(aa)) == 0 )
                a      = aa[1] 
                a_flag = max(a < a_min , max( ((z*a/w_0 - (ψ/ε)*(hp(a)*σ_η)^2) < 0), (length(aa) > 1) ) )
            else
                a       = eps()
                a_flag  = 1
            end
        elseif isempty(aa) 
            a           = eps()
            a_flag      = 1
        end
    end

    y      = a*z # Expectation of y_t = z_t*(a_t+ η_t) over η_t (given z_t)

    return a, y, a_flag

end

"""
Solve the infinite horizon EGSS model using a bisection search on θ.
"""
function solveModel(modd; z_0 = nothing, max_iter1 = 50, max_iter2 = 1000, max_iter3 = 1000, a_min = 10^-6,
    tol1 = 10^-8, tol2 = 10^-8, tol3 =  10^-8, noisy = true, q_lb_0 =  0.0, q_ub_0 = 1.0, check_mult = false)

    @unpack β, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, procyclical, N_z, z_ss_idx = modd  
    
    # find index of z_0 on the productivity grid 
    if isnothing(z_0)
        z_0_idx = z_ss_idx
    else
        z_0_idx = findfirst(isapprox(z_0, atol = 1e-6), zgrid)  
    end

    # set tolerance parameters for outermost loop
    err1    = 10
    iter1   = 1
    # initialize tolerance parameters for inner loops (for export)
    err2    = 10
    iter2   = 1
    err3    = 10
    iter3   = 1
    IR_err  = 10
    flag_IR = 0

    # Initialize default values and search parameters
    ω_0    = procyclical ? ω[z_0_idx] : ω # unemployment value at z_0
    ω_vec  = procyclical ?  copy(ω) : ω*ones(N_z)
    q_lb   = q_lb_0          # lower search bound for q
    q_ub   = q_ub_0          # upper search bound for q
    q_0    = (q_lb + q_ub)/2 # initial guess for q
    α      = 0               # dampening parameter
    Y_0    = 0               # initalize Y for export
    U      = 0               # initalize worker's EU from contract for export
    w_0    = 0               # initialize initial wage constant for export

    # Initialize series
    az     = zeros(N_z)   # a(z|z_0)                         
    yz     = zeros(N_z)   # y(z|z_0)                         
    a_flag = zeros(N_z)   # flag for a(z|z_0)                         

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 <= max_iter1  

        if noisy 
            println("iter:\t"*string(iter1))
            println("error:\t"*string(err1))
            println("q_0:\t"*string(q_0))
        end

        # Look for a fixed point in Y(z | z_0), ∀ z
        err2   = 10
        iter2  = 1      
        Y_0    = ones(N_z)*(50*κ/q_0)   # initial guess for Y(z | z_0)
        
        @inbounds while err2 > tol2 && iter2 <= max_iter2   
           
            w_0  = ψ*(Y_0[z_0_idx] - κ/q_0) # constant for wage difference equation
           
            # Solve for optimal effort a(z | z_0)
            @inbounds for (iz,z) in enumerate(zgrid)
                az[iz], yz[iz], a_flag[iz] = optA(z, modd, w_0; check_mult = check_mult, a_min = a_min)
            end
            Y_1    = yz + β*(1-s)*P_z*Y_0    
            err2   = maximum(abs.(Y_0 - Y_1))  # Error       
            if (err2 > tol2) 
                iter2 += 1
                if (iter2 < max_iter2) 
                    Y_0    = α*Y_0 + (1 - α)*Y_1 
                end
            end
            #println(Y_0[z_0_idx])
        end

        # Solve recursively for the PV utility from the contract
        err3  = 10
        iter3 = 1  
        W_0   = copy(ω_vec) # initial guess
        flow  = -(1/(2ψ))*(ψ*hp.(az)*σ_η).^2 - h.(az) + β*s*(P_z*ω_vec)

        @inbounds while err3 > tol3 && iter3 <= max_iter3
            W_1  = flow + β*(1-s)*(P_z*W_0)
            err3 = maximum(abs.(W_1 - W_0))
            if (err3 > tol3) 
                iter3 += 1
                if (iter3 < max_iter3) 
                    W_0  = α*W_0 + (1 - α)*W_1
                end
            end
            #println(W_0[z_0_idx])
        end
        
        # Check the IR constraint (must bind)
        U      = (1/ψ)*log(max(eps(), w_0)) + W_0[z_0_idx] # nudge w_0 to avoid runtime error
        
        # Upate θ accordingly: U decreasing in θ (increasing in q)
        if U < ω_0              # increase q (decrease θ)
            q_lb  = copy(q_0)
        elseif U > ω_0          # decrease q (increase θ)
            q_ub  = copy(q_0)
        end

        # Bisection
        IR_err = U - ω_0                             # check whether IR constraint holds
        q_1    = (q_lb + q_ub)/2                     # update q
        #err1   = min(abs(IR_err), abs(q_1 - q_0))   # compute convergence criterion
        err1    = abs(IR_err)

        # Record info on IR constraint
        flag_IR = (IR_err < 0)*(abs(IR_err) > tol1)

        # Export the accurate iter & q value
        if err1 > tol1
            if min(abs(q_1 - q_ub_0), abs(q_1 - q_lb_0))  < 10^(-10) 
                break
            else
                q_0     = α*q_0 + (1 - α)*q_1
                iter1  += 1
            end
        end

    end

    return (θ = (q_0^(-ι) - 1)^(1/ι), Y = Y_0[z_0_idx], U = U, ω = ω_0, w_0 = w_0, mod = modd, IR_err = IR_err*flag_IR, flag_IR = flag_IR,
    az = az, yz = yz, err1 = err1, err2 = err2, err3 = err3, iter1 = iter1, iter2 = iter2, iter3 = iter3, wage_flag = (w_0 <= 0),
    effort_flag = maximum(a_flag), conv_flag1 = (iter1 > max_iter1), conv_flag2 = (iter2 > max_iter2), conv_flag3 = (iter3 > max_iter3))
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