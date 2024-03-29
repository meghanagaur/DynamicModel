using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DataStructures, Distributions, ForwardDiff, Interpolations,
 LinearAlgebra, Parameters, Random, Roots, StatsBase, DynamicModel

 # vary z_0 <-> vary μ_z
function model(; β = 0.99, s = 0.1, κ = 0.213, ι = 1.25, ε = 0.5, σ_η = 0.05, z_ss = 1.0,
    ρ =  0.978, σ_ϵ = 0.007, χ = 0.1, γ = 0.625, z0 = z_ss, μ_z = log(z0), N_z = 11, procyclical = true)

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
    
    return (β = β, r = r, s = s, κ = κ, ι = ι, ε = ε, σ_η = σ_η, ρ = ρ, σ_ϵ = σ_ϵ, 
    ω = ω, μ_z = μ_z, N_z = N_z, q = q, f = f, ψ = ψ, z0 = z0, h = h, u = u, hp = hp, 
    z0_idx = z0_idx, zgrid = zgrid, P_z = P_z, ξ = ξ, χ = χ, γ = γ, procyclical = procyclical)
end

# Define the z0grid and χ
χ       = 0.3
zmin    = 0.97
zmax    = 1.03
dz      = 0.0011
z0grid  = collect(zmin:dz:zmax)
idx     = floor(Int64,median(1:length(z0grid))) # SS
modd    = OrderedDict{Int64, Any}()

# Solve the model for different z_0
@time @inbounds for (iz,z0) in enumerate(z0grid)
    modd[iz] =  solveModel(model(z0 = z0, χ = χ), noisy = false)
end

@unpack β,s,ψ,ρ,σ_ϵ,hp,σ_η,q,κ,ι,ε  = model()

## Store series of interest
w0    = [modd[i].w_0 for i = 1:length(z0grid)]      # w0 (constant)
theta = [modd[i].θ for i = 1:length(z0grid)]        # tightness
W     = [modd[i].w_0/ψ[1] for i = 1:length(z0grid)] # PV of wages
Y     = [modd[i].Y for i = 1:length(z0grid)]        # PV of output
ω0    = [modd[i].ω_0 for i = 1:length(z0grid)]      # PV of unemployment at z0
J     = Y .- W

# Approx elasticity using forward finite differences for derivatives
function elasticity(yy, zgrid, dz) #, dlz)
    e1 = zgrid[1:end-1].*(yy[2:end]-yy[1:end-1])./(dz.*yy[1:end-1])
    #e2 = (log.(yy[2:end])-log.(yy[1:end-1]))./(dlz)
    return e1 #, e2
end

# Approx slope using forward finite differences 
function slope(xx, dz)
    return (xx[2:end]-xx[1:end-1])./dz
end

# plot series vs z0
p1 = plot(z0grid, theta, ylabel=L"\theta_0", xlabel=L" z_0")
p2 = plot(z0grid, W, ylabel=L"W_0", xlabel=L" z_0")
p3 = plot(z0grid, Y, ylabel=L"Y_0",xlabel=L" z_0")
p4 = plot(z0grid, J, ylabel=L"J_0",xlabel=L" z_0")
plot(p1, p2, p3, p4, layout = (2, 2), legend=:false)

# Plot unemployment value at z0 vs z0
plot(z0grid, ω0, ylabel=L"\omega(z_0)", xlabel=L" z_0",  linewidth=4, linecolor=:cyan, label="actual benefit")

# Plot elasticities
t1 = elasticity(theta, z0grid, dz)
w1 = elasticity(W, z0grid, dz)
y1 = elasticity(Y, z0grid, dz)
j1 = elasticity(J, z0grid, dz)

p1 = plot(z0grid[1:end-1], t1, ylabel=L"d\log \theta_0 / d \log z_0", xlabel=L" z_0")
p2 = plot(z0grid[1:end-1], w1, ylabel=L"d \log W_0 d / \log z_0", xlabel=L" z_0")
p3 = plot(z0grid[1:end-1], y1, ylabel=L"d \log Y_0 d / \log z_0",xlabel=L" z_0")
p4 = plot(z0grid[1:end-1], j1, ylabel=L"d \log J_0 d / \log z_0",xlabel=L" z_0")
plot(p1, p2, p3, p4, layout = (2, 2), legend=:false)

# plot slopes 
tt  = slope(theta, dz)
ww  = slope(W, dz)
yy  = slope(Y, dz)
jj  = slope(J, dz)

p1 = plot(z0grid[1:end-1], tt, ylabel=L"d \theta_0 / d  z_0", xlabel=L" z_0")
p2 = plot(z0grid[1:end-1], ww, ylabel=L"d  W_0 d /  z_0", xlabel=L" z_0")
p3 = plot(z0grid[1:end-1], yy, ylabel=L"d  Y_0 d / z_0",xlabel=L" z_0")
p4 = plot(z0grid[1:end-1], jj, ylabel=L"d J_0 d / z_0",xlabel=L" z_0")
plot(p1, p2, p3, p4, layout = (2, 2), legend=:false)

# double-check slopes
qq(x) =  -(x^(-1+ι))*(1+x^ι)^(-1 -1/ι) # q'(θ)
xx    = theta[1:end-1]
ww2   = yy + (κ./(q.(xx)).^2).*tt.*qq.(xx) # dW/dz0 = dy/dz0 - d ( k/q(θ) ) / dz0

# check on slopes
plot(slope(q.(theta[1:end]), dz))
plot!(qq.(xx).*tt)

plot(slope(1 ./q.(theta[1:end]), dz))
plot!( (-qq.(xx).*tt) ./ (q.(theta[1:end-1]).^2) )

# Plot dY/dz0, dW/z0, and dJ/dz0 and check that these make sense.
plot(z0grid[1:end-1],  ww2, label=L"d W_0 d / z_0 *",xlabel=L" z_0",linecolor=:yellow, linewidth=3) #check 
plot!(z0grid[1:end-1], ww, label=L"d W_0 d / z_0", xlabel=L" z_0", linecolor=:orange)
plot!(z0grid[1:end-1], yy, label=L"d  Y_0 d /  z_0",xlabel=L" z_0", linecolor=:red)
plot!(z0grid[1:end-1], yy-ww2, label=L"d  J_0 d /  z_0 *",xlabel=L" z_0", linecolor=:cyan, linewidth=3) # check
plot!(z0grid[1:end-1], jj, label=L"d  J_0 d / z_0",xlabel=L" z_0", linecolor=:blue)
plot!(z0grid[1:end-1], yy -ww, label=L"d  J_0 d / z_0",xlabel=L" z_0", linecolor=:green)
plot!(z0grid[1:end-1], -(κ./(q.(xx)).^2).*tt.*qq.(xx), label=L"d  J_0 d / z_0 *",xlabel=L" z_0", linecolor=:purple) # check

# Now consider a Hall vs Bonus Economy comparison

# Solve for expected PV of z_t's
exp_z = zeros(length(z0grid)) 
@inbounds for (iz,z0) in enumerate(z0grid)
    @unpack zgrid, P_z, N_z = modd[iz].mod
    z0_idx  = findfirst(isequal(z0), zgrid)  # index of z0 on zgrid
    
    # initialize guesses
    v0     = zgrid./(1-β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
    # solve via simple value function iteration
    @inbounds while err > 10^-8 && iter < 500
        v0_new = zgrid + β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end
    exp_z[iz]   = v0[z0_idx]
end

a_opt    = Y[idx]./exp_z[idx]  # exactly match SS PV of output in the 2 models
w        = W[idx]              # match SS PV of wages (E_0[w_t] = w_0 from martingale property)
JJ       = a_opt.*exp_z .- w   # Hall economy profits
YY       = a_opt.*exp_z        # Hall economy output 

# plot profits in the different economies
p4= plot(z0grid, JJ, label="Hall: Fixed w and fixed a", linecolor=:cyan,linewidth=3)
plot!(p4, z0grid, Y .- w, label="Fixed w and variable a", linecolor=:red)
plot!(p4,z0grid, YY.- W, label="Fixed a and variable w", linecolor=:blue)
plot!(p4,z0grid, J, label="Bonus economy: Variable w and variable a", linecolor=:black, legend=:topleft)
xlabel!(L"z_0")
ylabel!(L"J_0")

# isolate effort/wage movements
p1 = plot( z0grid, Y , label="Variable a", linecolor=:red, linewidth=3)
plot!(p1, z0grid, YY, label="Fixed a", linecolor=:blue)
ylabel!(L"Y_0")
xlabel!(L"z_0")
p2= plot(W, label="Variable w",linecolor=:red)
hline!(p2, [w], label="Fixed w",linecolor=:blue)
ylabel!(L"W_0")
xlabel!(L"z_0")
p3 = plot(z0grid./w0, ylabel=L"z_0/w_0", label="") # super flat 
plot(p1, p2, p3, layout = (3, 1), legend=:topleft)

# Zoom in
plot(z0grid[idx-3:idx+3], JJ[idx-3:idx+3], label="Hall: Fixed w and fixed a", linecolor=:cyan,linewidth=3)
xlabel!(L"z_0")
ylabel!(L"J_0")
plot!(z0grid[idx-3:idx+3], Y[idx-3:idx+3] .-w, label="Fixed w and variable a", linecolor=:red,legend=:topleft)
plot!(z0grid[idx-3:idx+3], YY[idx-3:idx+3].- W[idx-3:idx+3], label="Fixed a and variable w", linecolor=:blue,linewidth=3)
plot!(z0grid[idx-3:idx+3], Y[idx-3:idx+3] .-W[idx-3:idx+3], label="Bonus economy: Variable w and variable a", linecolor=:black,legend=:topleft)

# Compute slopes
JJ_B = slope(J,dz)     # Bonus 
JJ_H = slope(JJ,dz)    # Hall
ZZ   = slope(exp_z,dz) # slope of ∑ z_t (β(1-s))^(t-1)

# Check to compute ZZ analytically
exp_zz = zeros(length(z0grid)) 
@inbounds for (iz,z0) in enumerate(z0grid)
    @unpack zgrid, P_z, N_z = modd[iz].mod
    z0_idx  = findfirst(isequal(z0), zgrid)  # index of z0 on zgrid
    
    # initialize guesses
    v0     = zgrid./(1-β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
    # solve via simple value function iteration
    @inbounds while err > 10^-8 && iter < 500
        v0_new = zgrid./z0 + β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end
    exp_zz[iz]   = v0[z0_idx]
end

plot(z0grid[1:end-1], JJ_H, label="Hall: Fixed w and fixed a", linecolor=:cyan, linewidth=3,legend=:outerbottom, ylabel=L"dJ_0/ dz_0")
plot!(z0grid[1:end-1], JJ_B, label="Bonus economy: Variable w and variable a", linecolor=:black)
plot!(z0grid[1:end-1], a_opt*ZZ, label="Check 1")
plot!(z0grid, a_opt*exp_zz, label="Check 2")
xlabel!(L"z_0")

# try approximating total derivative of value function with the partial derivative
# take dz_t/d_z0 from analytical AR1 process
exp_az = zeros(length(z0grid)) 
@inbounds for (iz,z0) in enumerate(z0grid)
    @unpack zgrid, P_z, N_z = modd[iz].mod
    @unpack az = modd[iz]
    z0_idx  = findfirst(isequal(z0), zgrid)  # index of z0 on zgrid
    
    # initialize guesses
    v0     = zgrid./(1-β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    # solve via simple value function iteration
    @inbounds while err > 10^-8 && iter < 500
        v0_new = az.*zgrid/z0 + β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end
    exp_az[iz]   = v0[z0_idx]
end

# try central finite differences for this plot
JJ_B = (J[3:end] - J[1:end-2])/2dz # re-compute Bonus slopes
plot(z0grid, exp_az, label="Analytical Bonus w/ EVT", ylabel=L"dJ_0/ dz_0",linecolor=:red)
plot!(z0grid,  a_opt*exp_zz,label="Hall Numerical", linecolor=:cyan)
plot!(z0grid[2:end-1], JJ_B,label = "Numerical Bonus",linecolor=:black, legend=:topleft)


#= try with a continuous process (take expectations first but shouldn't matter)
TT      = 20000
N_sim   = 10000
exp_zz2 = zeros(length(z0grid))
Threads.@threads for iz = 1:length(z0grid)
    z0 = log(z0grid[iz])
    zt = zeros(N_sim, TT)
    zt[:,1] .= z0
    @inbounds for t=2:TT
        zt[:,t] = z0*(1-ρ) .+ ρ*zt[:,t-1] 
    end 
    zt = exp.(zt)
    ZT = mapreduce(t -> zt[:,t] *(β*(1-s))^(t-1), +, 1:TT)  # compute PV of Y_i
    exp_zz2[iz] = mean(ZT)
end

slope(exp_zz2,dz) 
exp_zz
=#
