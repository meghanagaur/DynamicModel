using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DataStructures, Distributions, ForwardDiff, Interpolations,
 LinearAlgebra, Parameters, Random, Roots, StatsBase, DynamicModel

# check to make sure we fall within bounds for all χ
zgrid = model().zgrid

# min χ
mod1 = solveModel(model(z0 = log(minimum(zgrid)), χ = 0.0))
mod2 = solveModel(model(z0 = log(maximum(zgrid)), χ = 0.0))

# max χ
mod3 = solveModel(model(z0 = log(minimum(zgrid)), χ = 0.5))
mod4 = solveModel(model(z0 = log(maximum(zgrid)), χ = 0.5))

# median χ
mod5 = solveModel(model(z0 = log(minimum(zgrid)), χ = 0.3))
mod6 = solveModel(model(z0 = log(maximum(zgrid)), χ = 0.3))

# median z_0
mod7 = solveModel(model(z0 = log(median(zgrid)), χ = 0.0))
mod8 = solveModel(model(z0 = log(median(zgrid)), χ = 0.3))
mod9 = solveModel(model(z0 = log(median(zgrid)), χ = 0.5))


 ## fix θ and look at how intermediates (Y, V, W) vary WITHOUT savings
function partial(θ_0; m =model(), max_iter2 = 1000, tol2 = 10^-8,  max_iter3 = 1000, tol3 = 10^-8)
 
    @unpack β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, procyclical, N_z, z0 = m 

    # Initialize default values and search parameters
    z0_idx = findfirst(isequal(z0), log.(zgrid))  # index of z0 on zgrid
    ω_0    = procyclical ? ω[z0_idx] : ω # unemployment value at z0
    α      = 0.25            # dampening parameter
    Y_0    = 0               # initalize Y_0 for export
    IR     = 0               # initalize IR for export
    w_0    = 0               # initialize initial wage constant for export

    # Initialize series
    az    = zeros(N_z)   # a(z|z_0)                         
    yz    = zeros(N_z)   # a(z|z_0)                         
    flag  = zeros(N_z)   # a(z|z_0)                         

    # Look for a fixed point in θ_0
    
    err2   = 10
    iter2  = 1      
    Y_lb   = κ/q(θ_0)                    # lower search bound
    Y_ub   = 50*κ/q(θ_0)                 # upper search bound
    Y_0    = ones(N_z)*(Y_lb + Y_ub)/2   # initial guess for Y(z)
    
    # Look for a fixed point in Y_0
    @inbounds while err2 > tol2 && iter2 < max_iter2   
        @inbounds for (iz,z) in enumerate(zgrid)
            az[iz], yz[iz], flag[iz] = optA(z, m, Y_0[z0_idx], θ_0)
        end
        Y_1    = yz + β*(1-s)*P_z*Y_0    
        err2   = maximum(abs.(Y_0 - Y_1))  # Error       
        α      = 0 #iter2 > 100 ? 0.75 : α 
        Y_0    = α*Y_0 + (1-α)*Y_1
        iter2 += 1
        #println(Y_0[z0_idx])
    end

    # Solve recursively for LHS of IR constraint
    w_0   = ψ*(Y_0[z0_idx] - κ/q(θ_0)) # time-0 earnings (constant)
    err3   = 10
    iter3  = 1  
    W_0    = β*s*ω
    flow   = -(1/2ψ)*(ψ*hp.(az)σ_η).^2 - h.(az)
    @inbounds while err3 > tol3 && iter3 < max_iter3
        W_1  = flow + P_z*(β*(1-s)*W_0 + β*s*ω)
        err3 = maximum(abs.(W_1 - W_0))
        α      = 0 #iter3 > 100 ? 0.75 : α 
        W_0    = α*W_0 + (1-α)*W_1
        #println(W_0[z0_idx])
        iter3 +=1
    end

    # Check IR constraint (must bind)
    IR_lhs = (1/ψ)*log(w_0) + W_0[z0_idx] 

    return (Y = Y_0[z0_idx], V = IR_lhs, ω_0 = ω_0, w_0 = w_0)
end

#= plot to see how present value for worker, output, and w0 change with θ
helpful for checking how we should update theta for convergence =#
tgrid = collect(0.0:0.5:10)
modd  = OrderedDict{Int64, Any}()
for i = 1:length(tgrid)
    modd[i] = partial.(tgrid[i])
end

Y     = [modd[i].Y for i = 1:length(tgrid)]
V     = [modd[i].V for i = 1:length(tgrid)]
ω0    = [modd[i].ω_0 for i = 1:length(tgrid)]
w0    = [modd[i].w_0 for i = 1:length(tgrid)]

p1 = plot(tgrid, Y , ylabel=L"Y_0", xlabel=L"\theta_0", label="")

@unpack ι = model()
qq(x) = 1/(1 + x^ ι)^(1/ι)
p2 = plot(qq,minimum(tgrid),maximum(tgrid), ylabel=L"q(\theta_0)",xlabel=L"\theta_0",label="")

p3 = plot(tgrid, V , label=L"V", legend=true)
plot!(p3, tgrid, ω0, label=L"\omega_0",xlabel=L"\theta_0")

p4 = plot(tgrid, w0 , ylabel=L"w_0", xlabel=L"\theta_0", legend=false)

plot(p1, p2, p3, p4,  layout = (2, 2))
#savefig(dir*"vary_theta.png")

## determine theta, as unemployment benefit scale χ varies
function tightness(bb)
    sol = solveModel(model(χ= bb),noisy=false)
    return (θ = sol.θ, w_0 = sol.w_0)
end

bgrid = 0.6:0.05:0.7

t1  = zeros(size(bgrid))
w01 = zeros(size(bgrid))

@inbounds for i = 1:length(bgrid)
    modd1  = tightness.(bgrid[i])
    t1[i]  = modd1.θ
    w01[i] = modd1.w_0
 end

p1=plot(bgrid, t1, ylabel=L"\theta",label="constant b")
ylabel!(p1,L"\theta")
xlabel!(p1,L"b")

# extra plot
p2=plot(bgrid, w01, ylabel=L"\theta",label="constant b")
ylabel!(p2,L"w_0")
xlabel!(p2,L"b")

plot(p1, p2, layout = (2, 1),legend=:topleft)

