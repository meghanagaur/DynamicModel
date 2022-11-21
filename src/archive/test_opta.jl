

modd = model(ε=0.5, γ=0.65, χ=0.0, z_1=model().zgrid[1])


# Solve for optimal effort a(z | z_1)
@inbounds for (iz,z) in enumerate(zgrid)

    if iz <11
        az[iz], yz[iz], a_flag[iz] = optA(z, modd, w_0)
    end
end


iz=11
z=zgrid[iz]
az[iz], yz[iz], a_flag[iz] = optA(z, modd, w_0)


aa         = fzero(x -> x - (x > a_min)*( max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2), 1 )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0)

x=aa
(x > a_min)*(x - max( z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10


using Plots

plot(x -> (x > a_min)*(x - max( z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, a_min )^(ε/(1+ε))) + (x <= a_min)*10^10,-0.1,2)


aa         = find_zeros(x -> x - max(z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), a_min, a_max)  
