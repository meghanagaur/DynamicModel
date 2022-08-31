"""
Simulate N {z_t} paths, given transition prob. matrix P_z and 
productivity grid, and return probability of each path. 
Simulation begins at z0_idx of zgrid.
"""
function simulateProd(P_z, zgrid, T; N = 10000, z0_idx = median(1:length(zgrid)), seed = 211, set_seed = true)
   
    if set_seed == true
        Random.seed!(seed)
    end
    
    sim      = rand(N, T)            # N X T - draw uniform numbers in (0,1)
    zt       = zeros(Int64, N, T)    # N X T - index on zgrid
    zt[:,1] .= floor(Int64, z0_idx)  # Set z0 with valid index
    CDF      = cumsum(P_z, dims = 2) # CDF for transition probs. given initial state
    probs    = ones(N)               # N x 1 - probability of a given sequence

    @inbounds for t = 2:T
        Threads.@threads for i = 1:N
            zt[i, t]  = findfirst(x-> x >=  sim[i,t], CDF[zt[i,t-1], :]) 
            probs[i]  =  P_z[zt[i,t-1], zt[i,t]]*probs[i]
        end
    end

    return zgrid[zt], probs, zt
end