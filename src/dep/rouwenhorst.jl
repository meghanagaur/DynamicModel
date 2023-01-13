"""
Inputs: mu, rho, sig of an AR(1) process y(t) = rho y(t-1) + e(t)
mu is the unconditional mean of the process
rho is the AR coefficient
sig is the standard deviation of e(t)

Summary: The function discretizes the AR(1) process into an n equally
spaced state Markov chain with transition probabilities given by the 
Rouwenhorst (1995) method. States are in the set [mu - nu , mu + nu]

Outputs: Lambda, P, p
Lambda is a nx1 vector of equally spaced states centered around mu
P is the Markov transition matrix
pi is the invariant distribution of P
"""
function rouwenhorst(mu::T, rho::T, sig::T, n::U) where {T,U}

    nu = sqrt( ((n-1)/(1-rho^2)))*sig

    Lambda = collect(range(mu-nu, stop = mu+nu, length = n))

    p = (1+rho)/2
    q = p

    P1 = [ p 1-p; 1-q q ]

    if n == 2
        P = P1
    else
        for ii = 3:n
            zcol = zeros(ii-1,1)
            zrow = zcol'
            
            A = [ P1 zcol ; zrow 0 ]
            B = [ zcol P1 ; 0 zrow ]
            C = [ zrow 0 ; P1 zcol ]
            D = [ 0 zrow ; zcol P1 ]
            
            P1 = p*A + (1-p)*B + (1-q)*C + q*D
            P1[2:end-1,:] = P1[2:end-1,:]/2
        end
        P = P1
    end

    # Compute the invariant distribution 
    A           = P - Matrix(1.0I, n, n)
    A[:,end]   .= 1
    O           = zeros(1,n)
    O[end]      = 1
    pi          = (O*inv(A))
    pi          = pi/sum(pi)
    
    return Lambda, P, pi

end

