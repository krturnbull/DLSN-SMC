#####################################################
## this code implement the GIRF for a latent space networks
# WITH parameter estimation for ONLINE/OFFLINE case
#
# model is sim to Sewell + Chen 2016
#
# p(X_1i | theta ) \sim N(0, tau)
# p(X_ti | X_t-1,i ) \sim N( \phi X_t-1 i, \sigma)
#
# p(Y_t | X_t ) is logistic regression with link function \eta = \alpha - d( xit, xjt )
#
#####################################################

using StatsBase
using LightGraphs

using MultivariateStats
include("procrustes.jl")

################## types ##########################
## for theta = (alpha, log sigma, tilde phi)
mutable struct PFcoords_prm
x :: Array{Float64,3} # latent coordinates, dim: (P,N,M)
w :: Array{Float64,1} # weights, dim: M
xHst :: Array{Float64,4} # history of xs, dim: (T,P,N,M)
wHst :: Array{Float64,2} # history of ws, dim: (T, M)
pOut :: Array{Float64,4} # weighted path as output, dim (T, N, N, 5) QUANTILES
my :: Float64 # estimated of marginal for y
eta :: Float64 # linear predictor
ESS :: Array{Float64,1} # estimate of ESS, dim T
ind :: Array{Int64,1} # resampling index vector, dim M
uOld :: Array{Float64,1} # dim M
uNew :: Array{Float64,1} # dim M
prob :: Array{Float64,3} # connection probs, dim: (N, N,M)
alphest :: Array{Float64,1} # dimension: T+1
lsigest :: Array{Float64,1} # dimension: T+1
phitest :: Array{Float64,1} # dimension: T+1
score :: Array{Float64,2} # dimension: T+1 x 3
means :: Array{Float64,2} # dimension: 3 x M
ks :: Array{Int64,1} # dimension: M
end

###################################################################
#################### USEFUL FUNCTIONS #############################
###################################################################

function createPFcoords_prm_online(args::netPrms)
    return PFcoords_prm( Array{Float64}(undef, args.P,args.N,args.M),
                         Array{Float64}(undef, args.M),
                         Array{Float64}(undef, args.T, args.P, args.N, args.M),
                         Array{Float64}(undef, args.T,args.M),
                         Array{Float64}(undef, args.T,args.N,args.N, 5),
                         0., 0.,
                         Array{Float64}(undef, args.T),
    Array{Int64}(undef, args.M),
    Array{Float64}(undef, args.M),
    Array{Float64}(undef, args.M),
    Array{Float64}(undef, args.N,args.N,args.M),
    Array{Float64}(undef, args.T + 1),
    Array{Float64}(undef, args.T + 1),
    Array{Float64}(undef, args.T + 1),
    Array{Float64}(undef, args.T + 1, 3),
    Array{Float64}(undef, 3, args.M),
    Array{Int64}(undef, args.M)
    )
end

function createPFcoords_prm_offline(args::netPrms, nIts::Int64)
    return PFcoords_prm( Array{Float64}(undef, args.P,args.N,args.M),
                         Array{Float64}(undef, args.M),
                         Array{Float64}(undef, args.T, args.P, args.N, args.M),
                         Array{Float64}(undef, args.T,args.M),
                         Array{Float64}(undef, args.T,args.N,args.N, 5),
                         0., 0.,
                         Array{Float64}(undef, args.T),
    Array{Int64}(undef, args.M),
    Array{Float64}(undef, args.M),
    Array{Float64}(undef, args.M),
    Array{Float64}(undef, args.N,args.N,args.M),
    Array{Float64}(undef, nIts+1), # now this is number of iterations
    Array{Float64}(undef, nIts+1),
    Array{Float64}(undef, nIts+1),
    Array{Float64}(undef, nIts + 1, 3),
    Array{Float64}(undef, 3, args.M),
    Array{Int64}(undef, args.M)
    )
end

###################################################################
################## USEFUL PF FUNCTIONS ############################
###################################################################

function normWeights!( PF::PFcoords_prm, param::netPrms, addmarg::String )
    # normalises log weights and calculates the marginal
    
    maxW = maximum(PF.w)
    PF.w .-= maxW
    PF.w = exp.( PF.w )
    if addmarg == "yes"
        PF.my += log( sum( PF.w ) ) - log( param.M ) + maxW  # marginal for obs
    end
    PF.w ./= sum(PF.w) # normalise

    return PF
end

function store!(PF::PFcoords_prm, param::netPrms, t::Int64)
    # save output - weights and weighted probabilities
    # t is current time point

    # probability quantiles
    for i in 1:param.N-1, j in i+1:param.N
        PF.pOut[t,i,j,:] = quantile(PF.prob[i,j,:], Weights(PF.w), [.025, .25, .5, .75, .975])
    end

    # weights
    PF.wHst[t,:] = PF.w

    # coordinates 
    PF.xHst[t,:,:,:] = PF.x
end

function phi_from_pt(phi_tilde::Float64)
    ##return phi_tilde / sqrt(1 + phi_tilde^2) ## for |phi|<1
    return 1. / (1. + exp(-phi_tilde) ) 
end

###################################################################
################# PARAMETER INIT FUNCTIONS ########################
###################################################################

function init_alpha!(obs::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, metric::String, likelihood::String, AR::String)

    ## try several alphas and pick the one which gives a network with the most sensible density
    avec = -1.:.25:2 # take most likely value
    dns_tmp = fill( 0., length(avec)) # calculatw the density of simulated graphs

    if AR == "no"
        xstmp = reshape( rand( Normal(0, args.tau), args.N*args.P),  args.P, args.N)
        xstmp += reshape( rand( Normal(0, exp(inpt.lsigest[1])), args.N*args.P),  args.P, args.N)
    elseif AR == "yes"
        xstmp = reshape( rand( Normal(0, sqrt( exp(inpt.lsigest[1])^2 / (1 - phi^2)) ), args.N*args.P),  args.P, args.N)
        xstmp = phi.*xstmp + reshape( rand( Normal(0, exp(inpt.lsigest[1])), args.N*args.P),  args.P, args.N)
    end

    for a in 1:length(avec)

        if metric == "dotprod"
            if likelihood == "binomial"
                for i in 1:args.N-1, j in i+1:args.N
                    # note: alpha index is t-1 (start at alph_0)
                    eta = avec[a] + (xstmp[:,i]' * xstmp[:,j])[1]
                    dns_tmp[a] += rand(Binomial(1, 1 / (1 + exp(-eta))), 1)[1]
                end               
            elseif likelihood == "poisson"
                for i in 1:args.N-1, j in i+1:args.N
                    ##note: alpha index is t-1 (start at alph_0)
                    eta = exp( avec[a] + (xstmp[:,i]' * xstmp[:,j])[1] )
                    dns_tmp[a] += rand(Poisson( eta ), 1)[1]
                end               
            end       
        elseif metric == "euclidean"
            if likelihood == "binomial"
                for i in 1:args.N-1, j in i+1:args.N
                    eta = avec[a] - sqrt( sum( (xstmp[:,i]-xstmp[:,j]).^2 ) )
                    dns_tmp[a] += rand(Binomial(1, 1 / (1 + exp(-eta))), 1)[1]
                end               
            elseif likelihood == "poisson"
                for i in 1:args.N-1, j in i+1:args.N
                    ##note: alpha index is t-1 (start at alph_0)
                    eta = exp( avec[a] - sqrt( sum( (xstmp[:,i]-xstmp[:,j]).^2 ) ) )
                    dns_tmp[a] += rand(Poisson( eta ), 1)[1]
                end               
            end
        end
    end
    
    obsdens = 0.
    for t in 1:args.T
        for i in 1:args.N-1, j in i+1:args.N
            obsdens += obs[t,i,j]
        end
    end
    obsdens /= (args.T * binomial(N,2) )

    dns_tmp ./= binomial(N,2)
    
    print( "\n", dns_tmp, "  ", obsdens, "\n" )
    
    absdiff = abs.( dns_tmp .- 1.1*obsdens )
    inpt.alphest[1] = avec[ absdiff .== minimum(absdiff) ][1]

end

function inittheta!(obs::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, metric::String, likelihood::String, AR::String)

    ############### estimate tau and sigma ########################
    
    ## guess initial latent coordinates to estimate tau
    g1 = SimpleGraph(obs[1,:,:])
    g2 = SimpleGraph(obs[2,:,:])
    ## 1) create a distance matrix between all nodes
    ## 2) determine MDS estimate of these distances
    ## 3) set tau as variation
    dist = zeros(args.N, args.N, 2) # this is graph distance
    for i in 1:(args.N-1)
        for j in (i+1):args.N
            dist[i,j,1] = size(a_star(g1,i,j))[1]
            dist[i,j,2] = size(a_star(g2,i,j))[1]
            ## ensure symmetric
            dist[j,i,1] = dist[i,j,1]
            dist[j,i,2] = dist[i,j,2]
        end
    end

    ## correct '0' distances (not on diagonal)
    maxd = maximum(dist)
    for i in 1:(args.N-1)
        for j in (i+1):args.N
            if dist[i,j,1]==0
                dist[i,j,1] = maxd + .25
                dist[j,i,1] = dist[i,j,1]
            end
            if dist[i,j,2]==0
                dist[i,j,2] = maxd + .25
                dist[j,i,2] = dist[i,j,2]
            end
        end
    end

    ## add some noise for numerical stability
    dist = abs.( dist + reshape(rand( Normal(0, .001), args.N*args.N*2), args.N, args.N, 2) )

    obsdens = 0. ## calculate density of the observed network
    for t in 1:args.T
        for i in 1:args.N-1, j in i+1:args.N
            obsdens += (obs[t,i,j] > 0)
        end
    end
    obsdens /= (args.T * binomial(N,2) )

    ## mds 
    if metric == "euclidean"
        scl = (0.:.1:1.)[(1.:-.1:0. .== round(obsdens, digits=1))]
        mds1 = transform(MultivariateStats.fit(MDS, scl .* dist[:,:,1], maxoutdim=2, distances=true))
        mds2 = transform(MultivariateStats.fit(MDS, scl .* dist[:,:,2], maxoutdim=2, distances=true))
    elseif metric == "dotprod"
        scl = (0.:.1:1.)[(1.:-.1:0. .== round(obsdens, digits=1))]
        mds1 = transform(MultivariateStats.fit(MDS, scl .* dist[:,:,1], maxoutdim=2, distances=true))
        mds2 = transform(MultivariateStats.fit(MDS, scl.* dist[:,:,2], maxoutdim=2, distances=true))
    end
    ## need to procrustes map mds1 and mds2
    mds2 = procrustes( mds2, mds1 ) # mds1 is target matrix

    args.tau = sqrt(sum( mds1 .^ 2 ) / (args.N*args.P))

    #################### estimate phi ########################
    
    ## calculate phi given lsig and tau
    if AR == "yes"
        phi = .5 
        inpt.phitest[1] = log( phi / (1-phi) )
        inpt.lsigest[1] = log( sqrt( sum( (mds2 - phi.*mds1) .^ 2 ) / (args.N*args.P) ) )
    elseif AR == "no"
        inpt.lsigest[1] = log( sqrt( sum( (mds2 - mds1) .^ 2 ) / (args.N*args.P) ) )
    end

    print("\n sig init = ", exp(inpt.lsigest[1]), "\n")
    print("\n phi tilde init = ", inpt.phitest[1], "\n")
    ############### estimate alpha ########################
    
    ## initialise alpha based on tau and sig est
    init_alpha!(obs, inpt, args, metric, likelihood, AR)
    print("\n alpha init = ", inpt.alphest[1], "\n")

end

###################################################################
###################### GIRF FUNCTIONS #############################
###################################################################

## f1! 
function f1!( PF::PFcoords_prm, param::netPrms, AR::String)
    # initial latent states
    # AR is "yes" or "no"
    if AR == "no"
        PF.x = reshape( rand( Normal(0, param.tau), param.N*param.P*param.M), param.P, param.N, param.M )
    elseif AR == "yes"
         PF.x = reshape( rand( Normal(0, sqrt(exp(PF.lsigest[1])^2/(1-phi_from_pt(PF.phitest[1])^2))), param.N*param.P*param.M), param.P, param.N, param.M )
    end
end

## f!
function f!( PF::PFcoords_prm, param::netPrms, AR::String, delta::Float64, t::Int64)
    # propagate latent states with delta noise
    # AR is "yes" or "no"
    if AR == "no"
        PF.x += reshape( rand( Normal(0, sqrt(delta).*exp(PF.lsigest[t])), param.N*param.P*param.M),  param.P, param.N, param.M )
    elseif AR == "yes"
        phitmp = phi_from_pt(PF.phitest[t])
        PF.x = (phitmp^delta).*PF.x + reshape( rand( Normal(0, sqrt( (1 - phitmp^(2*delta)) / (1 - phitmp^2) )*exp(PF.lsigest[t])), param.N*param.P*param.M),  param.P, param.N, param.M )
    end
end

function f_offln!( PF::PFcoords_prm, param::netPrms, AR::String, delta::Float64, it::Int64)
    # propagate latent states with delta noise
    # AR is "yes" or "no"
    if AR == "no"
        PF.x += reshape( rand( Normal(0, sqrt(delta).*exp(PF.lsigest[it])), param.N*param.P*param.M),  param.P, param.N, param.M )
    elseif AR == "yes"
        phitmp = phi_from_pt(PF.phitest[it])
        PF.x = ( phitmp^delta ).*PF.x + reshape( rand( Normal(0, sqrt( ( 1 - phitmp^(2*delta) )/(1 - phitmp^2) ) * exp(PF.lsigest[it])), param.N*param.P*param.M),  param.P, param.N, param.M )
    end
end

## u!
function u!(obs::Array{Int64,3}, PF::PFcoords_prm, prms::netPrms, metric::String, likelihood::String, t::Int64)
    # approximating the predictive p( yt+1 | xt ) by propagating xt deterministically
    # effectively evalulating log g( yt+1 | xt )

    if metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = PF.alphest[t] + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                    PF.uNew[m] += obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = exp( PF.alphest[t] + (PF.x[:,i,m]' * PF.x[:,j,m])[1] )
                    PF.uNew[m] += -PF.eta + obs[t,i,j]*log(PF.eta) -log( factorial(obs[t,i,j]) ) # log likelihood
                    PF.prob[i,j,m] = PF.eta # store rate
                end               
            end

        end

    elseif metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M
                PF.uNew[m] = 0
                for i in 1:prms.N-1, j in i+1:prms.N
                    PF.eta = PF.alphest[t] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                    PF.uNew[m] += obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = exp( PF.alphest[t] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                    PF.uNew[m] += -PF.eta + obs[t,i,j]*log(PF.eta) -log( factorial(obs[t,i,j]) ) # log likelihood
                    PF.prob[i,j,m] = PF.eta # store rate
                end               
            end
        end
    end

end

function u_B!(obs::Array{Int64,3}, PF::PFcoords_prm, prms::netPrms, metric::String, likelihood::String, t::Int64, s::Int64, S::Int64, B::Int64)
    # approximating the predictive p( yt+1 | xt ) by propagating xt deterministically
    # effectively evalulating log g( yt+1 | xt )
    # s is current intermediary stage
    # S it total number of intermediary steps
    # B is the number of look ahead observations
    
    Bmax = min(B, prms.T - t) # max number of things to consider (for the upper end point)
    prps = zeros(Bmax)
    for i in 1:Bmax # i corresponds to the b index
        prps[i] = 1 - (i*S - s)/(S*( (t+i) - max(t+i-B,0) ) )
    end

    if metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = prms.alpha + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                    ## loop over B observations
                    for tind in 1:Bmax
                        PF.uNew[m] += prps[tind] * (obs[t+tind-1,i,j]*PF.eta -log(1+exp(PF.eta)) ) #ll
                    end
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"

            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = exp( prms.alpha + (PF.x[:,i,m]' * PF.x[:,j,m])[1] )
                    ## loop over B observations
                    for tind in 1:Bmax
                        PF.uNew[m] += prps[tind] * ( -PF.eta + obs[t+tind-1,i,j]*log(PF.eta) -log( factorial(obs[t+tind-1,i,j]) ) ) #ll
                    end
                    PF.prob[i,j,m] = PF.eta # store rate
                end               
            end
        end

    elseif metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M
                PF.uNew[m] = 0
                for i in 1:prms.N-1, j in i+1:prms.N
                    PF.eta = prms.alpha - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                    ## loop over B observations
                    for tind in 1:Bmax
                        PF.uNew[m] += prps[tind] * (obs[t+tind-1,i,j]*PF.eta -log(1+exp(PF.eta)) ) #ll
                    end
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = exp( prms.alpha - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                    ## loop over B observations
                    for tind in 1:Bmax
                        PF.uNew[m] += prps[tind] * ( -PF.eta + obs[t+tind-1,i,j]*log(PF.eta) -log( factorial(obs[t+tind-1,i,j]) ) ) #ll
                    end
                    PF.prob[i,j,m] = PF.eta # store rate
                end               
            end
        end

    end
end

function u_offln!(obs::Array{Int64,3}, PF::PFcoords_prm, prms::netPrms, metric::String, likelihood::String, t::Int64, it::Int64)
    # approximating the predictive p( yt+1 | xt ) by propagating xt deterministically
    # effectively evalulating log g( yt+1 | xt )

    if metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = PF.alphest[it] + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                    PF.uNew[m] += obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = exp( PF.alphest[it] + (PF.x[:,i,m]' * PF.x[:,j,m])[1] )
                    PF.uNew[m] += -PF.eta + obs[t,i,j]*log(PF.eta) -log(factorial(obs[t,i,j])) # log likelihood
                    PF.prob[i,j,m] = PF.eta # store rate
                end               
            end
            
        end

    elseif metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M
                PF.uNew[m] = 0
                for i in 1:prms.N-1, j in i+1:prms.N
                    PF.eta = PF.alphest[it] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                    PF.uNew[m] += obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = exp( PF.alphest[it] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                    PF.uNew[m] += -PF.eta + obs[t,i,j]*log(PF.eta) -log(factorial(obs[t,i,j])) # log likelihood
                    PF.prob[i,j,m] = PF.eta # store rate
                end               
            end
            
        end
    end

end

function u_offln_B!(obs::Array{Int64,3}, PF::PFcoords_prm, prms::netPrms, metric::String, likelihood::String, t::Int64, s::Int64, S::Int64, B::Int64, it::Int64)
    # approximating the predictive p( yt+1 | xt ) by propagating xt deterministically
    # effectively evalulating log g( yt+1 | xt )

    Bmax = min(B, prms.T - t) # max number of things to consider (for the upper end point)
    prps = zeros(Bmax)
    for i in 1:Bmax # i corresponds to the b index
        prps[i] = 1 - (i*S - s)/(S*( (t+i) - max(t+i-B,0) ) )
    end

    if metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = PF.alphest[it] + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                    ## loop over B observations
                    for tind in 1:Bmax
                        PF.uNew[m] += prps[tind] * (obs[t+tind-1,i,j]*PF.eta -log(1+exp(PF.eta)) ) #ll
                    end
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = exp( PF.alphest[it] + (PF.x[:,i,m]' * PF.x[:,j,m])[1] )
                    ## loop over B observations
                    for tind in 1:Bmax
                        PF.uNew[m] += prps[tind] * ( -PF.eta + obs[t+tind-1,i,j]*log(PF.eta) -log(factorial(obs[t+tind-1,i,j]) ) ) #ll
                    end
                    PF.prob[i,j,m] = PF.eta # store rate
                end               
            end

        end

    elseif metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M
                PF.uNew[m] = 0
                for i in 1:prms.N-1, j in i+1:prms.N
                    PF.eta = PF.alphest[it] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                    ## loop over B observations
                    for tind in 1:Bmax
                        PF.uNew[m] += prps[tind] * (obs[t+tind-1,i,j]*PF.eta -log(1+exp(PF.eta)) ) #ll
                    end
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                PF.uNew[m] = 0 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # note: alpha index is t-1 (start at alph_0)
                    PF.eta = exp( PF.alphest[it] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                    ## loop over B observations
                    for tind in 1:Bmax
                        PF.uNew[m] += prps[tind] * ( -PF.eta + obs[t+tind-1,i,j]*log(PF.eta) -log(factorial(obs[t+tind-1,i,j]) ) ) #ll
                    end
                    PF.prob[i,j,m] = PF.eta # store rate
                end               
            end

        end
    end

end

## u_update!
function u_update!(obs::Array{Int64,3}, PF::PFcoords_prm, prms::netPrms, metric::String, likelihood::String, t::Int64)
    # this function 'divides' uOld by previous observation

    if metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = PF.alphest[t] + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                    PF.uOld[m] -= obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = exp( PF.alphest[t] + (PF.x[:,i,m]' * PF.x[:,j,m])[1] )
                    PF.uOld[m] -= -PF.eta + obs[t,i,j]*log(PF.eta) -log(factorial(obs[t,i,j])) # log likelihood
                end               
            end
        end

    elseif metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N
                    PF.eta = PF.alphest[t] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                    PF.uOld[m] -= obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = exp( PF.alphest[t] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                    PF.uOld[m] -= -PF.eta + obs[t,i,j]*log(PF.eta) -log(factorial(obs[t,i,j])) # log likelihood
                end               
            end

        end
    end
    
end

function u_update_offln!(obs::Array{Int64,3}, PF::PFcoords_prm, prms::netPrms, metric::String, likelihood::String, t::Int64, it::Int64)
    # this function 'divides' uOld by previous observation

    if metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = PF.alphest[it] + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                    PF.uOld[m] -= obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = exp( PF.alphest[it] + (PF.x[:,i,m]' * PF.x[:,j,m])[1] )
                    PF.uOld[m] -= -PF.eta + obs[t,i,j]*log(PF.eta) -log(factorial(obs[t,i,j])) # log likelihood
                end               
            end

        end

    elseif metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N
                    PF.eta = PF.alphest[it] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                    PF.uOld[m] -= obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = exp( PF.alphest[it] - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                    PF.uOld[m] -= -PF.eta + obs[t,i,j]*log(PF.eta) -log(factorial(obs[t,i,j])) # log likelihood
                end               
            end
            
        end
    end
    
end

###################################################################
################## THETA UPDATE FUNCTIONS #########################
###################################################################

function update_alpha!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, metric::String, likelihood::String, lambda::Float64, t::Int64)
    if likelihood == "binomial"
        if metric == "euclidean"
            inpt.means[1,:] = lambda.*inpt.means[1,inpt.ks] .+ (1. - lambda) * inpt.score[t,1]
            npr = binomial(args.N,2)
            # add gradients
            for i in 1:(args.N-1)
                for j in (i+1):args.N
                    dtmp = sqrt.( sum( (inpt.x[:,i,:] - inpt.x[:,j,:]).^2, dims=1 )[1,:] ) 
                    ptmp = 1. ./ ( 1. .+ exp.( dtmp .- inpt.alphest[t]) )
                    inpt.means[1,:] += (data[t,i,j] .- ptmp) ./ npr
                end
            end
        elseif metric == "dotprod"
            inpt.means[1,:] = lambda.*inpt.means[1,inpt.ks] .+ (1. - lambda) * inpt.score[t,1]
            npr = binomial(args.N,2)
            # add gradients
            for i in 1:(args.N-1)
                for j in (i+1):args.N
                    dtmp = sum( (inpt.x[:,i,:] .* inpt.x[:,j,:]), dims=1 )[1,:]
                    ptmp = 1. ./ ( 1. .+ exp.( dtmp .- inpt.alphest[t]) )
                    inpt.means[1,:] += (data[t,i,j] .- ptmp) ./ npr
                end
            end 
        end
    elseif likelihood == "poisson"
        if metric == "euclidean"
            inpt.means[1,:] = lambda.*inpt.means[1,inpt.ks] .+ (1. - lambda) * inpt.score[t,1]
            npr = binomial(args.N,2)
            # add gradients
            for i in 1:(args.N-1)
                for j in (i+1):args.N
                    dtmp = sqrt.( sum( (inpt.x[:,i,:] - inpt.x[:,j,:]).^2, dims=1 )[1,:] ) 
                    inpt.means[1,:] += (data[t,i,j] .- exp.( inpt.alphest[t] .- dtmp ) ) ./ npr
                end
            end                
        elseif metric == "dotprod"
            inpt.means[1,:] = lambda.*inpt.means[1,inpt.ks] .+ (1. - lambda) * inpt.score[t,1]
            npr = binomial(args.N,2)
            # add gradients
            for i in 1:(args.N-1)
                for j in (i+1):args.N
                    dtmp = sum( (inpt.x[:,i,:] .* inpt.x[:,j,:]), dims=1 )[1,:]
                    inpt.means[1,:] += (data[t,i,j] .- exp.( inpt.alphest[t] .- dtmp ) ) ./ npr
                end
            end 
        end
    end
end

function update_alpha!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, metric::String, likelihood::String, lambda::Float64, t::Int64, it::Int64)
    if likelihood == "binomial"
        if metric == "euclidean"
            inpt.means[1,:] = lambda.*inpt.means[1,inpt.ks] .+ (1. - lambda) * inpt.score[it,1]
            npr = binomial(args.N,2)
            # add gradients
            for i in 1:(args.N-1)
                for j in (i+1):args.N
                    dtmp = sqrt.( sum( (inpt.x[:,i,:] - inpt.x[:,j,:]).^2, dims=1 )[1,:] ) 
                    ptmp = 1. ./ ( 1. .+ exp.( dtmp .- inpt.alphest[it]) )
                    inpt.means[1,:] += (data[t,i,j] .- ptmp) ./ npr
                end
            end
        elseif metric == "dotprod"
            inpt.means[1,:] = lambda.*inpt.means[1,inpt.ks] .+ (1. - lambda) * inpt.score[it,1]
            npr = binomial(args.N,2)
            # add gradients
            for i in 1:(args.N-1)
                for j in (i+1):args.N
                    dtmp = sum( (inpt.x[:,i,:] .* inpt.x[:,j,:]), dims=1 )[1,:]
                    ptmp = 1. ./ ( 1. .+ exp.( dtmp .- inpt.alphest[it]) )
                    inpt.means[1,:] += (data[t,i,j] .- ptmp) ./ npr
                end
            end 
        end
    elseif likelihood == "poisson"
        if metric == "euclidean"
            inpt.means[1,:] = lambda.*inpt.means[1,inpt.ks] .+ (1. - lambda) * inpt.score[it,1]
            npr = binomial(args.N,2)
            # add gradients
            for i in 1:(args.N-1)
                for j in (i+1):args.N
                    dtmp = sqrt.( sum( (inpt.x[:,i,:] - inpt.x[:,j,:]).^2, dims=1 )[1,:] ) 
                    inpt.means[1,:] += (data[t,i,j] .- exp.( inpt.alphest[it] .- dtmp ) ) ./ npr
                end
            end                
        elseif metric == "dotprod"
            inpt.means[1,:] = lambda.*inpt.means[1,inpt.ks] .+ (1. - lambda) * inpt.score[it,1]
            npr = binomial(args.N,2)
            # add gradients
            for i in 1:(args.N-1)
                for j in (i+1):args.N
                    dtmp = sum( (inpt.x[:,i,:] .* inpt.x[:,j,:]), dims=1 )[1,:]
                    inpt.means[1,:] += (data[t,i,j] .- exp.( inpt.alphest[it] .- dtmp ) ) ./ npr
                end
            end 
        end
    end
end

function update_sigma!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, xprev::Array{Float64,3}, lambda::Float64, t::Int64, AR::String)
    if AR == "yes"
        xdiff = inpt.x - phi_from_pt(inpt.phitest[t]) .* xprev[:,:,inpt.ks] 
    elseif AR == "no"
        xdiff = inpt.x - xprev[:,:,inpt.ks] 
    end
    inpt.means[2,:] = lambda.*inpt.means[2,inpt.ks] .+ (1. - lambda) * inpt.score[t,2] .- 1. .+ sum( xdiff.^2, dims=[1,2] )[1,1,:] ./ (args.N * args.P * exp(inpt.lsigest[t])^2)
end

function update_sigma!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, xprev::Array{Float64,3}, lambda::Float64, t::Int64, it::Int64, AR::String)
    if AR == "yes"
        xdiff = inpt.x - phi_from_pt(inpt.phitest[it]) .* xprev[:,:,inpt.ks] 
    elseif AR == "no"
        xdiff = inpt.x - xprev[:,:,inpt.ks] 
    end
    inpt.means[2,:] = lambda.*inpt.means[2,inpt.ks] .+ (1. - lambda) * inpt.score[it,2] .- 1. .+ sum( xdiff.^2, dims=[1,2] )[1,1,:] ./ (args.N * args.P * exp(inpt.lsigest[it])^2)
end

function update_phi!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, xprev::Array{Float64,3}, lambda::Float64, t::Int64, AR::String)
    # note: record phi tilde (phit) and log(sigma)
    if AR == "yes"
        xdiff = inpt.x - (phi_from_pt(inpt.phitest[t]) .* xprev[:,:,inpt.ks])
        inpt.means[3,:] = lambda .* inpt.means[3,inpt.ks] .+ (1. - lambda) * inpt.score[t,3] .+ (sum( (xprev[:,:,inpt.ks] .* xdiff ), dims=[1,2] )[1,1,:] .* (phi_from_pt(inpt.phitest[t]) .* (1- phi_from_pt(inpt.phitest[t])) )) ./ ( args.N * args.P * exp(inpt.lsigest[t])^2)
    elseif AR == "no"
        inpt.means[3,:] .= 0.
    end
end

function update_phi!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, xprev::Array{Float64,3}, lambda::Float64, t::Int64, it::Int64, AR::String)
    # note: record phi tilde (phit) and log(sigma)
    if AR == "yes"
        xdiff = inpt.x - (phi_from_pt(inpt.phitest[it]) .* xprev[:,:,inpt.ks])
        inpt.means[3,:] = lambda.*inpt.means[3,inpt.ks] .+ (1. - lambda) * inpt.score[it,3] .+ (sum( (xprev[:,:,inpt.ks] .* xdiff ), dims=[1,2] )[1,1,:] .* (phi_from_pt(inpt.phitest[it]) .* (1- phi_from_pt(inpt.phitest[it])) )) ./ ( args.N * args.P * exp(inpt.lsigest[it])^2) 
    elseif AR == "no"
        inpt.means[3,:] .= 0.
    end
end

###################################################################
##################### GIRF FUNCTION ###############################
###################################################################

## CONTAINS:
## ONLINE: B=1 lookahead, B>1 lookahead
## OffLINE: B=1 lookahead, B>1 lookahead

function GIRF_tht_online!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, S::Int64, metric::String, likelihood::String, lambda::Float64, scl::Float64, AR::String)
    # delta controls the intermediary steps, it is specified by S
    # delta = 1/S
    # lambda controls 'noise' in score vector estimations
    # scl is for gradient ascent steps
    # AR is "yes" or "no"

    # set up progress bar
    print("\n")
    p = Progress(args.T, 1, "Running GIRF, online prm est  ", 50)
    # init
    inpt.my = 0 # marginal likelihood
    delta = 1 / S # intermediary stepsize
    inpt.uOld .= 0 # to be sure (these are logs!)
    # initialise parameters (all at index 0)

    inittheta!(data, inpt, args, metric, likelihood, AR) # sets tau (prior), log sigma and alpha
    
    # initial x's
    f1!(inpt, args, AR) #sample initial states from prior
    inpt.score .= 0.
    inpt.means .= 0.
    for t in 1:args.T

        next!(p) #update progress bar        
        # account for prev. obs (ie divide by g(yt|xt)
        if t > 1
            u_update!(data, inpt, args, metric, likelihood, t-1) # update log(uOld)
        end
        # set the indices (these are for theta update)
        inpt.ks = 1:args.M
        # store xt-1
        xprev = copy(inpt.x) # make a copy of the m xs
        # intermediary loop
        for s in 1:S
            # sample states
            f!(inpt, args, AR, delta, t) #this adds on delta*noise
            # calc log(uNew)
            u!(data, inpt, args, metric, likelihood, t) #work up to t^th obsg
            # update log(w)
            inpt.w = inpt.uNew - inpt.uOld
            # update marginal and normalise weights
            normWeights!(inpt, args, "yes")
            # resample
            inpt.ind = resampleSystematic( inpt.w )
            inpt.x = inpt.x[:,:,inpt.ind] #reshuffle states
            inpt.uOld = inpt.uNew[inpt.ind] #log uOld
            # update ks 
            inpt.ks = inpt.ks[inpt.ind] # keeps track of ancestor
        end

        #####################################
        # update parameters (dimension 2 x M)
        #####################################

        # ALPHA
        update_alpha!(data, inpt, args, metric, likelihood, lambda, t)
        # SIGMA
        update_sigma!(data, inpt, args, xprev, lambda, t, AR)
        # PHI (sets means[3,:] to 0 if AR = no
        update_phi!(data, inpt, args, xprev, lambda, t, AR)

        # find the score
        for th in 1:3
            inpt.score[t+1,th] = sum(inpt.means[th,:] .* inpt.w) / sum(inpt.w)  # initial index is 0
        end

        # update parameters (difference in score, and gamma scaling)
        inpt.alphest[t+1] = inpt.alphest[t] + (t^(-scl))*(inpt.score[t+1,1] - inpt.score[t,1])
        inpt.lsigest[t+1] = inpt.lsigest[t] + (t^(-scl))*(inpt.score[t+1,2] - inpt.score[t,2])
        if AR == "yes"
            inpt.phitest[t+1] = inpt.phitest[t] + (t^(-scl))*(inpt.score[t+1,3] - inpt.score[t,3])
        end
                
        #####################################
        # store output
        #####################################
        store!( inpt, args, t ) # this stores mean state and history of x and weights
    end
    print("\n")
end

function GIRF_tht_online_B!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, S::Int64, B::Int64, metric::String, likelihood::String, lambda::Float64, scl::Float64, AR::String)
    # delta controls the intermediary steps, it is specified by S
    # delta = 1/S
    # lambda controls 'noise' in score vector estimations
    # scl is for gradient ascent steps
    # AR is "yes" or "no"

    # set up progress bar
    print("\n")
    p = Progress(args.T, 1, "Running GIRF, online prm est  ", 50)

    inpt.my = 0 # marginal likelihood
    delta = 1 / S # intermediary stepsize
    inpt.uOld .= 0 # to be sure (these are logs!)

    # initialise parameters (all at index 0)
    inittheta!(data, inpt, args, metric, likelihood, AR) # sets tau (prior), log sigma and alpha
    # initial x's
    f1!(inpt, args, AR) #sample initial states from prior
    
    inpt.score .= 0.
    inpt.means .= 0.

    for t in 1:args.T

        next!(p) #update progress bar
        
        # account for prev. obs (ie divide by g(yt|xt)
        if t > 1
            u_update!(data, inpt, args, metric, likelihood, t-1) # update log(uOld)
        end

        # set the indices (these are for theta update)
        inpt.ks = 1:args.M
        # store xt-1
        xprev = copy(inpt.x) # make a copy of the m xs
        
        # intermediary loop
        for s in 1:S
            
            # sample states
            f!(inpt, args, AR, delta, t) #this adds on delta*noise

            # calc log(uNew)
            u_B!(data, inpt, args, metric, likelihood, t, s, S, B) #work up to t^th obsg
            
            # update log(w)
            inpt.w = inpt.uNew - inpt.uOld
            
            # update marginal and normalise weights
            normWeights!(inpt, args, "yes")

            # resample
            inpt.ind = resampleSystematic( inpt.w )
            inpt.x = inpt.x[:,:,inpt.ind] #reshuffle states
            inpt.uOld = inpt.uNew[inpt.ind] #log uOld

            # update ks 
            inpt.ks = inpt.ks[inpt.ind] # keeps track of ancestor

        end

        #####################################
        # update parameters (dimension 2 x M)
        #####################################
        
        # ALPHA
        update_alpha!(data, inpt, args, metric, likelihood, lambda, t)
        # SIGMA
        update_sigma!(data, inpt, args, xprev, lambda, t, AR)
        # PHI (sets means[3,:] to 0 if AR = no
        update_phi!(data, inpt, args, xprev, lambda, t, AR)
        
        # find the score
        inpt.score[t+1,:] = inpt.means * inpt.w  # initial index is 0 

        # update parameters (difference in score, and gamma scaling)
        inpt.alphest[t+1] = inpt.alphest[t] + (t^(-scl))*(inpt.score[t+1,1] - inpt.score[t,1])
        inpt.lsigest[t+1] = inpt.lsigest[t] + (t^(-scl))*(inpt.score[t+1,2] - inpt.score[t,2])
        if AR == "yes"
            inpt.phitest[t+1] = inpt.phitest[t] + (t^(-scl))*(inpt.score[t+1,3] - inpt.score[t,3])
        end

        #####################################
        # store output
        #####################################
        store!( inpt, args, t ) # this stores mean state and history of x and weights
    end
    print("\n")
end

function GIRF_tht_offline!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, S::Int64, metric::String, likelihood::String, lambda::Float64, scl::Float64, nIts::Int64, AR::String)
    # delta controls the intermediary steps, it is specified by S
    # delta = 1/S
    # lambda controls 'noise' in score vector estimations
    # scl is for gradient ascent steps
    # AR is "yes" or "no"

    # set up progress bar
    print("\n")
    #p = Progress(args.T, 1, "Running GIRF offline prm est  ", nIts)

    # set initial parameters values
    delta = 1 / S # intermediary stepsize

    # initialise parameters (all at index 0)
    inittheta!(data, inpt, args, metric, likelihood, AR) # sets tau (prior), log sigma and alpha   

    for it in 1:nIts # interate over parameter values

        #################################################
        # run GIRF algorithm (overwrite output each time)
        #################################################
        
        # initial values for girf
        inpt.my = 0 # marginal likelihood
        inpt.uOld .= 0 # to be sure (these are logs!)
        inpt.score[it,:] .= 0.
        inpt.means .= 0.

        # initial x's
        f1!(inpt, args, AR) #sample initial states from prior        
        
        for t in 1:args.T
            
            # account for prev. obs (ie divide by g(yt|xt)
            if t > 1
                u_update_offln!(data, inpt, args, metric, likelihood, t-1, it) # update log(uOld)
            end

            # set the indices (these are for theta update)
            inpt.ks = 1:args.M
            # store xt-1
            xprev = copy(inpt.x) # make a copy of the m xs
            
            # intermediary loop
            for s in 1:S
                
                # sample states
                f_offln!(inpt, args, AR, delta, it) #this adds on delta*noise

                # calc log(uNew)
                u_offln!(data, inpt, args, metric, likelihood, t, it) #work up to t^th obsg
                
                # update log(w)
                inpt.w = inpt.uNew - inpt.uOld
                
                # update marginal and normalise weights
                normWeights!(inpt, args, "yes")

                # resample
                inpt.ind = resampleSystematic( inpt.w )
                inpt.x = inpt.x[:,:,inpt.ind] #reshuffle states
                inpt.uOld = inpt.uNew[inpt.ind] #log uOld

                # update ks 
                inpt.ks = inpt.ks[inpt.ind] # keeps track of ancestor

            end

            #####################################
            # update means (dimension 2 x M)
            #####################################
            
            # ALPHA
            update_alpha!(data, inpt, args, metric, likelihood, lambda, t, it)
            # SIGMA
            update_sigma!(data, inpt, args, xprev, lambda, t, it, AR)
            # PHI (sets means[3,:] to 0 if AR = no
            update_phi!(data, inpt, args, xprev, lambda, t, it, AR)

            # find the score
            inpt.score[it,1] = sum( inpt.means[1,:] .* inpt.w  ) 
            inpt.score[it,2] = sum( inpt.means[2,:] .* inpt.w  )
            inpt.score[it,3] = sum( inpt.means[3,:] .* inpt.w  )
            
            #####################################
            # store output
            #####################################
            store!( inpt, args, t )
        end
        
        #####################################
        # update parameters (scale score by T)
        #####################################
        inpt.alphest[it+1] = inpt.alphest[it] + ((it)^(-scl))*inpt.score[it,1]/args.T 
        inpt.lsigest[it+1] = inpt.lsigest[it] + ((it)^(-scl))*inpt.score[it,2]/args.T
        if AR == "yes"
            inpt.phitest[it+1] = inpt.phitest[it] + ((it)^(-scl))*inpt.score[it,3]/args.T
        end
        
        print( "\n alpha = ", inpt.alphest[it+1], "\n" )
        print( "\n sig = ", exp(inpt.lsigest[it+1]), "\n" )
        if AR == "yes"
            print("\n phi = ", phi_from_pt(inpt.phitest[it+1]), "\n")
        end
    end
    
    print("\n")
end


function GIRF_tht_offline_B!(data::Array{Int64,3}, inpt::PFcoords_prm, args::netPrms, S::Int64, B::Int64, metric::String, likelihood::String,  lambda::Float64, scl::Float64, nIts::Int64, AR::String)
    # delta controls the intermediary steps, it is specified by S
    # delta = 1/S
    # lambda controls 'noise' in score vector estimations
    # scl is for gradient ascent steps
    # AR is "yes" or "no"

    # set up progress bar
    print("\n")
    p = Progress(args.T, 1, "Running GIRF offline prm est  ", nIts)
    # set initial parameters values
    delta = 1 / S # intermediary stepsize

    # initialise parameters (all at index 0)
    inittheta!(data, inpt, args, metric, likelihood, AR) # sets tau (prior), log sigma and alpha
    for it in 1:nIts # interate over parameter values

        #################################################
        # run GIRF algorithm (overwrite output each time)
        #################################################
        
        # initial values for girf
        inpt.my = 0 # marginal likelihood
        inpt.uOld .= 0 # to be sure (these are logs!)
        inpt.score[it,:] .= 0.
        inpt.means .= 0.

        # initial x's
        f1!(inpt, args, AR) #sample initial states from prior        
        
        for t in 1:args.T

            # account for prev. obs (ie divide by g(yt|xt)
            if t > 1
                u_update_offln!(data, inpt, args, metric, likelihood, t-1, it) # update log(uOld)
            end
            # set the indices (these are for theta update)
            inpt.ks = 1:args.M
            # store xt-1
            xprev = copy(inpt.x) # make a copy of the m xs
            
            # intermediary loop
            for s in 1:S
                # sample states
                f_offln!(inpt, args, AR, delta, it) #this adds on delta*noise
                # calc log(uNew)
                u_offln_B!(data, inpt, args, metric, likelihood, t, s, S, B, it) #work up to t^th obsg
                # update log(w)
                inpt.w = inpt.uNew - inpt.uOld
                # update marginal and normalise weights
                normWeights!(inpt, args, "yes")
                # resample
                inpt.ind = resampleSystematic( inpt.w )
                inpt.x = inpt.x[:,:,inpt.ind] #reshuffle states
                inpt.uOld = inpt.uNew[inpt.ind] #log uOld
                # update ks 
                inpt.ks = inpt.ks[inpt.ind] # keeps track of ancestor
            end

            #####################################
            # update means (dimension 2 x M)
            #####################################
            
            # ALPHA
            update_alpha!(data, inpt, args, metric, likelihood, lambda, t, it)
            # SIGMA
            update_sigma!(data, inpt, args, xprev, lambda, t, it, AR)
            # PHI (sets means[3,:] to 0 if AR = no
            update_phi!(data, inpt, args, xprev, lambda, t, it, AR)
 
            # find the score
            inpt.score[it,1] = sum( inpt.means[1,:] .* inpt.w  ) 
            inpt.score[it,2] = sum( inpt.means[2,:] .* inpt.w  )
            inpt.score[it,3] = sum( inpt.means[3,:] .* inpt.w  )
            
            #####################################
            # store output
            #####################################
            store!( inpt, args, t )
        end

        #####################################
        # update parameters (scale score by T)
        #####################################
        inpt.alphest[it+1] = inpt.alphest[it] + ((it)^(-scl))*inpt.score[it,1]/args.T 
        inpt.lsigest[it+1] = inpt.lsigest[it] + ((it)^(-scl))*inpt.score[it,2]/args.T
        if AR == "yes"
            inpt.phitest[it+1] = inpt.phitest[it] + ((it)^(-scl))*inpt.score[it,3]/args.T
        end
        
    end
    
    print("\n")
end
