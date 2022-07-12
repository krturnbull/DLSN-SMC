#####################################################
## this code implement the GIRF for a latent space networks
# WITHOUT parameter estimation
#
# model is sim to Sewell + Chen 2016
#
# p(X_1i | theta ) \sim N(0, tau)
# p(X_ti | X_t-1,i ) \sim N( \phi X_t-1 i, \sigma)
#
# p(Y_t | X_t ) is logistic regression with link function \eta = \alpha - d( xit, xjt )
#
#####################################################
#
# Parameters are \alpha, \phi, \tau and \sigma
#
# P is dimension of latent space
# N is number of nodes
# T is number of time points
# M is number of particles
#
#####################################################

using Statistics
using StatsBase
using Distributions
using Plots
using ProgressMeter

include("resampling.jl") # different sampling mechanisms

############ this script contains:
# 1) data generation (from model and other `scenarios'
# 2) GIRF for theta known
# 3) SIR and APF for theta known
# 4) functions for assessing fit (MSE and ESS)
# 5) script for theta unknown
# 6) script for GIRF variants (DO NOT WORK)

################## types ############################

mutable struct netPrms #store parameters for network
    N :: Int64 # number of nodes
    T :: Int64 # number of time points
    P :: Int64 # latent dimension
    M :: Int64 # number of particles
    alpha :: Float64 #base rate of edges
    sig :: Float64 #variance in transition
tau :: Float64 #initial variance
phi :: Float64 # AR term \in (-1,1)
end

mutable struct PFcoords
    x :: Array{Float64,3} # latent coordinates, dim: (P,N,M)
    w :: Array{Float64,1} # weights, dim: M
    xHst :: Array{Float64,4} # history of xs, dim: (T,P,N,M)
    wHst :: Array{Float64,2} # history of ws, dim: (T, M)
    pOut :: Array{Float64,4} # weighted path as output, dim (T, N, N, 5)
    my :: Float64 # estimated of marginal for y
    eta :: Float64 # linear predictor
    ESS :: Array{Float64,1} # estimate of ESS, dim T
    ind :: Array{Int64,1} # resampling index vector, dim M
    uOld :: Array{Float64,1} # dim M
    uNew :: Array{Float64,1} # dim M
prob :: Array{Float64,3} # connection probs, dim: (N, N,M)
xis :: Array{Float64,1} # dim M
mu :: Array{Float64,3} # dim P,N,M
end

############### functions ###########################

function store!(PF::PFcoords, param::netPrms, t::Int64)
    # save output - weights and weighted probabilities
    # t is current time point

    # probability quantiles
    for i in 1:param.N-1, j in i+1:param.N
        PF.pOut[t,i,j,:] = quantile(PF.prob[i,j,:], Weights(PF.w), [.025, .25, .5, .75, .975] )
    end

    # weights
    PF.wHst[t,:] = PF.w

    # coordinates 
    PF.xHst[t,:,:,:] = PF.x
end

function createPFinpt(args::netPrms)
    return PFcoords( Array{Float64}(undef, args.P,args.N,args.M), Array{Float64}(undef, args.M), Array{Float64}(undef, args.T, args.P, args.N, args.M), Array{Float64}(undef, args.T,args.M), Array{Float64}(undef, args.T,args.N,args.N, 5), 0., 0., Array{Float64}(undef, args.T), Array{Int64}(undef, args.M), Array{Float64}(undef, args.M), Array{Float64}(undef, args.M), Array{Float64}(undef, args.N,args.N,args.M), Array{Float64}(undef, args.M), Array{Float64}(undef, args.P,args.N,args.M) )
end

#################################################################
##################### GENERATE DATA #############################
#################################################################

function calceta(x1::Array{Float64,1}, x2::Array{Float64,1}, prms::netPrms, metric::String)
    # calculate linear predictor
    if metric == "euclidean"
        ans = prms.alpha - sqrt( sum( ( x1 - x2 ).^2 ) )
    elseif metric == "dotprod" 
        ans = prms.alpha + (x1' * x2)[1]
    end
    return ans
end

function GenRWNet(prms::netPrms, metric::String, AR::String)
    # string can be "euclidean" or "dotprod"
    # AR can be "yes" or "no"

    # storage
    xdat = Array{Float64,3}(undef, prms.T + 1, prms.P, prms.N) 
    ydat = Array{Int64,3}(undef, prms.T, prms.N, prms.N)
    pdat = Array{Float64,3}(undef, prms.T, prms.N, prms.N)

    ## latent coords
    
    # t = 0
    if AR == "yes"
        xdat[1,:,:] = reshape( rand(Normal(0, sqrt( prms.sig^2/ (1 - prms.phi^2) ) ), prms.N*prms.P ), prms.P, prms.N ) 
    elseif AR == "no"
        xdat[1,:,:] = reshape( rand(Normal(0, prms.tau), prms.N*prms.P ), prms.P, prms.N ) 
    end
    
    # t > 1
    if AR == "yes"
        for t in 2:(prms.T + 1)
            xdat[t,:,:] = (prms.phi .* xdat[t-1,:,:]) + reshape( rand(Normal(0, prms.sig), prms.N*prms.P ), prms.P, prms.N )
        end
    elseif AR == "no"
        for t in 2:(prms.T + 1)
            xdat[t,:,:] = xdat[t-1,:,:] + reshape( rand(Normal(0, prms.sig), prms.N*prms.P ), prms.P, prms.N )
        end
    end

    ## adjacency matrices
    for t in 1:prms.T
        for i in 1:prms.N-1
            for j in i+1:prms.N
                pTmp = 1/( 1 + exp( - calceta(xdat[t+1,:,i], xdat[t+1,:,j], prms, metric) ) )
                pdat[t,i,j] = pTmp
                pdat[t,j,i] = pdat[t,i,j] #symmetric
                ydat[t,i,j] = rand( Bernoulli( pTmp ) )
                ydat[t,j,i] = ydat[t,i,j] #symmetric ties
            end
        end

        for i in 1:prms.N
            ydat[t,i,i] = 0 #don't allow self ties
            pdat[t,i,i] = 0
        end
    end

    return xdat, ydat, pdat   
end

#####################################################################
################## data from alternative scenarios ##################
#####################################################################

## stuff for generating networks from alternative scenarios
include("genaltnet.jl")

###################################################################
#################### USEFUL FUNCTIONS #############################
###################################################################

function normWeights!( PF::PFcoords, param::netPrms, addmarg::String )
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

function f1!( PF::PFcoords, param::netPrms, AR::String)
    # initial latent states
    # AR is "yes" or "no"
    if AR == "no"
        PF.x = reshape( rand( Normal(0, param.tau), param.N*param.P*param.M), param.P, param.N, param.M )
    elseif AR == "yes"
        PF.x = reshape( rand( Normal(0, sqrt( param.sig^2/(1-param.phi^2)) ), param.N*param.P*param.M), param.P, param.N, param.M )
    end
end

function f!( PF::PFcoords, param::netPrms, AR::String)
    # propagate latent states
    if AR == "no"
        PF.x += reshape( rand( Normal(0, param.sig), param.N*param.P*param.M),  param.P, param.N, param.M )
    elseif AR == "yes"
         PF.x = param.phi.*PF.x + reshape( rand( Normal(0, param.sig), param.N*param.P*param.M),  param.P, param.N, param.M )
    end
end

function f!( PF::PFcoords, param::netPrms, AR::String, delta::Float64)
    # propagate latent states with delta noise
    if AR == "no"
        PF.x += reshape( rand( Normal(0, sqrt(delta).*param.sig), param.N*param.P*param.M),  param.P, param.N, param.M )
    elseif AR == "yes" ## delta = 1/S
        PF.x = ((param.phi^delta) .* PF.x) + reshape( rand( Normal(0, sqrt( (1 - param.phi^(2*delta)) /( 1 - param.phi^2) ).*param.sig), param.N*param.P*param.M),  param.P, param.N, param.M )
    end
end

function g!( Ys::Array{Int64,3}, t::Int64, PF::PFcoords, param::netPrms, metric::String, likelihood::String)
    # evaluate likelihood over each particle

    if metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:param.M
                PF.w[m] = 0 #log(PF.w[m]) # previous weights
                for i in 1:param.N-1, j in i+1:param.N  
                    PF.eta = param.alpha + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                    PF.w[m] += Ys[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:param.M
                PF.w[m] = 0 #log(PF.w[m]) # previous weights
                for i in 1:param.N-1, j in i+1:param.N  
                    PF.eta = exp( param.alpha + (PF.x[:,i,m]' * PF.x[:,j,m])[1] )
                    PF.w[m] += -PF.eta + Ys[t,i,j]*log( PF.eta ) - log( factorial(Ys[t,i,j]) ) # log likelihood
                    PF.prob[i,j,m] = PF.eta # store rate parameter
                end               
            end
        end

    elseif metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:param.M
                PF.w[m] = 0 #log(PF.w[m])        
                for i in 1:param.N-1, j in i+1:param.N
                    #print( "\n",i, " ", j, "\n")
                    PF.eta = param.alpha - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                    PF.w[m] += Ys[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                    PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:param.M
                PF.w[m] = 0 #log(PF.w[m]) # previous weights
                for i in 1:param.N-1, j in i+1:param.N  
                    PF.eta = exp( param.alpha - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                    PF.w[m] += - PF.eta + Ys[t,i,j]*log( PF.eta ) - log( factorial(Ys[t,i,j]) ) # log likelihood
                    PF.prob[i,j,m] = PF.eta # store rate parameter
                end               
            end
        end
    end

end

####################################################################
########################### GIRF FUNCTIONS #########################
####################################################################

# also want to write a 'look ahead' u function
function u_B!(obs::Array{Int64,3}, PF::PFcoords, prms::netPrms, metric::String, likelihood::String, t::Int64, s::Int64, S::Int64, B::Int64)
    # approximating the predictive p( yt+1 | xt ) by propagating xt deterministically
    # here I am effectively evalulating log g( yt+1 | xt )
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
                     PF.prob[i,j,m] = PF.eta # store rate parameter
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
                 PF.prob[i,j,m] = PF.eta # store rate parameter
             end
         end
     end
end

function u!(obs::Array{Int64,3}, PF::PFcoords, prms::netPrms, metric::String, likelihood::String, t::Int64)
    # approximating the predictive p( yt+1 | xt ) by propagating xt deterministically
    # effectively evalulating log g( yt+1 | xt )

     if metric == "dotprod"

         if likelihood == "binomial"
             for m in 1:prms.M
                 PF.uNew[m] = 0 
                 for i in 1:prms.N-1, j in i+1:prms.N  
                     PF.eta = prms.alpha + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                     PF.uNew[m] += obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                     PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                 end               
             end

         elseif likelihood == "poisson"
             for m in 1:prms.M
                 PF.uNew[m] = 0 
                 for i in 1:prms.N-1, j in i+1:prms.N  
                     PF.eta = exp(prms.alpha + (PF.x[:,i,m]' * PF.x[:,j,m])[1])
                     PF.uNew[m] += -PF.eta + Ys[t,i,j]*log( PF.eta ) - log( factorial(Ys[t,i,j]) ) # log likelihood
                     PF.prob[i,j,m] = PF.eta # store rate parameter
                 end               
             end

         end

     elseif metric == "euclidean"

         if likelihood == "binomial"
             for m in 1:prms.M
                 PF.uNew[m] = 0
                 for i in 1:prms.N-1, j in i+1:prms.N
                     PF.eta = prms.alpha - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                     PF.uNew[m] += obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                     PF.prob[i,j,m] = 1/(1+exp(-PF.eta)) # store connection prob
                 end               
             end
         end

     elseif likelihood == "poisson"
         for m in 1:prms.M
             PF.uNew[m] = 0 
             for i in 1:prms.N-1, j in i+1:prms.N  
                 PF.eta = exp( prms.alpha - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                 PF.uNew[m] += -PF.eta + Ys[t,i,j]*log( PF.eta ) - log( factorial(Ys[t,i,j]) ) # log likelihood
                 PF.prob[i,j,m] = PF.eta # store rate parameter
             end               
         end
     end

end


function u_update!(obs::Array{Int64,3}, PF::PFcoords, prms::netPrms, metric::String, likelihood::String, t::Int64)
    # this function 'divides' uOld by previous observation

    if metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = prms.alpha + (PF.x[:,i,m]' * PF.x[:,j,m])[1]
                    PF.uOld[m] -= obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                end               
            end
            
        elseif likelihood == "poisson"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = exp( prms.alpha + (PF.x[:,i,m]' * PF.x[:,j,m])[1] )
                    PF.uOld[m] -= -PF.eta + Ys[t,i,j]*log( PF.eta ) - log( factorial(Ys[t,i,j]) ) ##
                end               
            end

        end

    elseif metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N
                    PF.eta = prms.alpha - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) )
                    PF.uOld[m] -= obs[t,i,j]*PF.eta -log(1+exp(PF.eta)) # log likelihood
                end               
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M
                for i in 1:prms.N-1, j in i+1:prms.N  
                    PF.eta = exp( prms.alpha - sqrt( sum( (PF.x[:,i,m]-PF.x[:,j,m]).^2 ) ) )
                    PF.uOld[m] -= -PF.eta + Ys[t,i,j]*log( PF.eta ) - log( factorial(Ys[t,i,j]) ) ##
                end               
            end

        end
    end
    
end

#############################################################
############################ GIRF ###########################
#############################################################

function GIRF!(data::Array{Int64,3}, inpt::PFcoords, args::netPrms, S::Int64, metric::String, likelihood::String, AR::String)
    # delta controls the intermediary steps, it is specified by S
    # delta = 1/S
    # AR is "yes" or "no"

    # set up progress bar
    print("\n")
    p = Progress(args.T, 1, "Running GIRF for network  ", 50)

    inpt.my = 0 # marginal likelihood
    delta = 1 / S # intermediary stepsize
    inpt.uOld .= 0 # to be sure (these are logs!)

    # initial x's
    f1!(inpt, args, AR) #sample initial states from prior

    for t in 1:args.T

        next!(p) #update progress bar
        
        # account for prev. obs (ie divide by g(yt|xt)
        if t > 1
            u_update!(data, inpt, args, metric, likelihood, t-1) # update log(uOld)
        end

        # intermediary loop
        for s in 1:S
            
            # sample states
            f!(inpt, args, AR, delta) #this adds on delta*noise and AR^delta

            # calc log(uNew)
            u!(data, inpt, args, metric, likelihood, t) #working way up to t^th obsg
            
            # update log(w)
            inpt.w = inpt.uNew - inpt.uOld
            
            # update marginal and normalise weights
            normWeights!(inpt, args, "yes")

            # resample
            inpt.ind = resampleSystematic( inpt.w )
            inpt.x = inpt.x[:,:,inpt.ind] #reshuffle states
            inpt.uOld = inpt.uNew[inpt.ind] #log uOld

        end

        # store output
        store!( inpt, args, t ) # this stores mean state and history of x and weights
    end
    print("\n")
end

function GIRF_B!(data::Array{Int64,3}, inpt::PFcoords, args::netPrms, S::Int64, B::Int64, metric::String, likelihood::String, AR::String)
    # delta controls the intermediary steps, it is specified by S
    # delta = 1/S
    # AR is "yes" or "no"

    # set up progress bar
    print("\n")
    p = Progress(args.T, 1, "Running GIRF for network  ", 50)

    inpt.my = 0 # marginal likelihood
    delta = 1 / S # intermediary stepsize
    inpt.uOld .= 0 # to be sure (these are logs!)

    # initial x's
    f1!(inpt, args, AR) #sample initial states from prior

    for t in 1:args.T

        next!(p) #update progress bar
        
        # account for prev. obs (ie divide by g(yt|xt)
        if t > 1
            u_update!(data, inpt, args, metric, likelihood, t-1) # update log(uOld)
        end

        # intermediary loop
        for s in 1:S
            
            # sample states
            f!(inpt, args, AR, delta) #this adds on delta*noise

            # calc log(uNew)
            u_B!(data, inpt, args, metric, likelihood, t, s, S, B) #working up to t^th obsg
            
            # update log(w)
            inpt.w = inpt.uNew - inpt.uOld
            
            # update marginal and normalise weights
            normWeights!(inpt, args, "yes")

            # resample
            inpt.ind = resampleSystematic( inpt.w )
            inpt.x = inpt.x[:,:,inpt.ind] #reshuffle states
            inpt.uOld = inpt.uNew[inpt.ind] #log uOld

        end

        # store output
        store!( inpt, args, t ) # this stores mean state and history of x and weights
    end
    print("\n")
end

####################################################################
########################### bootstrap filter #######################
####################################################################

function SIR!(data::Array{Int64,3}, PF::PFcoords, param::netPrms, metric::String, likelihood::String, AR::String )
    # standard bootstrap filter
    # data is dimension (T,N,N)
    # AR is "yes" or "no"
    
    PF.my = 0
    PF.w .= 1

    # set up progress bar
    print("\n")
    p = Progress(param.T, 1, "Running SIR for network  ", 50)

    # sample from prior
    f1!( PF, param, AR )
    PF.w .= 1/param.M #set equally weighted
    
    # loop over t
    for t in 1:param.T

        next!(p) #update progress bar
        
        ## propagate
        f!( PF, param, AR )
        
        ## weight
        g!( data, t, PF, param, metric, likelihood ) # calculates log weights
        normWeights!( PF, param, "yes" ) # finds normalised weights

        ## store
        store!( PF, param, t )

        ## resample
        PF.ind = resampleSystematic( PF.w )
        PF.x = PF.x[:,:,PF.ind]
        PF.w .= 1/param.M #reweight

    end
    print("\n")

    return PF
end

#####################################################################
############################### APF #################################
#####################################################################

function calcxi!(data::Array{Int64,3}, strg::PFcoords, prms::netPrms, metric::String, likelihood::String, t::Int64)
    # different cases for different metrics

    if metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M # calculate likelihood for each eta
                strg.xis[m] = log(strg.w[m])
                for i in 1:prms.N-1, j in i+1:prms.N  
                    strg.eta = prms.alpha - sqrt( sum((strg.mu[:,i,m]-strg.mu[:,j,m])).^2 )
                    strg.xis[m] += data[t,i,j]*strg.eta -log(1 + exp(strg.eta)) # log likelihood
                end              
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M # calculate likelihood for each eta
                strg.xis[m] = log(strg.w[m])
                for i in 1:prms.N-1, j in i+1:prms.N  
                    strg.eta = exp( prms.alpha - sqrt( sum((strg.mu[:,i,m]-strg.mu[:,j,m])).^2 ) )
                    strg.xis[m] += - strg.eta + data[t,i,j]*log(strg.eta) -log(factorial(data[t,i,j])) # log likelihood
                end              
            end
            
        end

    elseif metric == "dotprod"

        if likelihood == "binomial"
        for m in 1:prms.M # calculate likelihood for each eta
            strg.xis[m] = log(strg.w[m])
            for i in 1:prms.N-1, j in i+1:prms.N  
                strg.eta = prms.alpha + (strg.mu[:,i,m]' * strg.mu[:,j,m])[1]
                strg.xis[m] += data[t,i,j]*strg.eta -log(1 + exp(strg.eta)) # log likelihood
            end              
        end

        elseif likelihood == "poisson"
            for m in 1:prms.M # calculate likelihood for each eta
                strg.xis[m] = log(strg.w[m])
                for i in 1:prms.N-1, j in i+1:prms.N  
                    strg.eta = exp(prms.alpha + (strg.mu[:,i,m]' * strg.mu[:,j,m])[1])
                    strg.xis[m] += - strg.eta + data[t,i,j]*log(strg.eta) -log(factorial(data[t,i,j])) # log likelihood
                end              
            end
        end

    end
    
end

function normxis!( strg::PFcoords, prms::netPrms )
    # normalises log weights and calculates the marginal
       
    maxXi = maximum(strg.xis)
    strg.xis .-= maxXi
    strg.xis = exp.( strg.xis )
    strg.my += log( sum( strg.xis ) ) - log( prms.M ) + maxXi  # marginal 
    strg.xis /= sum(strg.xis) # normalise

end

function w_apf!(data::Array{Int64,3}, strg::PFcoords, prms::netPrms, metric::String, likelihood::String, t::Int64)

    if metric == "euclidean"

        if likelihood == "binomial"
            for m in 1:prms.M # calculate likelihood for each eta
                
                strg.w[m] = log(strg.w[m]) 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # numerator
                    strg.eta = prms.alpha - sqrt(sum((strg.x[:,i,m]-strg.x[:,j,m]).^2)) 
                    strg.w[m] += data[t,i,j]*strg.eta -log(1 + exp(strg.eta)) # log likelihood
                    strg.prob[i,j,m] = 1 ./ (1 + exp(-strg.eta))
                    # denomenator
                    strg.eta = prms.alpha - sqrt(sum((strg.mu[:,i,m]-strg.mu[:,j,m]).^2))
                    strg.w[m] -= data[t,i,j]*strg.eta -log(1 + exp(strg.eta)) # log likelihood
                end
                
            end

        elseif likelihood == "poisson"
            for m in 1:prms.M # calculate likelihood for each eta
                
                strg.w[m] = log(strg.w[m]) 
                for i in 1:prms.N-1, j in i+1:prms.N
                    # numerator
                    strg.eta = exp( prms.alpha - sqrt(sum((strg.x[:,i,m]-strg.x[:,j,m]).^2)) )
                    strg.w[m] += -strg.eta + data[t,i,j]*log(strg.eta) -log( factorial(data[t,i,j]) ) # log likelihood
                    strg.prob[i,j,m] = strg.eta
                    # denomenator
                    strg.eta = exp( prms.alpha - sqrt(sum((strg.mu[:,i,m]-strg.mu[:,j,m]).^2)) )
                    strg.w[m] -= -strg.eta + data[t,i,j]*log(strg.eta) -log( factorial(data[t,i,j]) ) # log likelihood
                end
                
            end

        end
        
    elseif metric == "dotprod"

        if likelihood == "binomial"
            for m in 1:prms.M # calculate likelihood for each eta
                
                strg.w[m] = log(strg.w[m])
                for i in 1:prms.N-1, j in i+1:prms.N
                    # numerator
                    strg.eta = prms.alpha + (strg.x[:,i,m]' * strg.x[:,j,m])[1]
                    strg.w[m] += data[t,i,j]*strg.eta -log(1 + exp(strg.eta)) # log likelihood
                    strg.prob[i,j,m] = 1 ./ (1 + exp(-strg.eta))
                    # denomenator
                    strg.eta = prms.alpha + (strg.mu[:,i,m]' * strg.mu[:,j,m])[1]
                    strg.w[m] -= data[t,i,j]*strg.eta -log(1 + exp(strg.eta)) # log likelihood
                end
                
            end

        elseif likelihood == "poisson"

            for m in 1:prms.M # calculate likelihood for each eta
                
                strg.w[m] = log(strg.w[m])
                for i in 1:prms.N-1, j in i+1:prms.N
                    # numerator
                    strg.eta = exp( prms.alpha + (strg.x[:,i,m]' * strg.x[:,j,m])[1] )
                    strg.w[m] += -strg.eta + data[t,i,j]*log(strg.eta) -log(factorial(data[t,i,j])) # log likelihood
                    strg.prob[i,j,m] = strg.eta
                    # denomenator
                    strg.eta = exp( prms.alpha + (strg.mu[:,i,m]' * strg.mu[:,j,m])[1] )
                    strg.w[m] -= -strg.eta + data[t,i,j]*log(strg.eta) -log(factorial(data[t,i,j])) # log likelihood
                end
                
            end 
        end
    end
    
end

function APF!(data::Array{Int64,3}, strg::PFcoords, prms::netPrms, metric::String, likelihood::String, AR::String )
    # apf filter
    # AR is "yes" or "no"
    
    strg.my = 0.
    strg.w .= 1. / prms.M

    # initialise
    # initial x's and alpha
    f1!(strg, prms, AR) #sample initial states from prior
    
    # loop over t
    for t in 1:prms.T
        
        ### populate eta ###
        if AR == "yes"
            # 'mean' is phi * (t-1)th states
            strg.mu = prms.phi .* copy( strg.x ) 
        elseif AR == "no"
            strg.mu = copy( strg.x ) # 'mean' is just (t-1)th states
        end
        calcxi!(data, strg, prms, metric, likelihood, t) #propagation probs
        normxis!( strg, prms ) # norm weights and get marginal

        ### resample ###
        strg.ind = resampleSystematic( strg.xis ) 
        strg.x = strg.x[:,:,strg.ind] 
        strg.mu = strg.mu[:,:,strg.ind]

        ### propagate ###
        f!( strg, prms, AR )

        ### weight ###
        w_apf!(data, strg, prms, metric, likelihood, t)

        # normalise weights
        normWeights!( strg, prms, "no" )

        # store
        store!( strg, prms, t )

        # sample
        strg.ind = resampleSystematic(strg.w )
        strg.x = strg.x[:,:,strg.ind]
        strg.w .= 1 ./ prms.M #reweight
        
    end

end

#################################################################
##################### functions to evaluate fit #################
#################################################################

function msecalc(pest::Array{Float64,3}, param::netPrms, ps::Array{Float64,3})

    MSE = Array{Float64}(undef, param.T)
    nPr = param.N * (param.N-1) / 2
    
    for t in 1:param.T
        MSE[t] = 0 #reset
        for i in 1:param.N-1, j in i+1:param.N
            MSE[t] += (pest[t,i,j] - ps[t,i,j]).^2
        end
        MSE[t] /= nPr
    end

    return MSE

end

function xtoprob(PF::PFcoords, param::netPrms)
    # post processing function to get probabilities from latent coords
    # returns the weighted probailities

    pest = Array{Float64}(undef, param.T, param.N, param.N)
    ptmp = Array{Float64}(undef, param.N, param.N, param.M) # 
    for t in 1:param.T
        for m in 1:param.M # loop over all particles to convert to probs
            for i in 1:param.N-1, j in i+1:param.N
                eta = param.alpha - sqrt( sum( (PF.xHst[t,:,i,m]-PF.xHst[t,:,j,m]).^2 ) )
                ptmp[i,j,m] = 1 ./ (1 + exp(-eta) )
            end
        end

        for i in 1:param.N-1, j in i+1:param.N
            pest[t,i,j] = sum( ptmp[i,j,:].*PF.wHst[t,:] ) ./ sum( PF.wHst[t,:] )
        end
    end
    
    return pest
end

#####################################################################
###################### PFs with THETA ESTIMATION ####################
#####################################################################

## stuff for parameter esimtation
include("netgirf_gradest.jl")

