## code to fit GIRF on primary dataset

using Pkg
using RData
using JLD
using Random
Random.seed!(123); # set random seed

## code for girf
include("../SMCcode/netgirf.jl")
 
## function for mse calc
include("../SMCcode/mse.jl")

## function for predictive accuracy
include("../SMCcode/predictiveacc.jl")

###############################################################
############### read data and create storage ##################
###############################################################

## read in data
data = load("../data/primschool_network_class1.RData"); # aggregated onto 4 minute scale
data = data["dataset"];

names = data[1];
##times = data[2];
ys = convert( Array{Int64,3}, data[3]); # size T x N x N # run on day 2 observations
ys = 1 .* (ys .> 0)

nIts = 30 
T = size(ys)[1]
N = size(ys)[2]
S = 2*N
M = 5000
P = 2 

## sig, alpha and tau values don't matter
alpha = 1
sig = .15 
tau = .075
phi = .9
args = netPrms(N,T-1,P,M,alpha, sig, tau, phi)

ARvec = ["yes", "no"]

## create storage (for each case run AR=y and n, 1st index corresponds to this)
girf_primary_ess = Array{Float64}(undef, 2, 4, T-1); # have 4 cases, for each record ess, timetot, theta est, predictive prob
girf_primary_time = Array{Float64}(undef, 2, 4, 1); # have 4 cases, for each record ess, timetot, theta est, predictive prob
girf_primary_theta_online = Array{Float64}(undef, 2, 2, 3, T); # have 4 cases, for each record ess, timetot, theta est, predictive prob
girf_primary_theta_offline = Array{Float64}(undef, 2, 2, 3, nIts+1); # have 4 cases, for each record ess, timetot, theta est, predictive prob
girf_primary_predprob = Array{Float64}(undef, 2, 4, N, N, 5); # have 4 cases, for each record ess, timetot, theta est, predictive prob
girf_primary_probquant = Array{Float64}(undef, 2, 4, T-1, N, N, 5); # have 4 cases, for each record ess, timetot, theta est, predictive prob
###############################################################
############### run GIRF and store output #####################
###############################################################

############################
## run online, dotproduct
############################
for AR in 1:2
    girfinpt = createPFcoords_prm_online( args );
    strt = time()
    GIRF_tht_online!(ys[1:(T-1),:,:], girfinpt, args, S, "dotprod", "binomial", .95, 2/3, ARvec[AR])
    time_tot = time() - strt
    ## store output
    girf_primary_probquant[AR,1,:,:,:,:] = girfinpt.pOut # quantiles of probabilities
    girf_primary_ess[AR,1,:] = 1 ./ sum( girfinpt.wHst.^2,dims=2)
    girf_primary_time[AR,1,1] = time_tot
    girf_primary_theta_online[AR,1,1,:] = girfinpt.alphest
    girf_primary_theta_online[AR,1,2,:] = girfinpt.lsigest
    girf_primary_theta_online[AR,1,3,:] = girfinpt.phitest
    ## predictive accuracy:
    if ARvec[AR] == "yes"
        phiest = 1 / (1 + exp(-girfinpt.phitest[end]) )
        thtest = [girfinpt.alphest[end]; exp(girfinpt.lsigest[end]); phiest]
        girf_primary_predprob[AR,1,:,:,:] = calc_pred_prob(ys[T,:,:], args, thtest, girfinpt.xHst[T-1,:,:,:], "dotprod", "binomial", ARvec[AR])
    elseif ARvec[AR] == "no"
        thtest = [girfinpt.alphest[end]; exp(girfinpt.lsigest[end])]
        girf_primary_predprob[AR,1,:,:,:] = calc_pred_prob(ys[T,:,:], args, thtest, girfinpt.xHst[T-1,:,:,:], "dotprod", "binomial", ARvec[AR])
    end
end
############################w
## run online, euclidean
############################
for AR in 1:2
    girfinpt = createPFcoords_prm_online( args );
    strt = time()
    GIRF_tht_online!(ys[1:(T-1),:,:], girfinpt, args, S, "euclidean", "binomial", .95, 2/3,ARvec[AR])
    time_tot = time() - strt
    ## store output
    girf_primary_probquant[AR,2,:,:,:,:] = girfinpt.pOut # quantiles of probabilities
    girf_primary_ess[AR,2,:] = 1 ./ sum( girfinpt.wHst.^2,dims=2)
    girf_primary_time[AR,2,1] = time_tot
    girf_primary_theta_online[AR,2,1,:] = girfinpt.alphest
    girf_primary_theta_online[AR,2,2,:] = girfinpt.lsigest
    girf_primary_theta_online[AR,2,3,:] = girfinpt.phitest
    ## predictive accuracy:
    if ARvec[AR] == "yes"
        phiest = 1 / (1 + exp(-girfinpt.phitest[end]) )
        thtest = [girfinpt.alphest[end]; exp(girfinpt.lsigest[end]); phiest]
        girf_primary_predprob[AR,2,:,:,:] = calc_pred_prob(ys[T,:,:], args, thtest, girfinpt.xHst[T-1,:,:,:], "euclidean", "binomial", ARvec[AR])
    elseif ARvec[AR] == "no"
        thtest = [girfinpt.alphest[end]; exp(girfinpt.lsigest[end])]
        girf_primary_predprob[AR,2,:,:,:] = calc_pred_prob(ys[T,:,:], args, thtest, girfinpt.xHst[T-1,:,:,:], "euclidean", "binomial", ARvec[AR])
    end
end

## save online stuff
save("./output/alphinit_girf_primary_out.jld", "girf_primary_ess", girf_primary_ess, "girf_primary_time", girf_primary_time, "girf_primary_theta_offline", girf_primary_theta_offline, "girf_primary_theta_online", girf_primary_theta_online, "girf_primary_predprob", girf_primary_predprob, "girf_primary_probquant", girf_primary_probquant)
############################
## run offline, dotproduct
############################
for AR in 1:2
    girfinpt = createPFcoords_prm_offline( args, nIts );
    strt = time()
    GIRF_tht_offline!(ys[1:(T-1),:,:], girfinpt, args, S, "dotprod", "binomial", .95, 2/3, nIts, ARvec[AR])
    time_tot = time() - strt
    ## store output
    girf_primary_probquant[AR,3,:,:,:,:] = girfinpt.pOut # quantiles of probabilities
    girf_primary_ess[AR,3,:] = 1 ./ sum( girfinpt.wHst.^2,dims=2)
    girf_primary_time[AR,3,1] = time_tot
    girf_primary_theta_offline[AR,1,1,:] = girfinpt.alphest
    girf_primary_theta_offline[AR,1,2,:] = girfinpt.lsigest
    girf_primary_theta_offline[AR,1,3,:] = girfinpt.phitest
    ## predictive accuracy:
    if ARvec[AR] == "yes"
        phiest = 1 / (1 + exp(-girfinpt.phitest[end]) )
        thtest = [girfinpt.alphest[end]; exp(girfinpt.lsigest[end]); phiest]
        girf_primary_predprob[AR,3,:,:,:] = calc_pred_prob(ys[T,:,:], args, thtest, girfinpt.xHst[T-1,:,:,:], "dotprod", "binomial", ARvec[AR])
    elseif ARvec[AR] == "no"
        thtest = [girfinpt.alphest[end]; exp(girfinpt.lsigest[end])]
        girf_primary_predprob[AR,3,:,:,:] = calc_pred_prob(ys[T,:,:], args, thtest, girfinpt.xHst[T-1,:,:,:], "dotprod", "binomial", ARvec[AR])
    end
 end

############################
## run offline, euclidean
############################
for AR in 1:2
    girfinpt = createPFcoords_prm_offline( args, nIts );
    strt = time()
    GIRF_tht_offline!(ys[1:(T-1),:,:], girfinpt, args, S, "euclidean", "binomial", .95, 2/3, nIts, ARvec[AR])
    time_tot = time() - strt
    ## store output
    girf_primary_probquant[AR,4,:,:,:,:] = girfinpt.pOut # quantiles of probabilities
    girf_primary_ess[AR,4,:] = 1 ./ sum( girfinpt.wHst.^2,dims=2)
    girf_primary_time[AR,4,1] = time_tot
    girf_primary_theta_offline[AR,2,1,:] = girfinpt.alphest
    girf_primary_theta_offline[AR,2,2,:] = girfinpt.lsigest
    girf_primary_theta_offline[AR,2,3,:] = girfinpt.phitest
    ## predictive accuracy:
    if ARvec[AR] == "yes"
        phiest = 1 / (1 + exp(-girfinpt.phitest[end]) )
        thtest = [girfinpt.alphest[end]; exp(girfinpt.lsigest[end]); phiest]
        girf_primary_predprob[AR,4,:,:,:] = calc_pred_prob(ys[T,:,:], args, thtest, girfinpt.xHst[T-1,:,:,:], "euclidean", "binomial", ARvec[AR])
    elseif ARvec[AR] == "no"
        thtest = [girfinpt.alphest[end]; exp(girfinpt.lsigest[end])]
        girf_primary_predprob[AR,4,:,:,:] = calc_pred_prob(ys[T,:,:], args, thtest, girfinpt.xHst[T-1,:,:,:], "euclidean", "binomial", ARvec[AR])
    end
end
#########################
## save output

save("./output/alphinit_girf_primary_out.jld", "girf_primary_ess", girf_primary_ess, "girf_primary_time", girf_primary_time, "girf_primary_theta_offline", girf_primary_theta_offline, "girf_primary_theta_online", girf_primary_theta_online, "girf_primary_predprob", girf_primary_predprob, "girf_primary_probquant", girf_primary_probquant)
