# this simulation study with explore the scalbility of the GIRF procedure
# will look at two cases: 1) N fixed, T grows 2) N grows, T fixed

using JLD
using Random
Random.seed!(12653); # set random seed

## code for girf
include("../SMCcode/netgirf.jl")

## function for mse calc
include("../SMCcode/mse.jl")

###############################################################
############## test scalability of GIRF #######################
###############################################################

nIts = 20 ## for offline estimation

###############################################################
## CASE 2: N grows, T fixed
###############################################################
T = 25
Nvec = [50, 75, 100] 
Svec = [.25, .5, 1.]

P = 2
M = 5000
alpha = 1. 
sig = .2
tau = .075
phi = .9
AR = "yes"
metric = "euclidean"

## create the data and save it
for n in 1:3
    args = netPrms(Nvec[n],T,P,M,alpha, sig, tau, phi)
    metric = "euclidean"
    xs, ys, ps = GenRWNet( args, metric, AR ) # data is Tmax x N x N dim
    nameTmp = string("./simdat/data_", Nvec[n], "N_scalesims.jld", )
    save(nameTmp, "xs", xs, "ys", ys, "ps", ps)
end

## set up storage
Nscale_online_ess = Array{Float64}(undef, 3, 3, T); # ns x ss x time
Nscale_online_mse = Array{Float64}(undef, 3, 3, T); # ns x ss x time
Nscale_online_time = Array{Float64}(undef, 3, 3, 1); # ns x ss xtime
Nscale_online_theta = Array{Float64}(undef, 3, 3,  3, T+1); # ns x ss x theta x time

## run filters (ONLINE)
for n in 1:3

    ## read in data
    nameTmp = string("./simdat/data_", Nvec[n], "N_scalesims.jld", )
    xs = load(nameTmp, "xs") # dim T x P x N
    ys = load(nameTmp, "ys") # dim T x N x N
    ps = load(nameTmp, "ps") # dim T x N x N

    for s in 1:3
        ## apply filter
        argsTmp = netPrms(Nvec[n],T,P,M,alpha, sig, tau, phi)
        girf = createPFcoords_prm_online( argsTmp );
        sTmp = convert( Int64, ceil(Nvec[n] * 2 * Svec[s]) )
        strt = time()
        GIRF_tht_online!(ys, girf, argsTmp, sTmp, metric, "binomial", .95, 2/3, AR)
        time_tot = time() - strt
        ## store output
        Nscale_online_ess[n,s,:] = 1 ./ sum( girf.wHst.^2,dims=2)
        Nscale_online_mse[n,s,:] = calcMSE(girf.pOut[:,:,:,3], ps)
        Nscale_online_time[n,s,1] = time_tot
        Nscale_online_theta[n,s,1,:] = girf.alphest
        Nscale_online_theta[n,s,2,:] = girf.lsigest
        Nscale_online_theta[n,s,3,:] = girf.phitest

        print("\n Finished online, N = ", n, " S = ", s, "\n")
    end
    
    # save output
    save("./output/Nscale_online.jld", "Nscale_online_ess", Nscale_online_ess, "Nscale_online_mse", Nscale_online_mse, "Nscale_online_time", Nscale_online_time, "Nscale_online_theta", Nscale_online_theta)
end
