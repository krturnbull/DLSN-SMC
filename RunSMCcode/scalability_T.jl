# this simulation study with explore the scalbility of the GIRF procedure
# will look at two cases: 1) N fixed, T grows 2) N grows, T fixed

using JLD
using Random
Random.seed!(123); # set random seed

## code for girf
include("../SMCcode/netgirf.jl")

## function for mse calc
include("../SMCcode/mse.jl")

###############################################################
############## test scalability of GIRF #######################
###############################################################

nIts = 20 ## for offline estimation

###############################################################
## CASE 1: N fixed, T grows
###############################################################

N = 20 # something reasonable
S = N
P = 2
M = 5000
Tvec = [50, 100, 500, 1000] 
alpha = 1.25 
sig = .2 
phi = .9
tau = .075
args = netPrms(N,Tvec[end],P,M,alpha, sig, tau, phi)
metric = "euclidean"
AR = "yes"
xs, ys, ps = GenRWNet( args, metric, AR ) # data is Tmax x N x N dim
## save this data
nameTmp = string("./simdat/data_", Tvec[end], "T_scalesims.jld" )
save(nameTmp, "xs", xs, "ys", ys, "ps", ps)

## set up storage -> all different lengths!
Tscale_online_ess = Array{Float64}(undef, 4, Tvec[end]); # times x length
Tscale_online_mse = Array{Float64}(undef, 4, Tvec[end]); # times x length
Tscale_online_time = Array{Float64}(undef, 4, 1); # times
Tscale_online_theta = Array{Float64}(undef, 4, 3, Tvec[end]+1); # times x thetas x length

## run filters (ONLINE)
for t_ind in 1:4

    ## apply filter
    argsTmp = netPrms(N,Tvec[t_ind],P,M,alpha, sig, tau, phi)
    girf = createPFcoords_prm_online( argsTmp );
    strt = time()
    GIRF_tht_online!(ys[1:Tvec[t_ind],:,:], girf, argsTmp, S, metric, "binomial", .95, .6, AR)
    time_tot = time() - strt
    ## store output
    Tscale_online_ess[t_ind,1:Tvec[t_ind]] = 1 ./ sum( girf.wHst.^2,dims=2)
    Tscale_online_mse[t_ind,1:Tvec[t_ind]] = calcMSE(girf.pOut[:,:,:,3], ps[1:Tvec[t_ind],:,:])
    Tscale_online_time[t_ind,1] = time_tot
    Tscale_online_theta[t_ind,1, 1:(Tvec[t_ind]+1)] = girf.alphest
    Tscale_online_theta[t_ind,2, 1:(Tvec[t_ind]+1)] = girf.lsigest
    Tscale_online_theta[t_ind,3, 1:(Tvec[t_ind]+1)] = girf.phitest

    print("\n Finished online, T = ", Tvec[t_ind], "\n")
    # save output
    save("./output/Tscale_online.jld", "Tscale_online_ess", Tscale_online_ess, "Tscale_online_mse", Tscale_online_mse, "Tscale_online_theta", Tscale_online_theta, "Tscale_online_time", Tscale_online_time)
end
