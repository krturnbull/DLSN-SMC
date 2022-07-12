# for making appendix D plots

using JLD
using Random
Random.seed!(123); # set random seed

## code for girf
include("../SMCcode/netgirf.jl")

## function for mse calc
include("../SMCcode/mse.jl")

###############################################################
####################### RUN GIRF ##############################
###############################################################

## set parameters
Nvec = [5,10,20,30,50] # should be enough cases
T = 30
P = 2
M = 10000 
alpha = 1.2
sig = .2
tau = NaN 
phi = .9 
AR = "yes" process is autoregressive
Svec = [.25, .5, .75, 1.] # number of intermediary steps as % of state space dimension
Lvec = [0,1,2,3] # number of look ahead steps
## create storage
girf_mat = Array{Float64}(undef, T, length(Svec), length(Nvec), length(Lvec), 2) # last index: 1 - ess, 2 - mse

## run
for nval in 1:length(Nvec)

    # get the data
    ## define parameters
    N = Nvec[nval] #number of nodes
    args = netPrms(N,T,P,M,alpha, sig, tau, phi)
    metric = "euclidean"
    ## read in the data
    name = string("./simdat/data_", Nvec[nval], "N_euc.jld", )
    xs = load(name, "xs") # dim T x P x N
    ys = load(name, "ys") # dim T x N x N
    ps = load(name, "ps") # dim T x N x N

    Stmp = convert(Array{Int64}, ceil.(Svec * 2 * N))
    # loop over:
    # 1) choices of S
    # 2) choices of L
    for sval in 1:length(Svec)
        for lval in 1:length(Lvec)

            ## run girf
            if (Lvec[lval] == 0)
                ## no look ahead               
                girf = createPFinpt( args );
                GIRF!(ys, girf, args, Stmp[sval], metric, "binomial", AR) 
            else
                ## look ahead
                girf = createPFinpt( args );
                GIRF_B!(ys, girf, args, Stmp[sval],Lvec[lval],metric, "binomial", AR) 
            end
            
            # record ess
            girf_mat[:, sval, nval, lval, 1] = 1 ./ sum( girf.wHst.^2,dims=2)
            # record mse (in prob)
            girf_mat[:, sval, nval, lval, 2] = calcMSE(girf.pOut[:,:,:,3], ps) # dimension pOut is T x N x N x 5 (quantiles .025, .25, .5, .75, .975)
            
            print(string("\n finished girf with N = ", N, " S = ", Stmp[sval], " L = ", Lvec[lval], "\n"))
        end
    end

    ## save for plotting in R (can get results earlier saving here)
    save("./output/girfmat.jld", "girfmat", girf_mat)
end
