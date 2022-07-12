using JLD
using Random

Random.seed!(123); # set random seed

## code for girf
include("../SMCcode/netgirf.jl")

## function for mse calc
include("../SMCcode/mse.jl")

###############################################################
####### want to test GIRF for different scenarios #############
###############################################################

## for the following cases test different Ls
## case 1: "clust_circ"
## case 2: "clust_line"
## case 3: "var_dens"

N = 20
S = 10
T = 30
P = 2
M = 1000
alpha = 1.5
sig = .15
tau = .075
args = netPrms(N,T,P,M,alpha, sig, tau)
metric = "euclidean"

Svec = [5, 10, 15, 20]
Lvec = [0,1,2,3] # number of look ahead steps
## create storage
girf_mat = Array{Float64}(undef, T, 3, length(Svec), length(Lvec), 2) # last index: 1 - ess, 2 - mse


# check altnerative scenarios work
casevec = ["clust_circ", "clust_line", "var_dens"]

for case in 1:length(casevec)

    ## generate data
    xs, ys, ps = gen_alt_coords(N, T, casevec[case], 0.2);
    sig1 = sqrt( var(xs[5,:,:] - xs[4,:,:]) ) 
    sig2 = sqrt( var(xs[10,:,:] - xs[9,:,:]) )
    args.sig = ( sig1 + sig2 )/2
    
    for sval in 1:length(Svec)
        for lval in 1:length(Lvec)
            
            ## run girf
            if (Lvec[lval] == 0)
                ## no look ahead               
                girf = createPFinpt( args );
                GIRF!(ys, girf, args, Svec[sval], metric) 
            else
                ## look ahead
                girf = createPFinpt( args );
                GIRF_B!(ys, girf, args, Svec[sval],Lvec[lval],metric) 
            end
            
            # record ess
            girf_mat[:, case, sval, lval, 1] = 1 ./ sum( girf.wHst.^2,dims=2)
            # record mse (in prob)
            girf_mat[:, case, sval, lval, 2] = calcMSE(girf.pOut, ps)

        end
    end
end

## save for plotting in R
save("./output/girf_cases_test.jld", "girfmat", girf_mat)
