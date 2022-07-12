## script to demonstrate smac sampler
include("../SMCcode/netgirf.jl") # call functions

## set model parameters
N = 10 # number of nodes
P = 2 # latent dim
T = 30 # number of time points
M = 5000 # number of particles
S = 5 # number of intermediary states
alpha = 1.25
sig = .2
tau = .2
phi = .9
args = netPrms(N,T,P,M,alpha, sig, tau, phi)
AR = "yes" # autoregressive (yes) or not (no) for latent states

# generate data
metric = "euclidean"
xs, ys, ps = GenRWNet( args, metric, AR ); # data is T x N x N dim

# run girf without parameter estimation
girf = createPFinpt( args );
GIRF!(ys, girf, args, S, metric, "binomial", AR) 

# run girf with online parameter estimation
girf_online = createPFcoords_prm_online( args );
GIRF_tht_online!(ys, girf_online, args, S, metric, "binomial", .95, 0.6, AR)

# run girf with parameter offline estimation
nIts = 10
girf = createPFcoords_prm_offline( args, nIts );
GIRF_tht_offline!(ys, girf, args, S, metric, "binomial", .95, .6, nIts, AR)

