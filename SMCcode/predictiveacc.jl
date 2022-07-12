## function to take a fitted PF and calculate predictive accuracy
using Statistics

function calc_pred_prob(yT::Array{Int64,2}, param::netPrms, theta_est::Array{Float64}, xs_fnl::Array{Float64,3}, metric::String, likelihood::String, AR::String)
    ## yT is NxN final observation matrix
    ## theta estimate is vector (alpha, sigma, phi) (if ar == "yes")
    ## xs_final is P x N x M at time T

    ## propagate
    if AR == "no"
        xs_fnl += reshape( rand( Normal(0, theta_est[2]), param.N*param.P*param.M), param.P, param.N, param.M )
    elseif AR == "yes"
        xs_fnl = (theta_est[3].*xs_fnl) + reshape( rand( Normal(0, theta_est[2]), param.N*param.P*param.M), param.P, param.N, param.M )
    end
        
    ## calculate pij
    probs = zeros(param.N, param.N, param.M)
    if likelihood == "binomial"
        for m in 1:param.M
            for i in 1:(param.N-1)
                for j in (i+1):param.N
                    if metric == "euclidean"
                        eta = theta_est[1] - sqrt( sum( (xs_fnl[:,i,m]-xs_fnl[:,j,m]).^2 ) )
                    elseif metric == "dotprod"
                        eta = theta_est[1] + sum( xs_fnl[:,i,m].*xs_fnl[:,j,m] )
                    end
                    probs[i,j,m] = 1 / (1 + exp( - eta ) )
                    probs[j,i,m] = probs[i,j,m]
                end
            end
        end
        
    elseif likelihood == "poisson"

        for m in 1:param.M
            for i in 1:(param.N-1)
                for j in (i+1):param.N
                    if metric == "euclidean"
                        eta = exp( theta_est[1] - sqrt( sum( (xs_fnl[:,i,m]-xs_fnl[:,j,m]).^2 ) ) )
                    elseif metric == "dotprod"
                        eta = exp( theta_est[1] + sum( xs_fnl[:,i,m].*xs_fnl[:,j,m] ) )
                    end
                    probs[i,j,m] = eta
                    probs[j,i,m] = probs[i,j,m]
                end
            end
        end
        
    end


    ## get quantiles
    probest = zeros( param.N, param.N, 5)
    for i in 1:(param.N-1)
        for j in (i+1):param.N
            probest[i,j,:] = quantile( probs[i,j,:], [.025, .25, .5, .75, .975] )
            probest[j,i,:] = probest[i,j,:]
        end
    end

    return probest
end
