## function to calculate the mse

function calcMSE(pest::Array{Float64,3}, ptrue::Array{Float64,3})
    ## dimension is T x N x N
    T = size(ptrue)[1]
    N = size(ptrue)[2]
    mse = Array{Float64}(undef, T)
    for t in 1:T
        mse[t] = 0.
        for i in 1:(N-1)
            for j in (i+1):N
                mse[t] += (pest[t,i,j] - ptrue[t,i,j]).^2
            end
        end
        mse[t] /= binomial(N,2)
    end
    return( mse )
end
