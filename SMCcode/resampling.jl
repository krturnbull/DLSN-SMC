# Contains: Multinomial, residual, stratified and systematic resampling
#################################################################################

using Distributions

############################## Multinomial #####################################

function resampleMultinom(w::Array{Float64,1})

    M = length(w)
    Q = cumsum(w)

    Q[M] = 1 #check sums to 1

    index = Array{Int64}(undef,M)
    i = 1
    while i <= M
        sample = rand( Uniform() )

        j = 1
        while Q[j] < sample
            j += 1
        end
        index[i] = j
        
        i+=1    
    end
    return(index)
end

################################ Residual #######################################

function resampleResid(w::Array{Float64})

    M = length(w)
    index = Array{Int64}(undef,M)

    Ns = floor(M .* w) ###REPLACE floor with fld????

    R = sum(Ns)

    Mrdn = M - R

    Ws = (M*w - floor(M*w) ) ./ Mrdn

    i = 1

    for j in 1:M
        if Ns[j] > 0
            for k in 1: Ns[j]
                index[i] = j
                i += 1
            end
        end
    end

    Q = cumsum(Ws)
    Q[M] = 1

    while i <= M
        sample = rand( Uniform() )
        j = 1
        while Q[j] < sample
            j += 1
        end
        index[i] = j
        i += 1
    end
    return(index)
end

################################ Systematic ##################################

function resampleSystematic(w::Array{Float64})

    N = length(w)
    Q = cumsum(w)
    T = Array{Float64}(undef,N+1)
    index = Array{Int64}(undef,N)

    T[1:N] = range(0, stop=1-1/N, length=N) .+ rand( Uniform() )/N
    T[N+1] = 1

    i=1
    j=1

    while i <= N
        if T[i] < Q[j]
            index[i] = j
            i += 1
        else
            j += 1
        end
    end
    return(index)    
end

####### stratified #####

function resampleStratified(w::Array{Float64,1})

    N = length(w)
    Q = cumsum(w)
    T = Array{Float64}(undef,N+1)
    index = Array{Int64}(undef,N)

    for i in 1:N
        T[i] = rand( Uniform() )/N + (i-1)/N
    end
    T[N+1] = 1

    i=1
    j=1

    while i<=N
        
        if T[i]<Q[j]
            index[i]=j
            i+=1
        else
            j+=1           
        end
    end
    
    return index
end
