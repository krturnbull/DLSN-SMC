## code to simulate networks from alternative scenarios
## have:
## 1) communities 1 -> 2 -> 1 (each travels in a circle)
## 2) communities 1 -> 2 (each travels on a line)
## 3) constant alpha, but varying density

function genclustcirc!(xdat::Array{Float64,3}, N::Int64, T::Int64, q::Float64)

    # means for cluster 1
    theta = range(0.05*pi, stop=0.95*pi, length=T)
    rs = 2.5.* ( sin.( theta ) ).^2
    mus1_x = rs .* cos.(theta)
    mus1_y = rs .* sin.(theta)

    # means for cluster 2
    theta = range(2.05*pi, stop=1.95*pi, length=T)
    rs = 2.5.* ( sin.( theta ) ).^2
    mus2_x = rs .* cos.(theta)
    mus2_y = rs .* sin.(theta)

    # generate trajectories
    Nhlf = convert( Int64, N/2 )
    xdat[1,:,:] = reshape(rand( Normal(0, .1), 2*N ), 2, N )
    for t in 2:(T+1)
        # cluster 1
        xdat[t,1,1:Nhlf] = q .* xdat[t-1,1,1:Nhlf] .+ (1-q)*mus1_x[t] + rand( Normal(0, .1), Nhlf )
        xdat[t,2,1:Nhlf] = q .* xdat[t-1,2,1:Nhlf] .+ (1-q)*mus1_y[t] + rand( Normal(0, .1), Nhlf )
        # cluster 2
        xdat[t,1,(Nhlf+1):N] = q .* xdat[t-1,1,(Nhlf+1):N] .+ (1-q)*mus2_x[t] + rand( Normal(0, .4), N-Nhlf )
        xdat[t,2,(Nhlf+1):N] = q .* xdat[t-1,2,(Nhlf+1):N] .+ (1-q)*mus2_y[t] + rand( Normal(0, .4), N-Nhlf )

    end

    return xdat
end

function genclustline!(xdat::Array{Float64,3}, N::Int64, T::Int64, q::Float64)

    # means for cluster 1
    mus1_x = range(0, stop=2, length=T)
    mus1_y = .5 .* mus1_x

    # means for cluster 2
    mus2_x = range(0, stop=-2, length=T)
    mus2_y = -.5 .* ( mus2_x )

    # generate trajectories
    Nhlf = convert( Int64, N/2 )
    xdat[1,:,:] = reshape(rand( Normal(0, .1), 2*N ), 2, N )
    for t in 2:(T+1)
        # cluster 1
        xdat[t,1,1:Nhlf] = q .* xdat[t-1,1,1:Nhlf] .+ (1-q)*mus1_x[t] + rand( Normal(0, .1), Nhlf )
        xdat[t,2,1:Nhlf] = q .* xdat[t-1,2,1:Nhlf] .+ (1-q)*mus1_y[t] + rand( Normal(0, .1), Nhlf )
        # cluster 2
        xdat[t,1,(Nhlf+1):N] = q .* xdat[t-1,1,(Nhlf+1):N] .+ (1-q)*mus2_x[t] + rand( Normal(0, .4), N-Nhlf )
        xdat[t,2,(Nhlf+1):N] = q .* xdat[t-1,2,(Nhlf+1):N] .+ (1-q)*mus2_y[t] + rand( Normal(0, .4), N-Nhlf )

    end

    return xdat
end

function genvardens!(xdat::Array{Float64,3}, N::Int64, T::Int64, q::Float64)

    # means for each node
    Thlf = convert( Int64, ceil((T+1)/2) )
    theta = range(0, stop=2*pi, length=N+1)[1:N]
    rs = range(.1, stop=4, length=Thlf ) 
    rs = [rs; rs[(end-1):-1:1]][1:T]
    
    # generate trajectories
    xdat[1,:,:] = reshape(rand( Normal(0, .1), 2*N ), 2, N )
    for t in 2:(T+1)
        xdat[t,1,:] = q .* xdat[t-1,1,:] .+ (1-q)*rs[t].*cos.(theta) + rand( Normal(0, .1), N )
        xdat[t,2,:] = q .* xdat[t-1,2,:] .+ (1-q)*rs[t].*sin.(theta) + rand( Normal(0, .1), N )
    end

    return xdat
end

function genclust_sc!(xdat::Array{Float64,3}, N::Int64, T::Int64, q::Float64, tau::Float64, sig::Float64, mu1::Array{Float64,1}, mu2::Array{Float64,1})

    ## this function generates clusters like sewell and chen

    # generate trajectories
    Nhlf = convert( Int64, N/2 ) # gives the 'halfway point'
    xdat[1,:,:] = reshape(rand( Normal(0, tau), 2*N ), 2, N )
    for t in 2:(T+1)
        # cluster 1
        xdat[t,:,1:Nhlf] = (1 - q) .* xdat[t-1,:,1:Nhlf] .+ q.*mu1 + reshape( rand( Normal(0, sig), Nhlf*2 ), 2, Nhlf)
        # cluster 2
        xdat[t,:,(Nhlf+1):N] = (1 - q) .* xdat[t-1,:,(Nhlf+1):N] .+ q.*mu2 + reshape( rand( Normal(0, sig), (N-Nhlf)*2 ), 2, N-Nhlf)

    end

    return xdat
end

function genrw!(xdat::Array{Float64,3}, N::Int64, T::Int64, q::Float64, sig::Float64, phi::Float64)

    ## this function generates clusters like AR rw

    # generate trajectories
    xdat[1,:,:] = reshape(rand( Normal(0, sqrt(sig^2/(1- phi^2))), 2*N ), 2, N )
    for t in 2:(T+1)
        xdat[t,:,:] = phi.*xdat[t-1,:,:] + reshape( rand( Normal(0, sig), N*2 ), 2, N)
    end

    return xdat
end

function gen_alt_coords(N::Int64, T::Int64, case::String, q::Float64, alpha::Float64, mu1::Array{Float64,1}, mu2::Array{Float64,1})
    # 1st round: case can be 'clust_circ', 'clust_line', 'var_dens'
    # 2nd round: case can be 'clust_sc', 'alpha_t'

    ## for clust_sc we take (-1,0), (1,0) and (-.5,0), (.5,0)
    ## this is 'distant' and 'close' clusters, should be able to accomate both but we will see
    
    # set up storage
    xdat = Array{Float64,3}(undef, T + 1, 2, N) 
    ydat = Array{Int64,3}(undef, T, N, N)
    pdat = Array{Float64,3}(undef, T, N, N)

    # generate coordinates
    if (case == "clust_circ")
        genclustcirc!(xdat, N, T+1, q)
    elseif (case == "clust_line")
        genclustline!(xdat, N, T+1, q)
    elseif (case == "var_dens")
        genvardens!(xdat, N, T+1, q)
    elseif (case == "clust_sc")
        sig = .4 
        tau = 1.
        genclust_sc!(xdat, N, T, q, tau, sig, mu1, mu2)
    elseif (case == "alpha_t")
        sig = .4
        phi = .9
        avec = collect(range(2., -2., length=T))
        genrw!(xdat, N, T, q, sig, phi)
    elseif (case == "none") # this is from the model (needed for reference)
        sig = .4
        phi = .9
        genrw!(xdat, N, T, q, sig, phi)
    else
        return("Invalid choice for case")
    end
    
    # get adjacency matrices
    for t in 1:T
        for i in 1:(N-1)
            for j in (i+1):N
                if (case == "alpha_t")
                    eta = avec[t] - sqrt( sum( (xdat[t+1,:,i] - xdat[t+1,:,j]).^2 ) )
                else
                    eta = alpha - sqrt( sum( (xdat[t+1,:,i] - xdat[t+1,:,j]).^2 ) )
                end
                pTmp = 1. / ( 1. + exp( -eta ) )
                pdat[t,i,j] = pTmp
                pdat[t,j,i] = pdat[t,i,j] #symmetric
                ydat[t,i,j] = rand( Bernoulli( pTmp ) )
                ydat[t,j,i] = ydat[t,i,j] #symmetric ties
            end
        end

        for i in 1:N
            ydat[t,i,i] = 0 #don't allow self ties
            pdat[t,i,i] = 0
        end
    end
    
    return xdat, ydat, pdat
end


