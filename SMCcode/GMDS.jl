# call GMDS estimate from R

using RCall

function GMDS(Y::Array{Int64,2})

    @rput Y
    R"""
    library(igraph)
    g = graph_from_adjacency_matrix(Y, mode="undirected")
    ds = distances(g)
    xest = cmdscale(ds, k=2)
    """
    @rget xest

    return xest
end
