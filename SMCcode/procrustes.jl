## code to call procrustes from R
## can use this to check errors on Xs

using RCall
R"library('MCMCpack')"

function procrustes( mat::Array{Float64,2}, matStr::Array{Float64,2})
    # mat is current
    # matStr is 'target' matrix (ie. mean, or truth)
    # input as PxN matrices, will need to transform

    mat = mat'
    matStr = matStr'

    @rput mat
    @rput matStr
    R"""
    ans = procrustes( mat, matStr, translation=TRUE, dilation=FALSE)
    ans = ans$X.new
    """

    @rget ans

    return ans'
end
