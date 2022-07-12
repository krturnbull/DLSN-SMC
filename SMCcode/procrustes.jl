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

# function procrustxHst!( strg::storage, refcoord::Array{Float64,3} )
#     # function takes in storage object and transforms all x coordinates onto reference mat
#     # this code is slow and dumb!
#     # refcord is dim T x P x N

#     print("\n")
#     p = Progress(prms.T, 1, "Procrustes...!!", 25)
    
#     for t in 1:T
        
#         next!(p)
        
#         for m in 1:M
#             ans = procrustes( strg.xHst[t,:,:,m], refcoord[t,:,:] )
#             strg.xHst[t,:,:,m] = ans
#         end
        
#     end

#     print("\n")
    
#     strg
# end
