push!(LOAD_PATH,"C:\\julia_code")

const Float = Float64;

using Distances
using GPbox


function sqvd2(ndim::Int64,x::Vector{T},y::Vector{T},len_fact::T) where T<:Float64
    # len_fact = 1/(2*len^2);
    s::T = 0.0;
    for i=1:ndim
        s+=(x[i] - y[i])^2;
    end
    # return s
    return exp(-s*len_fact)
end

function gen_cov(x::Matrix{T},y::Matrix{T},gpk::GPkernel) where T<:Float
    if(gpk.kernel_type=="squared exp")
        nx = size(x,2);
        ny = size(y,2);
        ndim::Int64 = size(x,1);
        lensq::T = (get(gpk.param,"len",1))^2;
        covs = Array{T}(nx,ny);
        for i=1:nx
            for j=1:ny
                # @inbounds covs[i,j] = sqvd(x[:,i],y[:,j]);
                @inbounds covs[i,j] = sqvd2(ndim,x[:,i],y[:,j],1/(2*lensq));
            end
        end
    end
    return covs
end


len = 0.3;
gpk = GPkernel("squared exp",Dict("len"=>len));
r1 = rand(10,5000);
r2 = rand(10,5000);


@time tt1 = gen_cov(r1,r2,gpk);
@time tt2 = genCovs(r1,r2,gpk);
