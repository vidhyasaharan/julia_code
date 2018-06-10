push!(LOAD_PATH,"C:\\julia_code")

const Float = Float64;

using Distances
using GPbox




@inline sqvd(x::Vector{T},y::Vector{T}) where T<:Float = @views evaluate(SqEuclidean(),x,y);

# @inline function sqvd(x::Vector{T},y::Vector{T}) where T<:Float
#     temp = x-y;
#     return temp'*temp;
# end

function gen_cov(x::Matrix{T},y::Matrix{T},gpk::GPkernel) where T<:Float
    if(gpk.kernel_type=="squared error")
        nx = size(x,2);
        ny = size(y,2);
        lensq::T = (get(gpk.param,"len",1))^2;
        covs = Array{T}(nx,ny);
        @inbounds for i=1:nx
            @inbounds for j=1:ny
                covs[i,j] = sqvd(x[:,i],y[:,j]);
            end
        end
    end
    return covs
end


function sdist(x::Matrix{T},y::Matrix{T}) where T<:Float
    nx = size(x,2);
    ny = size(y,2);
    sd = Array{T}(nx,ny);
    @inbounds for i=1:nx
        @inbounds for j=1:ny
            sd[i,j] = sqvd(x[:,i],y[:,j]);
        end
    end
    return sd
end

# sqd(x::T,y::T) where T<:Float = (x-y)^2;

# function vdist(x::Vector{T},y::Vector{T}) where T<:Float
#     nx = length(x);
#     ny = length(y);
#     vd = zeros(nx,ny);
#     for i=1:nx
#         for j=1:ny
#             vd[i,j] = sqd(x[i],y[j]);
#         end
#     end
#     return vd
# end

temp = linspace(-5,5,5000);
# x = reshape(x,1,length(x));
x = Array(Float,1,length(temp));
y = Array(Float,length(temp));
for i in eachindex(x)
    x[i] = temp[i];
    y[i] = temp[i];
end
len = 0.3;
gpk = GPkernel("squared error",Dict("len"=>len));

@time tt1 = gen_cov(x,x,gpk);
@time tt2 = genCovs(y,y,gpk);
@time tt3 = sdist(x,x);
@time tt4 = vdist(y,y);