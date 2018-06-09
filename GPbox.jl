module GPbox

using Gadfly: layer, Geom
using Distributions

export gen_covs, PlotGPsamples1D

se_kernel(p::T,q::T,len::T) where T<:AbstractFloat = exp(-((p-q)/(2*len))^2);

function gen_covs(x::AbstractArray{T},y::AbstractArray{T},kernel_type = "squared error",param...) where T<:AbstractFloat
    if kernel_type == "squared error"
        kernel = se_kernel;
        knl_param = param[1];
        println(kernel_type)
    end

    if(x==y)
        nsam = length(x)
        covs = zeros(nsam,nsam);
        for i=1:nsam
            for j=1:i
                covs[i,j] = kernel(x[i],x[j],knl_param);
                covs[j,i] = covs[i,j];
            end
        end
    else
        nx = length(x);
        ny = length(y);
        covs = zeros(nx,ny);
        for i=1:nx
            for j=1:ny
                covs[i,j] = kernel(x[i],y[j],knl_param);
            end
        end
    end
    return covs
end




####Plotting Functions####

#Draw samples from 1D GP and output as array of Gadfly plotting layers
function PlotGPsamples1D(test_pts,nsam,gp_mn,gp_cov)
    dist = MultivariateNormal(gp_mn,gp_cov)
    f = rand(dist,nsam);
    layers = [layer(
                    x=test_pts,
                    y=f[:,i],
                    Geom.line) for i in 1:nsam];
    return layers
end



end