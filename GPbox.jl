module GPbox

using Gadfly: layer, Geom
using Distributions

export PlotGPsamples1D, estGP, GPkernel, genCovs

####Kernel functions####
struct GPkernel
    kernel_type::String
    param::Dict
end

#Squared error kernel for scalars

se_kernel(p::T,q::T,len::T) where T<:AbstractFloat = exp((-((p-q)/(len))^2)/2);

function genCovs(x::AbstractArray{T},y::AbstractArray{T},gpk::GPkernel) where T<:AbstractFloat
    if gpk.kernel_type == "squared error"
        kernel::Function = se_kernel;
        knl_param::T = gpk.param["len"];
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




####Estimating posterior GP (mean and covarainces) given training point, test points and GP kernel####
function estGP(test_pts::AbstractArray{T},train_pts::AbstractArray{T},train_vals::AbstractArray{T},gpk::GPkernel) where T<:AbstractFloat
    cv = genCovs(test_pts,test_pts,gpk);
    mn = zeros(length(test_pts));
    traincv = genCovs(train_pts,train_pts,gpk);
    crosscv = genCovs(test_pts,train_pts,gpk);
    itraincv = inv(traincv);
    gp_mn = crosscv*itraincv*train_vals;
    temp = crosscv*itraincv*crosscv';
    gp_cv = cv - (temp + temp')/2;  #hack for positive definite CV
    return gp_mn, gp_cv
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