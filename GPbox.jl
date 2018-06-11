module GPbox

const Float = Float64

using Gadfly: layer, Geom
using Distributions
using Distances

export PlotGPsamples1D, estGP, GPkernel, genCovs

####Kernel functions####
struct GPkernel
    kernel_type::String
    param::Dict
end


####Covariance Matrix Generation#### 
function genCovs(x::Matrix{T},y::Matrix{T},gpk::GPkernel) where T<:Float
    if(gpk.kernel_type=="squared exp")
        nx = size(x,2);
        ny = size(y,2);
        lensq::T = (get(gpk.param,"len",1))^2;
        len_fact::T = 1/(2*lensq);
        covs = pairwise(SqEuclidean(),x,y);
        covs = exp.((-covs).*len_fact);
    end
    return covs
end



####Estimating posterior GP (mean and covarainces) given training point, test points and GP kernel####
function estGP(test_pts::AbstractArray{T},train_pts::AbstractArray{T},train_vals::AbstractArray{T},gpk::GPkernel) where T<:Float
    cv = genCovs(test_pts,test_pts,gpk);
    num_testpts = size(test_pts,2);
    num_fdim = size(train_vals,1);
    mn = zeros(num_fdim,num_testpts);
    traincv = genCovs(train_pts,train_pts,gpk);
    crosscv = genCovs(test_pts,train_pts,gpk);
    itraincv = inv(traincv);
    gp_mn = crosscv*itraincv*train_vals';
    temp = crosscv*itraincv*crosscv';
    gp_cv = cv - (temp + temp')/2;  #hack for positive definite CV
    return gp_mn, gp_cv
end



####Plotting Functions####

#Draw samples from 1D GP and output as array of Gadfly plotting layers
function PlotGPsamples1D(test_pts,nsam,gp_mn,gp_cov)
    dist = MultivariateNormal(gp_mn[:],gp_cov)
    f = rand(dist,nsam);
    layers = [layer(
                    x=test_pts,
                    y=f[:,i],
                    Geom.line) for i in 1:nsam];
    return layers
end



end