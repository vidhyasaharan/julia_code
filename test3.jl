using Distributions
using Gadfly


# struct se_knl
#     len::AbstractFloat
# end

# function (knl::se_knl)(p::T,q::T) where T<:AbstractFloat
#     return exp(-((p-q)/(2*knl.len))^2)
# end

# function gen_cov(x,kernel::se_knl)
#     nsam = length(x);
#     cv = zeros(nsam,nsam);
#     for i=1:nsam
#         for j=1:i
#             cv[i,j] = kernel(x[i],x[j]);
#             cv[j,i] = cv[i,j];
#         end
#     end
#     return cv
# end


struct kernel_param
    se_len::AbstractFloat
end

function kernel_param(;se_len=1)
    kernel_param(se_len)
end


se_krnl(p::T,q::T,param::kernel_param) where T<:AbstractFloat = exp(-((p-q)/(2*param.se_len))^2);

function gen_cv(x,kernel::Function,param::kernel_param)
    nsam = length(x);
    cv = zeros(nsam,nsam);
    for i=1:nsam
        for j=1:i
            cv[i,j] = kernel(x[i],x[j],param);
            cv[j,i] = cv[i,j];
        end
    end
    return cv
end



se_kernel(p::T,q::T,len::T) where T<:AbstractFloat = exp(-((p-q)/(2*len))^2);

function gen_se_cov(x::AbstractArray{T},len::T) where T<:AbstractFloat
    nsam = length(x)
    cv = zeros(nsam,nsam);
    for i=1:nsam
        for j=1:i
            cv[i,j] = se_kernel(x[i],x[j],len);
            cv[j,i] = cv[i,j];
        end
    end
    return cv
end


function gen_covs(x::AbstractArray{T},y::AbstractArray{T},kernel_type = "squared error",param...) where T<:AbstractFloat
    if kernel_type == "squared error"
        kernel = se_kernel;
        knl_param = param[1];
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



function gen_se_xcov(x,y,len::Number)
    nx = length(x);
    ny = length(y);
    xcv = zeros(nx,ny);
    for i=1:nx
        for j=1:ny
            xcv[i,j] = se_kernel(x[i],y[j],len);
        end
    end
    return xcv
end

function plotmvnSamples(nsam,dist)
    f = rand(dist,nsam);
    for i=1:nsam
        display(plot(x=x,y=f[:,i],Geom.line,Coord.cartesian(ymin=-4, ymax=4)))
        sleep(.5);
    end
end


function plotmvnSamplesTrainPoints(nsam,dist,trainx,trainf)
    f = rand(dist,nsam);
    for i=1:nsam
        display(plot(layer(x=trainx,y=trainf,Geom.point),layer(x=x,y=f[:,i],Geom.line),Coord.cartesian(ymin=-4, ymax=4)))
        sleep(.5);
    end
end


x = linspace(-5,5,5000);
len = .3;
cv = gen_se_cov(x,len);
mn = zeros(length(x));

mvn = MultivariateNormal(mn,cv);
plotmvnSamples(20,mvn)


trainx = [-4, -1.6, -.8, 2.5, 3.9]
trainf = [-1, -.1, .4, .2, -1.2]

traincv = gen_se_cov(trainx,len);
crosscv = gen_se_xcov(x,trainx,len);

pred_mn = crosscv*inv(traincv)*trainf;
temp = crosscv*inv(traincv)*crosscv';
pred_cv = cv - (temp + temp')/2;

pred_mvn = MultivariateNormal(pred_mn,pred_cv);

#plotmvnSamplesTrainPoints(10,pred_mvn,trainx,trainf)

pred_std = sqrt.(diag(pred_cv));


l1 = layer(x=x,
            y=pred_mn,
            Geom.line,
            style(default_color=color("red")));

l2 = layer(
            x=trainx,
            y=trainf,
            Geom.point,
            style(default_color=color("green")));

l3 = layer(x=x,
            ymin=pred_mn-pred_std,
            ymax=pred_mn+pred_std,
            Geom.ribbon);
plot(l1,l2,l3)