push!(LOAD_PATH,"C:\\julia_code")


using GPbox
using Distributions
using Gadfly



x = linspace(-5,5,50);
len = .3;
trainx = [-4, -1.6, -.8, 2.5, 3.9]
trainf = [-1, -.1, .4, .2, -1.2]

gpk = GPkernel("squared error",Dict("len"=>len))

# @time t1 = gen_covs(x,x,"squared error",len);
# @time t1 = gen_covs(x,x,"squared error",len);
# @time t2 = genCovs(x,x,gpk);
# @time t2 = genCovs(x,x,gpk);


pred_mn, pred_cv = estGP(x,trainx,trainf,gpk);

pred_std = sqrt.(diag(pred_cv));


l1 = layer(x=x,
            y=pred_mn,
            Geom.line,
            Theme(default_color=colorant"violet"));

l2 = layer(
            x=trainx,
            y=trainf,
            Geom.point,
            style(default_color=colorant"green"));

l3 = layer(x=x,
            ymin=pred_mn-pred_std,
            ymax=pred_mn+pred_std,
            Geom.ribbon);


lyrs = PlotGPsamples1D(x,5,pred_mn,pred_cv);
plot(l1,l2,l3,lyrs...)