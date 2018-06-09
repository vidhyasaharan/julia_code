push!(LOAD_PATH,"C:\\julia_code")

using GPbox
using Distributions
using Gadfly



function PlotGPsamples1D(nsam,gp_mn,gp_cov)
    dist = MultivariateNormal(gp_mn,gp_cov)
    f = rand(dist,nsam);
    layers = [layer(
                    x=x,
                    y=f[:,i],
                    Geom.line) for i in 1:nsam];
    return layers
end



x = linspace(-5,5,50);
len = .3;
cv = gen_covs(x,x,"squared error",len);
mn = zeros(length(x));

mvn = MultivariateNormal(mn,cv);

trainx = [-4, -1.6, -.8, 2.5, 3.9]
trainf = [-1, -.1, .4, .2, -1.2]

traincv = gen_covs(trainx,trainx,"squared error",len);
crosscv = gen_covs(x,trainx,"squared error",len);

pred_mn = crosscv*inv(traincv)*trainf;
temp = crosscv*inv(traincv)*crosscv';
pred_cv = cv - (temp + temp')/2;





pred_std = sqrt.(diag(pred_cv));


l1 = layer(x=x,
            y=pred_mn,
            Geom.line,
            style(default_color=colorant"red"));

l2 = layer(
            x=trainx,
            y=trainf,
            Geom.point,
            #color="green";)
            style(default_color=colorant"green"));

l3 = layer(x=x,
            ymin=pred_mn-pred_std,
            ymax=pred_mn+pred_std,
            Geom.ribbon);


lyrs = PlotGPsamples1D(5,pred_mn,pred_cv);
plot(l1,l2,l3,lyrs...)