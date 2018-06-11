push!(LOAD_PATH,"C:\\julia_code")


using GPbox
using Distributions
using Gadfly

temp = linspace(-5,5,50);
x = Array{Float64}(1,length(temp));
for i in eachindex(temp)
    x[1,i] = temp[i];
end
len = .3;
trainx = [-4, -1.6, -.8, 2.5, 3.9]
trainf = [-1, -.1, .4, .2, -1.2]

trainx = reshape(trainx,1,length(trainx));
trainf = reshape(trainf,1,length(trainf));

gpk = GPkernel("squared exp",Dict("len"=>len))
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