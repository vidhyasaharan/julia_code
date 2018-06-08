push!(LOAD_PATH,"/home/vidhya/work/julia_code/")

using GPbox
using Distributions
using Gadfly


function plotmvnSamplesTrainPoints(nsam,dist,trainx,trainf)
    f = rand(dist,nsam);
    for i=1:nsam
        display(plot(layer(x=trainx,y=trainf,Geom.point),layer(x=x,y=f[:,i],Geom.line),Coord.cartesian(ymin=-4, ymax=4)))
        sleep(.5);
    end
end




x = linspace(-5,5,50);
len = .2;
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

pred_mvn = MultivariateNormal(pred_mn,pred_cv);

#plotmvnSamplesTrainPoints(10,pred_mvn,trainx,trainf)

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

plot(l1,l2,l3)