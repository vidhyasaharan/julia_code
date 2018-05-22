using Distributions
using Gadfly

x = rand(MultivariateNormal([0.0,0.0],[1.0 0.5; 0.5 1.0]),10000);

plot(x=x[1,:],y=x[2,:],Geom.hexbin(xbincount=300,ybincount=300))