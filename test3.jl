using Distributions
using Gadfly

ndim = 10;
mn = zeros(ndim);
cv = zeros(ndim,ndim)

sq_exp_kernel(x::Number,y::Number,len::Number) = exp(-(x-y)^2/len)

len = 3;
for i=1:ndim
    for j=1:i
        cv[i,j] = sq_exp_kernel(i,j,len);
        cv[j,i] = cv[i,j];
    end
end

mvn = MultivariateNormal(mn,cv);

nsam = 20;
x = rand(mvn,nsam);
for i=1:nsam
    display(plot(x=1:ndim,y=x[:,i],Geom.line,Coord.cartesian(ymin=-4, ymax=4)))
    sleep(.5)
end

x_fixed = 