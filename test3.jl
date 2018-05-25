using Distributions
using Gadfly


se_kernel(p::Number,q::Number,len::Number) = exp(-((p-q)/(2*len))^2);

function gen_se_cov(x,len::Number)
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

x = linspace(0,1,15);
len = .2;
cv = gen_se_cov(x,len);
mn = zeros(length(x));

mvn = MultivariateNormal(mn,cv);

nsam = 20;
f = rand(mvn,nsam);
for i=1:nsam
    display(plot(x=x,y=f[:,i],Geom.line,Coord.cartesian(ymin=-4, ymax=4)))
    sleep(.5)
end
