using Speechbox
using GMMbox
using Distances
using PyPlot
using Distributions

ffile = "/home/vidhya/work/workspace/swb_mfcc/sw_20001_A_1.mfc";
data = read_binary(ffile);

g = gaussian(data);
#p = zeros(size(data,2),1);


mvn = MvNormal(vec(g.mean),g.cov);
@time t1 = logpdf(mvn,data);
@time t1 = logpdf(mvn,data);
@time p = logprob(g,data);
@time p = logprob(g,data);



#=
function fit_gaussian(data::Matrix{AbstractFloat})
  ndim = size(data,1);
  gauss = gaussian(ndim,vec(mean(data,2)),cov(data,2));
  return gauss
end


function dmean(data::Array{Float64,2})
  return mean(data,2);
end

g1 = fit_gaussian(data);
=#
