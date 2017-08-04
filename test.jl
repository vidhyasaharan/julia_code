using Speechbox
using GMMbox



ffile = "/home/vidhya/work/workspace/swb_mfcc/sw_20001_A_1.mfc";
data = read_binary(ffile);

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
