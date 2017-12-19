using Speechbox
using GMMbox
#using Distances
#using PyPlot
using Distributions
#using Clustering



ndata = convert(Array{Float32},rand(MvNormal(zeros(13),eye(13)),100));
g = gaussian(ndata);
gm = gmm(4,ndata);


data = rand(gm,5000000);


#=
fpath = "/home/vidhya/work/workspace/swb_mfcc";
mfc_flist = filter(x->endswith(x,".mfc"),readdir(fpath));
flist = mfc_flist;
for i=1:length(mfc_flist)
  flist[i] = joinpath(fpath,mfc_flist[i]);
end
=#

#=
ofile = "/home/vidhya/work/workspace/swb.mfc";
#combine_binary(ofile,flist);
data = read_binary(ofile);
=#


data = convert(Matrix{Float32},data);
g = gaussian(data);
gm = gmm(4,data);


p = zeros(eltype(data),size(data,2));

#=
@time p = logprob(gm,data);
@time p = logprob(gm,data);
@time logprob!(p,gm,data);
=#

@time @sync for i=1:10
  logprob!(p,gm,data);
end
