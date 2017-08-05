using Speechbox
using GMMbox
#using Distances
#using PyPlot
using Distributions

fpath = "/home/vidhya/work/workspace/swb_mfcc";
mfc_flist = filter(x->endswith(x,".mfc"),readdir(fpath));
flist = mfc_flist;
for i=1:length(mfc_flist)
  flist[i] = joinpath(fpath,mfc_flist[i]);
end

ofile = "/home/vidhya/work/workspace/swb.mfc";

#combine_binary(ofile,flist);

data = read_binary(ofile);
d1 = read_binary(flist[end]);


data = convert(Matrix{Float32},data);

g = gaussian(data);
t1 = zeros(typeof(data[1]),size(data,2));
p = zeros(typeof(data[1]),size(data,2));

mvn = MvNormal(vec(g.mean),g.cov);
@time t1 = logpdf(mvn,data);
@time t1 = logpdf(mvn,data);
@time p = logprob(g,data);
@time p = logprob(g,data);
