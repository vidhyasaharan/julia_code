using Speechbox
using GMMbox
#using Distances
#using PyPlot
using Distributions
#using Clustering


ndata = convert(Array{Float32},rand(MvNormal(zeros(13),eye(13)),100));

gm = gmm(4,ndata);



data = rand(gm,500);



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
gm = gmm(4,data);


t1 = zeros(typeof(data[1]),size(data,2));
p = zeros(typeof(data[1]),size(data,2),1);

function logprob!(p::Array{T},gm::gmm{T},data::Matrix{T}) where T <: AbstractFloat
  mixprob = zeros(eltype(data),size(data,2),gm.nmix);
  for i=1:gm.nmix
    @time mixprob[:,i] = logprob(gm.mix[i],data);
    @time logprob!(t1,gm.mix[i],data);
  end
end


@time logprob!(t1,g,data);
@time logprob!(t1,g,data);
