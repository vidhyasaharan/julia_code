module GMMbox

using Distances
using Distributions
using Clustering
import Base.rand
import Base.rand!

export gaussian, logprob!, logprob, gmm, rand, rand!

mutable struct gaussian{T<:AbstractFloat}
  ndim::Int64
  mean::AbstractArray{T,2}
  cov::AbstractArray{T,2}
end

gaussian(ndim::Integer) = gaussian(ndim,zeros(ndim,1),eye(ndim))
gaussian(data::Matrix{T} where T<:AbstractFloat)  = gaussian(size(data,1),mean(data,2),cov(data,2))

mutable struct gmm{T<:AbstractFloat}
  nmix::Int64
  wts::AbstractArray{T,2}
  mix::AbstractArray{gaussian{T}}
end

function gmm(nmix::Integer, data::Matrix{T} where T <: AbstractFloat; method = "rand")
  if(method=="rand")
    ndim = size(data,1);
    dcov = cov(data,2);
    dmvn = MvNormal(vec(mean(data,2)),dcov);
    mix = Array{gaussian{eltype(data)}}(nmix);
    for i=1:nmix
      mixmean = convert(Array{eltype(data)},rand(dmvn));
      mix[i] = gaussian(ndim,reshape(mixmean,ndim,1),dcov);
    end
  elseif(method=="kmeans")
    km = kmeans(data,nmix);
    mix = Array{gaussian{eltype(data)}}(nmix);
    dcov = cov(data,2);
    for i=1:nmix
      mix[i] = gaussian(data[:,find(km.assignments.==i)]);
      mix[i].cov = dcov;
    end
  end

  return gmm(nmix,convert(Array{eltype(data)},(1/nmix)*ones(nmix,1)),mix);
end

rand!(g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat = rand!(MvNormal(vec(g.mean),g.cov),data);
rand(g::gaussian{T},nvec::Integer) where T <: AbstractFloat = rand(MvNormal(vec(g.mean),g.cov),nvec);

function rand!(gm::gmm{T},data::Matrix{T}) where T <: AbstractFloat
  ftype = eltype(data);
  nvec = size(data,2);
  pmix = rand(ftype,nvec);
  nvecmix = zeros(Int64,gm.nmix);
  thr = zeros(ftype,1);
  for i=1:gm.nmix
    nvecmix[i] = sum((pmix.>=thr).&(pmix.<thr+gm.wts[i]))
    thr += gm.wts[i];
  end
  if(sum(nvecmix)<nvec)
    nvecmix[end] += nvec-sum(nvecmix);
    println("sum of number of samples per mixture not equal to requested number of samples");
  end
  sindx = 1;
  eindx = 0;
  for i=1:gm.nmix
    eindx += nvecmix[i];
    data[:,sindx:eindx] = rand(gm.mix[i],nvecmix[i]);
    sindx += nvecmix[i];
  end
  return nothing
end

function rand(gm::gmm{T},nvec::Integer) where T <: AbstractFloat
  ftype = eltype(gm.wts);
  pmix = rand(ftype,nvec);
  nvecmix = zeros(Int64,gm.nmix);
  thr = zeros(ftype,1);
  for i=1:gm.nmix
    nvecmix[i] = sum((pmix.>=thr).&(pmix.<thr+gm.wts[i]))
    thr += gm.wts[i];
  end
  if(sum(nvecmix)<nvec)
    nvecmix[end] += nvec-sum(nvecmix);
    println("sum of number of samples per mixture not equal to requested number of samples");
  end
  data = zeros(ftype,gm.mix[1].ndim,sum(nvec));
  sindx = 1;
  eindx = 0;
  for i=1:gm.nmix
    eindx += nvecmix[i];
    data[:,sindx:eindx] = rand(gm.mix[i],nvecmix[i]);
    sindx += nvecmix[i];
  end
  return data
end


logprob!(p::Array{T},g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat = logpdf!(p,MvNormal(vec(g.mean),g.cov),data)
logprob(g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat = logpdf(MvNormal(vec(g.mean),g.cov),data)

function logprob!(p::Array{T},gm::gmm{T},data::Matrix{T}) where T <: AbstractFloat
  mixprob = zeros(eltype(data),size(data,2),gm.nmix);
  for i=1:gm.nmix
    mixprob[:,i] = logprob(gm.mix[i],data);
  end
  maxprob = maximum(mixprob,2);
  broadcast!(-,mixprob,mixprob,maxprob);
  p .= vec(log.((exp.(mixprob))*gm.wts) .+ maxprob);
  return nothing
end

function logprob(gm::gmm{T},data::Matrix{T}) where T <: AbstractFloat
  mixprob = zeros(eltype(data),size(data,2),gm.nmix);
  for i=1:gm.nmix
    mixprob[:,i] = logprob(gm.mix[i],data);
  end
  maxprob = maximum(mixprob,2);
  broadcast!(-,mixprob,mixprob,maxprob);
  return vec(log.((exp.(mixprob))*gm.wts) .+ maxprob);
end



#=
function logprob!(p::Matrix{T},g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat
  logpdf!(reshape(p,length(p)),MvNormal(vec(g.mean),g.cov),data);
  reshape(p,length(p),1);
end
=#

#=
#logpdf and logpdf! using Distributions.jl is faster than this version which uses Distances.jl for Float32 - not using this version for CPU code
function logprob!(p::Matrix{T},g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat
  ndim = size(data,1);
  icov = inv(g.cov);
  lconst = ndim*log(2*pi) + log(det(g.cov));
  pairwise!(p,SqMahalanobis(icov),data,g.mean);
  p .+= lconst;
  p .*= -0.5;
  return nothing
end

function logprob(g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat
  p = zeros(eltype(data),size(data,2),1);
  ndim = size(data,1);
  icov = inv(g.cov);
  lconst = ndim*log(2*pi) + log(det(g.cov));
  pairwise!(p,SqMahalanobis(icov),data,g.mean);
  p .+= lconst;
  p .*= -0.5;
  return p
end
=#


end
