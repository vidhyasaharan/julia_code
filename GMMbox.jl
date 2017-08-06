module GMMbox

using Distances
using Distributions
using Clustering

export gaussian, logprob!, logprob, gmm

mutable struct gaussian{T<:AbstractFloat}
  ndim::Int64
  mean::Array{T,2}
  cov::Array{T,2}
end

gaussian(ndim::Integer) = gaussian(ndim,zeros(ndim,1),eye(ndim))
gaussian(data::Matrix{T} where T<:AbstractFloat)  = gaussian(size(data,1),mean(data,2),cov(data,2))

mutable struct gmm{T<:AbstractFloat}
  nmix::Int64
  wts::Array{T,2}
  mix::Array{gaussian{T}}
end

function gmm(nmix::Integer, data::Matrix{T} where T <: AbstractFloat; method = "rand")
  if(method=="rand")
    println("rnd")
    ndim = size(data,1);
    dcov = cov(data,2);
    dmvn = MvNormal(vec(mean(data,2)),dcov);
    mix = Array{gaussian{eltype(data)}}(nmix);
    for i=1:nmix
      mixmean = convert(Array{eltype(data)},rand(dmvn));
      mix[i] = gaussian(ndim,reshape(mixmean,ndim,1),dcov);
    end
  elseif(method=="kmeans")
    println("km")
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



logprob!(p::Array{T},g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat = logpdf!(p,MvNormal(vec(g.mean),g.cov),data)
logprob(g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat = logpdf(MvNormal(vec(g.mean),g.cov),data)

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
