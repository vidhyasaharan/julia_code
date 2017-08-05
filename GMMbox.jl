module GMMbox

using Distances
using Distributions

export gaussian, logprob!, logprob

mutable struct gaussian{T<:AbstractFloat}
  ndim::Int64
  mean::Array{T,2}
  cov::Array{T,2}
end

gaussian(ndim::Integer) = gaussian(ndim::Integer,zeros(ndim),eye(ndim))
gaussian(data::Matrix{T} where T<:AbstractFloat)  = gaussian(size(data,1),mean(data,2),cov(data,2))


logprob!(p::Array{T},g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat = logpdf!(p,MvNormal(vec(g.mean),g.cov),data)
logprob(g::gaussian{T},data::Matrix{T}) where T <: AbstractFloat = logpdf(MvNormal(vec(g.mean),g.cov),data)

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
  p = zeros(size(data,2),1);
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
