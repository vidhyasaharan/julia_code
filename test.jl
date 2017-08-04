mutable struct gaussian{T<:AbstractFloat}
  ndim::Int64
  mean::Array{T,1}
  cov::Array{T,2}
end

gaussian(ndim::Integer) = gaussian(ndim::Integer,zeros(ndim),eye(ndim))


using Speechbox
