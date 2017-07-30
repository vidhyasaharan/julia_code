FFTW.set_num_threads(1)

function test_fft()
  n = 1:2048;
  fs = 2000;
  f = 500;
  s = sin.(2*pi*f/fs*n);
  p = plan_rfft(s);
  for i=1:1000
    ffs = p*s;
#    ffs = rfft(s);
    ms = abs.(ffs);
  end
  return(ms)
end

#=
@time test_fft()
@time test_fft()
@time ms1 = test_fft()
=#


#=
function specgram(x,fs,win_dur=0.02,win_overlap=0.01)
  wlen = win_dur*fs

=#

a = 1:1:10000;
function splt(s)
  t = zeros(5,1);
  x = arraysplit(s,3,1);
  for i=1:length(x)
    t[1:1:3] = x[i];
  end
  return t
end

function splt2(s)
  x = arraysplit(s,3,1);
  for i=1:length(x)
    t = [x[i];zeros(2,1)];
  end
  return t
end

x = arraysplit(a,3,1);
for i=1:length(x)
  t = [x[i];zeros(2,1)];
end



#=
@time y = splt(a)
@time y = splt(a)
@time y = splt(a)

@time z = splt2(a)
@time z = splt2(a)
@time z = splt2(a)
=#

a = 1.0f0

function prs(x::Float64)
  y = 2.5*x;
  return y
end
