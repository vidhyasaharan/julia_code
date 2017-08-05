module Speechbox

using WAV
using DSP

export specgram, enframe, mel2frq, frq2mel, melbankm, melfcc, vad, normalise_energy, write_binary, read_binary, combine_binary

function combine_binary(ofile,flist)
  if(isempty(flist))
    return nothing
  end
  data = read_binary(flist[1]);
  ndim,nvec = size(data);
  f = open(ofile,"w");
  write(f,ndim);
  write(f,nvec);
  write(f,data[:]);
  if(length(flist)>1)
    for i=2:length(flist)
      data = read_binary(flist[i]);
      nvec += size(data,2);
      write(f,data[:]);
    end
    seekstart(f);
    write(f,ndim);
    write(f,nvec);
    close(f);
  end
  return nothing
end

function read_binary(ofile)
  f = open(ofile,"r");
  ndim = read(f,Int64);
  nvec = read(f,Int64);
  feature_matrix = read(f,Float64,ndim,nvec);
  close(f)
  return feature_matrix;
end

function write_binary(ofile,feature_matrix)
  ndim,nvec = size(feature_matrix);
  f = open(ofile,"w");
  write(f,ndim);
  write(f,nvec);
  write(f,feature_matrix[:]);
  close(f);
  return nothing
end

function normalise_energy(x,fs)
  vindx = vad(x,fs);
  frames = enframe(x,fs);
  nframes = length(frames);
  en = zeros(nframes);
  for i=1:nframes
    en[i] = rms(frames[i]);
  end
  mn_en = mean(en[find(vindx)]);
  sc_factor = 0.1/mn_en;
  sx = sc_factor*x;
  return sx;
end

function vad(x,fs,win_dur=0.02,win_overlap=0.01,energy_thr=0.05)
  frames = enframe(x,fs,win_dur,win_overlap);
  nframes = length(frames);
  energy = zeros(nframes);
  for i=1:nframes
    energy[i] = rms(frames[i]);
  end
  max_energy = maximum(energy);
  abs_thr = max_energy*energy_thr;
  v_indx = zeros(Int,nframes);
  v_indx[find(energy.>abs_thr)]=1;
  return v_indx
end

function melfcc(x,fs,ncoef=13,nfilt=17,win_dur=0.02,win_overlap=0.01)
  frames = enframe(x,fs,win_dur,win_overlap);
  nframes = length(frames);
  flen = length(frames[1]);
  nfft = nextfastfft(flen);
  win = hamming(flen);

  buf = zeros(nfft);
  fbuf = zeros(nfilt);
  rfp = plan_rfft(buf);
  dcp = plan_dct(fbuf);

  nrfft = length(rfp*buf);
  fbank = melbankm(fs,nrfft,nfilt); #generate mel filterbank filters
  fbank = fbank.^2; #squaring triangular mel-filters to multiple with power spectrum

  mfcc = zeros(ncoef,nframes);

  for i=1:nframes
    buf[1:1:flen] = win.*frames[i];
    #fbuf = dcp*(fbank*(abs.(rfp*buf)));
    fbuf = fbank*(abs2.(rfp*buf) + eps()); #multiplying with power spectrum (square of mag spectrum) and accumulating
    fbuf = dcp*(log.(fbuf));
    mfcc[:,i] = fbuf[1:ncoef];
  end
  return mfcc
end


function melbankm(fs,npts,nfilt=17)
  mspace = linspace(0,frq2mel(fs/2),nfilt); #equally spaced points in mel scale
  fspace = mel2frq.(mspace); #equal mel spaced points mapped back to Hz
  cindx = Int.(round.(fspace*(npts-1)/(fs/2))+1); #Filter centre indices (first at 0, final at Fs/2)
  fbank = zeros(nfilt,npts);
  #define triangular filters for 2 to nfilt-1
  for i=2:nfilt-1
    fbank[i,cindx[i-1]:cindx[i]] = linspace(0,1,cindx[i]-cindx[i-1]+1);
    fbank[i,cindx[i]:cindx[i+1]] = linspace(1,0,cindx[i+1]-cindx[i]+1);
  end
  #one sided triangular filters for first and last filters
  fbank[1,cindx[1]:cindx[2]] = linspace(1,0,cindx[2]-cindx[1]+1);
  fbank[nfilt,cindx[nfilt-1]:cindx[nfilt]] = linspace(0,1,cindx[nfilt]-cindx[nfilt-1]+1);
  return fbank
end

function specgram(x,fs,win_dur=0.02,win_overlap=0.01,wtype="hamm")
  frames = enframe(x,fs,win_dur,win_overlap);
  nframes = length(frames);
  flen = length(frames[1]);
  nfft = nextfastfft(flen);
  if(wtype=="rect")
    win = ones(flen);
  else
    win = hamming(flen);
  end
  buf = zeros(nfft);
  rfp = plan_rfft(buf);
  nrfft = length(rfp*buf);
  mspec = zeros(nrfft,nframes);
  for i=1:nframes
    buf[1:1:flen] = win.*frames[i];
    mspec[:,i] = abs.(rfp*buf);
  end
  return mspec + eps();
end

function enframe(x,fs,win_dur=0.02,win_overlap=0.01)
  wlen = Int(round(win_dur*fs));
  wolap = Int(round(win_overlap*fs));
  x = x[:];
  frames = arraysplit(x,wlen,wolap);
  return frames
end

frq2mel(frq) = log(1+frq/700)*1127.01048;

mel2frq(mel) = 700*(exp(mel/1127.01048)-1);

end
