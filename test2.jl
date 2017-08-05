addprocs(7);

@everywhere import WAV
@everywhere import Speechbox


@everywhere using WAV
@everywhere using Speechbox



fpath = "/home/vidhya/work/data/SWB2_part/seg_wav_files";
wav_flist = filter(x->endswith(x,".wav"),readdir(fpath));


@everywhere function ex_mfcc(fname)
  ext = ".mfc"
  fpath = "/home/vidhya/work/data/SWB2_part/seg_wav_files";
  opath = "/home/vidhya/work/workspace/swb_mfcc";
  bname,~ = splitext(fname);
  ifile = joinpath(fpath,fname);
  ofile = joinpath(opath,bname*ext);
  #println(ifile);

  x,fs = wavread(ifile);
  sx = normalise_energy(x,fs);
  mf = melfcc(sx,fs);
  write_binary(ofile,mf);
end


println(nprocs())

function run_ex_mfcc(wav_flist)
  @sync @parallel for i=1:10000
    ex_mfcc(wav_flist[i]);
  end
  return nothing
end


#@time pmap(ex_mfcc,wav_flist[1:1000])
@time run_ex_mfcc(wav_flist);

rmprocs(workers());
println(nprocs())
