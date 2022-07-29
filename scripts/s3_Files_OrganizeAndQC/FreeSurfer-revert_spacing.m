% resample Freesurfer output back from 256x256x256 1mm iso to original matrix size
% for SLEEEP study

maindir='/data/projects/depredict/sleeep'
vmaindir=fullfile(maindir,'vols_T1')

segs={'segs_fastsurfer','segs_freesurfer'};
files=cellstr(spm_select('fplistrec',vmaindir,'^2.*\.nii$'))

%%
for ii=1:length(files)
  ii
  file=files{ii};
  tfile=recurse_modify_field(file,'vols_T1','vols_MEMPRAGE'); % target file
  s1=recurse_modify_field(file,'vols_T1',segs{1});
  s2=recurse_modify_field(file,'vols_T1',segs{2});
  if ~exist(tfile,'file')
    tfile
    %error('hmm')
  end

  if ~exist(s1,'file')
    s1
    %error('hmm')
  end
  if ~exist(s2,'file')
    s2
    %error('hmm')
  end
  flags.which=1;
  flags.mean=0;
  flags.interp=0;
  rs2=recurse_modify_field(s2,'/2018','/r2018');
  if 1%~exist(rs2)
    try
      rs2
      spm_reslice({tfile,file,s1,s2},flags)
    end
  end
end
