%%
wd = '/home/mgpoirot/lood_storage/divi/Projects/depredict/repositories/SLEEEP/data/dti_working_dir/';
files = dir(fullfile(wd, '*', '*_t1.nii'));

for ii=1:length(files)
   disp(strcat(num2str(ii), '\', num2str(length(files))))
   fname = files(ii).name(1:15);
   [~, sub, ~] = fileparts(files(ii).folder);


   t1=fullfile(wd, sub, strcat(fname, '_t1.nii'));
   di=fullfile(wd, sub, strcat(fname, '_di.nii'));
   d2=fullfile(wd, sub, strcat('r', fname, '_di.nii'));

   if isfile(d2)
       continue
   else
      coregister_to_epi(di, t1)
   end
end


%%
function coregister_to_epi(sourcefile,varargin)
% function coregister_to_epi(sourcefile,[targetfile],[other],[estwrite],[initGravity])
%
% sourcefile: analyze-file with full path
%             e.g. 'D:/dti/b0.img'
% Using spm, coregister an .img file to the spm EPI-template
% (which does not contain the skull).
% Preferably, you'd like to match the DTI B0-image.
% nrrd2analyze.m might be useful.
% [estwrite] is default 0

%[path,name,ext]=fileparts(which('spm'));
path=spm('dir');
if isempty(which('spm_run_coreg_estimate'))
  addpath(genpath(fileparts(path)))
end

spmdir=path;
ref = fullfile(spmdir,'templates','EPI.nii');
isSPM12=0;
if ~isempty(strfind(path,'spm12'))
  isSPM12=1;
  ref = fullfile(spmdir,'canonical','avg152T1.nii');
end
if nargin>=2 && ~isempty(varargin{1})
  ref = varargin{1}; 
  if ~strcmp(ref(end-3:end),'.img') && ~strcmp(ref(end-3:end),'.nii')
    error('File extension must be .img or .nii')
  end
end
% datasets to co-align
other = '';
if nargin>=3
  other = varargin{2};
end
estwrite = 0;
if nargin>=4 && ~isempty(varargin{3}) && varargin{3}
  estwrite= 1;
end
initGravity=1;
if nargin>=5 
  initGravity = varargin{4};
end

if ~exist(ref)
    error(['Template does not exist: ' ref])
end
if ~exist(sourcefile)
    error(['Given sourcefile does not exist: ' sourcefile])
end

estimate.ref{1} = [ref ',1'];
estimate.source{1} = [sourcefile ',1'];
estimate.other = {other}; % other scans to apply transform to

estimate.eoptions.cost_fun = 'nmi';
estimate.eoptions.sep = [4 2];
estimate.eoptions.tol = [.02 .02 .02 .001*ones(1,9)];
estimate.eoptions.fwhm = [7 7];

estimate.roptions.interp = 2;
estimate.roptions.wrap = [0 0 0];
estimate.roptions.mask= 0;
estimate.roptions.prefix = 'r';

%% initialize by center of gravity (requires DIPimage)

if initGravity
  try
    dip_image; % invoke a DIPimage function
    disp('initialising by center of gravity.')
    s=nifti(sourcefile);
    r=nifti(ref);
    svol=s.dat(:,:,:); rvol=r.dat(:,:,:);
    svol(isnan(svol))=0; rvol(isnan(rvol))=0;
    sg=measure(svol>=0,svol,'gravity'); % DIPimage function
    rg=measure(rvol>=0,rvol,'gravity');
    sg=sg.Gravity([2 1 3]) + 1; rg=rg.Gravity([2 1 3]) + 1;
    if size(sg,1)==1, sg=sg'; end
    if size(rg,1)==1, rg=rg'; end
    wsg = s.mat*[sg;1];
    wrg = r.mat*[rg;1];
    s.mat(:,4) = s.mat(:,4) + wrg-wsg;
    %s.mat0 = s.mat;
    create(s)
    if ~isempty(other)
      if isstr(other)
        o=nifti(other);
        o.mat(:,4) = o.mat(:,4) + wrg-wsg;
        %o.mat = s.mat; o.mat0=s.mat0;
        create(o)
      elseif iscell(other)
        for iC=1:length(other)
          o=nifti(other{iC});
          o.mat(:,4) = o.mat(:,4) + wrg-wsg;
          %o.mat = s.mat; o.mat0=s.mat0;
          create(o)
        end
      end
    end
  catch
    warning('DIPimage not installed, skipping init to center of gravity.')
  end
end
%% generate job structure
%jobs{1}.spatial{1}.coreg{1}.estimate = estimate;

%% and run
%spm_jobman('run',jobs)

%% matlab r2008 spm5/8 mcc compilation compatible
if estwrite
  spm_run_coreg_estwrite(estimate);
else
  if isSPM12
    spm_run_coreg(estimate);
  else
    spm_run_coreg_estimate(estimate);
  end
end
%spm_run_normalise_estimate(est);
end