function make(varargin)
%MAKE Build a mex file.
%
%    lda.make('param1', ...)
%
% The function compiles a dependent mex function. The function optionally takes
% additional compiler flags passed to MEX command.
%
% See also mex
  package_dir = fileparts(fileparts(mfilename('fullpath')));
  cmd = sprintf(...
    'mex -largeArrayDims%s -outdir %s -output mex_function_%s',...
    find_source_files(fullfile(package_dir, 'src')),...
    fullfile(package_dir, '+lda', 'private'),...
    sprintf(' %s', varargin{:}));
  disp(cmd);
  eval(cmd);
end

function files = find_source_files(root_dir)
%SOURCE_FILES List of source files in a string.
  files = dir(root_dir);
  srcs = files(cellfun(@(x)~isempty(x), ...
               regexp({files.name},'\S+\.(c)|(cc)|(cpp)|(C)')));
  srcs = cellfun(@(x)fullfile(root_dir, x), {srcs.name},...
                 'UniformOutput', false);
  subdirs = files([files.isdir] & cellfun(@(x)x(1)~='.',{files.name}));
  subdir_srcs = cellfun(@(x)find_source_files(fullfile(root_dir,x)),...
                        {subdirs.name}, 'UniformOutput', false);
  files = [sprintf(' %s', srcs{:}), subdir_srcs{:}];
end
