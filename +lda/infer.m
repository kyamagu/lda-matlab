function [distribution, likelihoods] = infer(corpus, model, varargin)
%INFER Run inference on new samples using a learned LDA model.
%
%    [distribution, likelihoods] = lda.infer(corpus, model, ...)
%    [...] = lda.infer(..., 'param1', value1, ...)
%
% CORPUS is a sparse row vectors of word frequencies. MODEL is a previously
% trained LDA model in a struct given by lda.estimate(). The function takes
% following options.
%
% DISTRIBUTION is row vectors of topic distribution for each document. Each row
% contains weights for each topic. LIKELIHOODS is a vector of document
% likelihoods.
%
% OPTIONS
% -------
%
% var_max_iter [-1]
%
%  Maximum number of iterations in variational inference. Default doesn't
%  limit the iteration by number.
%
% var_converged [1e-6]
%
%  Tolerance value of iterations in the variational inference.
%
% verbose [true]
%
%  Verbosity of message printout.
%
% See also lda.estimate
  options = get_options_(varargin{:});
  [distribution, likelihoods] = mex_function_(mfilename, corpus, model, options);
end
