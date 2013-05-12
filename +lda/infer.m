function [likelihoods, distribution] = infer(corpus, model, varargin)
%INFER Run inference on new samples using a learned LDA model.
%
%    [likelihoods, distribution] = lda.infer(corpus, model, ...)
%    [...] = lda.infer(..., 'param1', value1, ...)
%
% CORPUS is a sparse row vectors of word frequencies. MODEL is a previously
% trained LDA model in a struct given by lda.estimate(). The function takes
% following options.
%
% # OPTIONS
%
% _var_max_iter_ [-1]
%
% Maximum number of iterations in variational inference. Default doesn't limit
% the iteration by number.
%
% _var_converged_ [1e-6]
%
% Tolerance value of iterations in the variational inference.
%
% _random_seed_ [4357]
%
% Seed number for random number generator.
%
% _verbose_ [true]
%
% Verbosity of message printout.
%
% See also lda.estimate
  options = get_options_(varargin{:});
  [likelihoods, distribution] = mex_function_(mfilename, corpus, model, options);
end
