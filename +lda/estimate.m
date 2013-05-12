function [model, assignment] = estimate(corpus, model, varargin)
%ESTIMATE Run the EM algorithm to estimate a topic model and assignment.
%
%    [model, assignment] = lda.estimate(corpus, model, 'param1', value1, ...)
%
% CORPUS is a sparse row vectors of word frequencies. MODEL can be 'seeded',
% 'random', or a previously trained LDA model in a struct. The function takes
% following options.
%
% # OPTIONS
%
% _num_topics_ [8]
%
% Number of topics.
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
% _em_max_iter_ [100]
%
% Maximum number of iterations in the EM algorithm.
%
% _em_convergence_ [1e-4]
%
% Tolerance value of iterations in the EM algorithm.
%
% _estimate_alpha_ [true]
%
% Flag to enable alpha estimation.
%
% _initial_alpha_ [0.5]
%
% Initial alpha value.
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
  [model, assignment] = mex_function_(mfilename, corpus, model, options);
end