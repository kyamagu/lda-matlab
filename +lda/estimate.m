function [model, distribution] = estimate(corpus, model, varargin)
%ESTIMATE Run the EM algorithm to estimate a topic model and its distribution.
%
%    [model, distribution] = lda.estimate(corpus, model, 'param1', value1, ...)
%
% CORPUS is a sparse row vectors of word frequencies. CORPUS(i, j) is a count
% of word j in document i.
%
% MODEL can be 'seeded', 'random', or a previously trained LDA model in a
% struct. The function takes following options. The resulting MODEL is a scalar
% struct of learned topics. The 'alpha' field is a scalar value of
% hyperparameter, and 'beta' is column vectors of topic distribution for each
% word.
%
% DISTRIBUTION is row vectors of topic distribution for each document. Each row
% contains weights for each topic.
%
% OPTIONS
% -------
%
% num_topics [8]
%
%  Number of topics.
%
% var_max_iter [-1]
%
%  Maximum number of iterations in variational inference. Default doesn't limit
%  the iteration by number.
%
% var_converged [1e-6]
%
%  Tolerance value of iterations in the variational inference.
%
% em_max_iter [100]
%
%  Maximum number of iterations in the EM algorithm.
%
% em_convergence [1e-4]
%
%  Tolerance value of iterations in the EM algorithm.
%
% estimate_alpha [true]
%
%  Flag to enable alpha estimation.
%
% initial_alpha [0.5]
%
%  Initial alpha value.
%
% random_seed [4357]
%
%  Seed number for random number generator.
%
% verbose [true]
%
%  Verbosity of message printout.
%
% See also lda.estimate
  options = get_options_(varargin{:});
  [model, distribution] = mex_function_(mfilename, corpus, model, options);
end
