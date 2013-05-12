function corpus = load_corpus(filename)
%LOAD_CORPUS Load dataset in lda-c file format into a sparse matrix.
%
%    corpus = lda.load_corpus(filename)
%
% See also lda.save_corpus
  corpus = mex_function_(mfilename, filename);
end