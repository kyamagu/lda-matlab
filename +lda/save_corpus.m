function save_corpus(corpus, filename)
%SAVE_CORPUS Save dataset in lda-c file format from a sparse matrix.
%
%    lda.save_corpus(corpus, filename)
%
% See also lda.load_corpus
  assert(issparse(corpus));
  assert(ischar(filename));
  fid = fopen(filename, 'w');
  try
    for i = 1:size(corpus, 1)
      [~, words, values] = find(corpus(i,:));
      fprintf(fid, '%d', numel(words));
      fprintf(fid, ' %d:%d', full([words-1; values]));
      fprintf(fid, '\n');
    end
  catch e
    fclose(fid);
    rethrow(e);
  end
  fclose(fid);
end