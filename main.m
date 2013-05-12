function main
%MAIN Demonstrates the usage of the LDA API.

  % Load data from lda-c format. Get AP data from
  % http://www.cs.princeton.edu/~blei/lda-c/
  corpus = lda.load_corpus('ap/ap.dat');

  % Find the topics and topic distribution for documents.
  [model, distribution] = lda.estimate(corpus, 'seeded', ...
                                       'num_topics', 32, ...
                                       'initial_alpha', 0.1);
  % Compute topic distribution with a trained model.
  [distribution2, likelihoods] = lda.infer(corpus, model);

end
