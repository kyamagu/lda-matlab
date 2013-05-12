function main
%MAIN Demonstrates the usage of the LDA API.

  % Load data from lda-c format. Get AP data from
  % http://www.cs.princeton.edu/~blei/lda-c/
  corpus = lda.load_corpus('ap/ap.dat');

  % Find the topics and document assignments.
  [model, assignment] = lda.estimate(corpus, 'seeded', ...
                                     'num_topics', 32, ...
                                     'initial_alpha', 0.1);
  % Find document assignments with a trained model.
  [likelihoods, assignment2] = lda.infer(corpus, model);

end