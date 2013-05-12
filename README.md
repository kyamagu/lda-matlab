Latent Dirichlet Allocation for Matlab
======================================

This is a Matlab version of the David Blei's original implementation of Latent
Dirichlet Allocation written in C.

http://www.cs.princeton.edu/~blei/lda-c/

The package includes a few API functions
that internally calls the original C implementation by mex interface.

Compile the code with `lda.make` function in Matlab before using the package.

Here is a simple usage example:

    corpus = lda.load_corpus('ap/ap.dat');
    [model, assignment] = lda.estimate(corpus, 'seeded', ...
                                       'num_topics', 100, ...
                                       'initial_alpha', 0.1);
    [likelihoods, assignment2] = lda.infer(corpus, model);

The code is under LGPL v2.1 license.
