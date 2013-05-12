Latent Dirichlet Allocation for Matlab
======================================

This is a Matlab version of the David Blei's original implementation of Latent
Dirichlet Allocation written in C.

http://www.cs.princeton.edu/~blei/lda-c/

The package includes a few API functions that internally calls the original C
implementation by mex interface.

Compile the code with `lda.make` function in Matlab before using the package.
Here is a simple usage example:

    corpus = lda.load_corpus('ap/ap.dat');
    [model, distribution] = lda.estimate(corpus, 'seeded', ...
                                         'num_topics', 100, ...
                                         'initial_alpha', 0.1);
    [likelihoods, distribution] = lda.infer(corpus, model);

The same example is in the `main.m` demo function.

API
---

All functions are scoped under `lda` namespace.

    estimate     Run the EM algorithm to estimate a topic model and its distribution.
    infer        Run inference on new samples using a learned LDA model.
    load_corpus  Load dataset in lda-c file format into a sparse matrix.
    make         Build a mex file.
    save_corpus  Save dataset in lda-c file format from a sparse matrix.

License
-------

The code may be redistributed under LGPL v2.1 license.
