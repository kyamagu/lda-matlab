Latent Dirichlet Allocation for Matlab
======================================

This is a Matlab version of the David Blei's original implementation of Latent
Dirichlet Allocation written in C.

http://www.cs.columbia.edu/~blei/lda-c/

The package includes a few API functions that internally calls the original C
implementation by mex interface.

Compile the code with `lda.make` function in Matlab before using the package.
Here is a quick usage example:

    corpus = lda.load_corpus('ap/ap.dat');
    [model, distribution] = lda.estimate(corpus, 'seeded', ...
                                         'num_topics', 100, ...
                                         'initial_alpha', 0.05);
    distribution = lda.infer(corpus, model);

The `corpus` is sparse row vectors of word count. `corpus(d, w)` is a count
of word `w` in document `d`. The resulting `distribution` contains row vectors
of topic weights for each document at each row.

The same example is in the `main.m` demo function.

API
---

All functions are scoped under `lda` namespace.

    estimate     Run the EM algorithm to estimate topics and distribution.
    infer        Run inference on new samples using a learned LDA model.
    load_corpus  Load dataset in lda-c file format into a sparse matrix.
    save_corpus  Save dataset in lda-c file format from a sparse matrix.
    make         Build a mex file.

Check `help` of each function for details.

License
-------

The code may be redistributed under LGPL v2.1 license.
