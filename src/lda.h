// (C) Copyright 2004, David M. Blei (blei [at] cs [dot] cmu [dot] edu)

// This file is part of LDA-C.

// LDA-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// LDA-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#ifndef __LDA_H__
#define __LDA_H__

#include "cokus.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int* words;
    int* counts;
    int length;
    int total;
} document;

typedef struct
{
    document* docs;
    int num_terms;
    int num_docs;
} lda_corpus;

typedef struct
{
    double alpha;
    double** log_prob_w;
    int num_topics;
    int num_terms;
} lda_model;

typedef struct
{
    double** class_word;
    double* class_total;
    double alpha_suffstats;
    int num_docs;
    int num_topics;
} lda_suffstats;

typedef struct {
    float VAR_CONVERGED;
    int VAR_MAX_ITER;
    int LAG;
    float EM_CONVERGED;
    int EM_MAX_ITER;
    int ESTIMATE_ALPHA;
    double INITIAL_ALPHA;
    int NTOPICS;
    unsigned int RANDOM_SEED;
    int VERBOSE;
} lda_settings;

// Model API.
void free_lda_model(lda_model*);
void save_lda_model(lda_model*, const char*);
lda_model* new_lda_model(int, int);
void free_lda_suffstats(lda_suffstats*);
lda_suffstats* new_lda_suffstats(lda_model* model);
void corpus_initialize_ss(lda_suffstats* ss, lda_model* model, lda_corpus* c);
void random_initialize_ss(lda_suffstats* ss, lda_model* model);
void zero_initialize_ss(lda_suffstats* ss, lda_model* model);
void lda_mle(lda_model* model, lda_suffstats* ss, int estimate_alpha);
lda_model* load_lda_model(const char* model_root);
lda_settings* get_settings();

// Inference API.
double lda_inference(document*, lda_model*, double*, double**);
double compute_likelihood(document*, lda_model*, double**, double*);

// Optimization API.
double alhood(double a, double ss, int D, int K);
double d_alhood(double a, double ss, int D, int K);
double d2_alhood(double a, int D, int K);
double opt_alpha(double ss, int D, int K);
void maximize_alpha(double** gamma, lda_model* model, int num_docs);

// High-level API.
double doc_e_step(document* doc,
                  double* gamma,
                  double** phi,
                  lda_model* model,
                  lda_suffstats* ss);

void save_gamma(const char* filename,
                double** gamma,
                int num_docs,
                int num_topics);

void run_em(const char* start,
            const char* directory,
            lda_corpus* corpus);

// Data API.
void read_settings(const char* filename);

void infer(const char* model_root,
           const char* save,
           lda_corpus* corpus);

lda_corpus* read_data(const char* data_filename);
int max_corpus_length(const lda_corpus* c);
void free_data(lda_corpus* c);

// Utility API.
double log_sum(double log_a, double log_b);
double trigamma(double x);
double digamma(double x);
double log_gamma(double x);
void make_directory(const char* name);
int argmax(double* x, int n);

#ifdef __cplusplus
}
#endif

#endif // __LDA_H__
