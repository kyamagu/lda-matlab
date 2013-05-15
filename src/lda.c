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

#include "lda.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef MATLAB_MEX_FILE
int mexPrintf(const char *message, ...);
#define printf if (settings_.VERBOSE) mexPrintf
#endif

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 1
#define NEWTON_THRESH 1e-5
#define MAX_ALPHA_ITER 1000
#define OFFSET 0;                  // offset for reading data

static lda_settings settings_ = {1e-6, -1, 5, 1e-4, 100, 1, 0.5, 8, 4357U, 1};

lda_settings* get_settings() { return &settings_; }


/*
 * variational inference
 *
 */

double lda_inference(document* doc, lda_model* model, double* var_gamma, double** phi)
{
    double converged = 1;
    double phisum = 0, likelihood = 0;
    double likelihood_old = 0, oldphi[model->num_topics];
    int k, n, var_iter;
    double digamma_gam[model->num_topics];

    // compute posterior dirichlet

    for (k = 0; k < model->num_topics; k++)
    {
        var_gamma[k] = model->alpha + (doc->total/((double) model->num_topics));
        digamma_gam[k] = digamma(var_gamma[k]);
        for (n = 0; n < doc->length; n++)
            phi[n][k] = 1.0/model->num_topics;
    }
    var_iter = 0;

    while ((converged > settings_.VAR_CONVERGED) &&
     ((var_iter < settings_.VAR_MAX_ITER) || (settings_.VAR_MAX_ITER == -1)))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++)
        {
            phisum = 0;
            for (k = 0; k < model->num_topics; k++)
            {
                oldphi[k] = phi[n][k];
                phi[n][k] =
                digamma_gam[k] +
                model->log_prob_w[k][doc->words[n]];

                if (k > 0)
                    phisum = log_sum(phisum, phi[n][k]);
                else
                    phisum = phi[n][k]; // note, phi is in log space
            }

            for (k = 0; k < model->num_topics; k++)
            {
                phi[n][k] = exp(phi[n][k] - phisum);
                var_gamma[k] =
                    var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
                // !!! a lot of extra digamma's here because of how we're computing it
                // !!! but its more automatically updated too.
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        likelihood = compute_likelihood(doc, model, phi, var_gamma);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;

        // printf("[LDA INF] %8.5f %1.3e\n", likelihood, converged);
    }
    return(likelihood);
}


/*
 * compute likelihood bound
 *
 */

double
compute_likelihood(document* doc, lda_model* model, double** phi, double* var_gamma)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0, dig[model->num_topics];
    int k, n;

    for (k = 0; k < model->num_topics; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    digsum = digamma(var_gamma_sum);

    likelihood =
        lgamma(model->alpha * model -> num_topics)
        - model -> num_topics * lgamma(model->alpha)
        - (lgamma(var_gamma_sum));

    for (k = 0; k < model->num_topics; k++)
    {
        likelihood +=
            (model->alpha - 1)*(dig[k] - digsum) + lgamma(var_gamma[k])
            - (var_gamma[k] - 1)*(dig[k] - digsum);

        for (n = 0; n < doc->length; n++)
        {
            if (phi[n][k] > 0)
            {
                likelihood += doc->counts[n]*
                    (phi[n][k]*((dig[k] - digsum) - log(phi[n][k])
                        + model->log_prob_w[k][doc->words[n]]));
            }
        }
    }
    return(likelihood);
}


/*
 * compute MLE lda model from sufficient statistics
 *
 */

void lda_mle(lda_model* model, lda_suffstats* ss, int estimate_alpha)
{
    int k; int w;

    for (k = 0; k < model->num_topics; k++)
    {
        for (w = 0; w < model->num_terms; w++)
        {
            if (ss->class_word[k][w] > 0)
            {
                model->log_prob_w[k][w] =
                    log(ss->class_word[k][w]) -
                    log(ss->class_total[k]);
            }
            else
                model->log_prob_w[k][w] = -100;
        }
    }
    if (estimate_alpha == 1)
    {
        model->alpha = opt_alpha(ss->alpha_suffstats,
                                 ss->num_docs,
                                 model->num_topics);

        printf("new alpha = %5.5f\n", model->alpha);
    }
}

/*
 * allocate sufficient statistics
 *
 */

lda_suffstats* new_lda_suffstats(lda_model* model)
{
    int num_topics = model->num_topics;
    int num_terms = model->num_terms;
    int i,j;

    lda_suffstats* ss = malloc(sizeof(lda_suffstats));
    ss->class_total = malloc(sizeof(double)*num_topics);
    ss->class_word = malloc(sizeof(double*)*num_topics);
    for (i = 0; i < num_topics; i++)
    {
        ss->class_total[i] = 0;
        ss->class_word[i] = malloc(sizeof(double)*num_terms);
        for (j = 0; j < num_terms; j++)
        {
            ss->class_word[i][j] = 0;
        }
    }
    ss->num_topics = model->num_topics;
    return(ss);
}

void free_lda_suffstats(lda_suffstats* ss)
{
    int i;
    for (i = 0; i < ss->num_topics; i++)
        free(ss->class_word[i]);
    free(ss->class_word);
    free(ss->class_total);
    free(ss);
}


/*
 * various intializations for the sufficient statistics
 *
 */

void zero_initialize_ss(lda_suffstats* ss, lda_model* model)
{
    int k, w;
    for (k = 0; k < model->num_topics; k++)
    {
        ss->class_total[k] = 0;
        for (w = 0; w < model->num_terms; w++)
        {
            ss->class_word[k][w] = 0;
        }
    }
    ss->num_docs = 0;
    ss->alpha_suffstats = 0;
}


void random_initialize_ss(lda_suffstats* ss, lda_model* model)
{
    int num_topics = model->num_topics;
    int num_terms = model->num_terms;
    int k, n;
    for (k = 0; k < num_topics; k++)
    {
        for (n = 0; n < num_terms; n++)
        {
            ss->class_word[k][n] += 1.0/num_terms + myrand();
            ss->class_total[k] += ss->class_word[k][n];
        }
    }
}


void corpus_initialize_ss(lda_suffstats* ss, lda_model* model, lda_corpus* c)
{
    int num_topics = model->num_topics;
    int i, k, d, n;
    document* doc;

    for (k = 0; k < num_topics; k++)
    {
        for (i = 0; i < NUM_INIT; i++)
        {
            d = floor(myrand() * c->num_docs);
            printf("initialized with document %d\n", d);
            doc = &(c->docs[d]);
            for (n = 0; n < doc->length; n++)
            {
                ss->class_word[k][doc->words[n]] += doc->counts[n];
            }
        }
        for (n = 0; n < model->num_terms; n++)
        {
            ss->class_word[k][n] += 1.0;
            ss->class_total[k] = ss->class_total[k] + ss->class_word[k][n];
        }
    }
}

/*
 * allocate new lda model
 *
 */

lda_model* new_lda_model(int num_terms, int num_topics)
{
    int i,j;
    lda_model* model;

    model = malloc(sizeof(lda_model));
    model->num_topics = num_topics;
    model->num_terms = num_terms;
    model->alpha = 1.0;
    model->log_prob_w = malloc(sizeof(double*)*num_topics);
    for (i = 0; i < num_topics; i++)
    {
        model->log_prob_w[i] = malloc(sizeof(double)*num_terms);
        for (j = 0; j < num_terms; j++)
            model->log_prob_w[i][j] = 0;
    }
    return(model);
}


/*
 * deallocate new lda model
 *
 */

void free_lda_model(lda_model* model)
{
    int i;

    for (i = 0; i < model->num_topics; i++)
    {
        free(model->log_prob_w[i]);
    }
    free(model->log_prob_w);
    free(model);
}


/*
 * save an lda model
 *
 */

void save_lda_model(lda_model* model, const char* model_root)
{
    char filename[100];
    FILE* fileptr;
    int i, j;

    sprintf(filename, "%s.beta", model_root);
    fileptr = fopen(filename, "w");
    for (i = 0; i < model->num_topics; i++)
    {
        for (j = 0; j < model->num_terms; j++)
        {
            fprintf(fileptr, " %5.10f", model->log_prob_w[i][j]);
        }
        fprintf(fileptr, "\n");
    }
    fclose(fileptr);

    sprintf(filename, "%s.other", model_root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "num_topics %d\n", model->num_topics);
    fprintf(fileptr, "num_terms %d\n", model->num_terms);
    fprintf(fileptr, "alpha %5.10f\n", model->alpha);
    fclose(fileptr);
}


lda_model* load_lda_model(const char* model_root)
{
    char filename[100];
    FILE* fileptr;
    int i, j, num_terms, num_topics;
    float x, alpha;

    sprintf(filename, "%s.other", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "num_topics %d\n", &num_topics);
    fscanf(fileptr, "num_terms %d\n", &num_terms);
    fscanf(fileptr, "alpha %f\n", &alpha);
    fclose(fileptr);

    lda_model* model = new_lda_model(num_terms, num_topics);
    model->alpha = alpha;

    sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (i = 0; i < num_topics; i++)
    {
        for (j = 0; j < num_terms; j++)
        {
            fscanf(fileptr, "%f", &x);
            model->log_prob_w[i][j] = x;
        }
    }
    fclose(fileptr);
    return(model);
}


/*
 * objective function and its derivatives
 *
 */

double alhood(double a, double ss, int D, int K)
{ return(D * (lgamma(K * a) - K * lgamma(a)) + (a - 1) * ss); }

double d_alhood(double a, double ss, int D, int K)
{ return(D * (K * digamma(K * a) - K * digamma(a)) + ss); }

double d2_alhood(double a, int D, int K)
{ return(D * (K * K * trigamma(K * a) - K * trigamma(a))); }


/*
 * newtons method
 *
 */

double opt_alpha(double ss, int D, int K)
{
    double a, log_a, init_a = 100;
    double f, df, d2f;
    int iter = 0;

    log_a = log(init_a);
    do
    {
        iter++;
        a = exp(log_a);
        if (isnan(a))
        {
            init_a = init_a * 10;
            printf("warning : alpha is nan; new init = %5.5f\n", init_a);
            a = init_a;
            log_a = log(a);
        }
        f = alhood(a, ss, D, K);
        df = d_alhood(a, ss, D, K);
        d2f = d2_alhood(a, D, K);
        log_a = log_a - df/(d2f * a + df);
        printf("alpha maximization : %5.5f   %5.5f\n", f, df);
    }
    while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
    return(exp(log_a));
}


/*
 * perform inference on a document and update sufficient statistics
 *
 */

double doc_e_step(document* doc, double* gamma, double** phi,
                  lda_model* model, lda_suffstats* ss)
{
    double likelihood;
    int n, k;

    // posterior inference

    likelihood = lda_inference(doc, model, gamma, phi);

    // update sufficient statistics

    double gamma_sum = 0;
    for (k = 0; k < model->num_topics; k++)
    {
        gamma_sum += gamma[k];
        ss->alpha_suffstats += digamma(gamma[k]);
    }
    ss->alpha_suffstats -= model->num_topics * digamma(gamma_sum);

    for (n = 0; n < doc->length; n++)
    {
        for (k = 0; k < model->num_topics; k++)
        {
            ss->class_word[k][doc->words[n]] += doc->counts[n]*phi[n][k];
            ss->class_total[k] += doc->counts[n]*phi[n][k];
        }
    }

    ss->num_docs = ss->num_docs + 1;

    return(likelihood);
}


/*
 * writes the word assignments line for a document to a file
 *
 */

void write_word_assignment(FILE* f, document* doc, double** phi, lda_model* model)
{
    int n;

    fprintf(f, "%03d", doc->length);
    for (n = 0; n < doc->length; n++)
    {
        fprintf(f, " %04d:%02d",
            doc->words[n], argmax(phi[n], model->num_topics));
    }
    fprintf(f, "\n");
    fflush(f);
}


/*
 * saves the gamma parameters of the current dataset
 *
 */

void save_gamma(const char* filename, double** gamma, int num_docs, int num_topics)
{
    FILE* fileptr;
    int d, k;
    fileptr = fopen(filename, "w");

    for (d = 0; d < num_docs; d++)
    {
        fprintf(fileptr, "%5.10f", gamma[d][0]);
        for (k = 1; k < num_topics; k++)
        {
            fprintf(fileptr, " %5.10f", gamma[d][k]);
        }
        fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}


/*
 * run_em
 *
 */

void run_em(const char* start, const char* directory, lda_corpus* corpus)
{

    int d, n;
    lda_model *model = NULL;
    double **var_gamma, **phi;

    // allocate variational parameters

    var_gamma = malloc(sizeof(double*)*(corpus->num_docs));
    for (d = 0; d < corpus->num_docs; d++)
        var_gamma[d] = malloc(sizeof(double) * settings_.NTOPICS);

    int max_length = max_corpus_length(corpus);
    phi = malloc(sizeof(double*)*max_length);
    for (n = 0; n < max_length; n++)
        phi[n] = malloc(sizeof(double) * settings_.NTOPICS);

    // initialize model

    char filename[100];

    lda_suffstats* ss = NULL;
    if (strcmp(start, "seeded")==0)
    {
        model = new_lda_model(corpus->num_terms, settings_.NTOPICS);
        ss = new_lda_suffstats(model);
        corpus_initialize_ss(ss, model, corpus);
        lda_mle(model, ss, 0);
        model->alpha = settings_.INITIAL_ALPHA;
    }
    else if (strcmp(start, "random")==0)
    {
        model = new_lda_model(corpus->num_terms, settings_.NTOPICS);
        ss = new_lda_suffstats(model);
        random_initialize_ss(ss, model);
        lda_mle(model, ss, 0);
        model->alpha = settings_.INITIAL_ALPHA;
    }
    else
    {
        model = load_lda_model(start);
        ss = new_lda_suffstats(model);
    }

    sprintf(filename,"%s/000",directory);
    save_lda_model(model, filename);

    // run expectation maximization

    int i = 0;
    double likelihood, likelihood_old = 0, converged = 1;
    sprintf(filename, "%s/likelihood.dat", directory);
    FILE* likelihood_file = fopen(filename, "w");

    while (((converged < 0) || (converged > settings_.EM_CONVERGED) || 
            (i <= 2)) && (i <= settings_.EM_MAX_ITER))
    {
        i++; printf("**** em iteration %d ****\n", i);
        likelihood = 0;
        zero_initialize_ss(ss, model);

        // e-step

        for (d = 0; d < corpus->num_docs; d++)
        {
            if ((d % 1000) == 0) printf("document %d\n",d);
            likelihood += doc_e_step(&(corpus->docs[d]),
               var_gamma[d],
               phi,
               model,
               ss);
        }

        // m-step

        lda_mle(model, ss, settings_.ESTIMATE_ALPHA);

        // check for convergence

        converged = (likelihood_old - likelihood) / (likelihood_old);
        if (converged < 0) settings_.VAR_MAX_ITER = settings_.VAR_MAX_ITER * 2;
        likelihood_old = likelihood;

        // output model and likelihood

        fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
        fflush(likelihood_file);
        if ((i % settings_.LAG) == 0)
        {
            sprintf(filename,"%s/%03d",directory, i);
            save_lda_model(model, filename);
            sprintf(filename,"%s/%03d.gamma",directory, i);
            save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);
        }
    }

    // output the final model

    sprintf(filename,"%s/final",directory);
    save_lda_model(model, filename);
    sprintf(filename,"%s/final.gamma",directory);
    save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);

    // output the word assignments (for visualization)

    sprintf(filename, "%s/word-assignments.dat", directory);
    FILE* w_asgn_file = fopen(filename, "w");
    for (d = 0; d < corpus->num_docs; d++)
    {
        if ((d % 100) == 0) printf("final e step document %d\n",d);
        likelihood += lda_inference(&(corpus->docs[d]), model, var_gamma[d], phi);
        write_word_assignment(w_asgn_file, &(corpus->docs[d]), phi, model);
    }
    fclose(w_asgn_file);
    fclose(likelihood_file);
    free_lda_model(model);
    free_lda_suffstats(ss);
    for (d = 0; d < corpus->num_docs; d++)
        free(var_gamma[d]);
    free(var_gamma);
    for (n = 0; n < max_length; n++)
        free(phi[n]);
    free(phi);
}


/*
 * read settings.
 *
 */

void read_settings(const char* filename)
{
    FILE* fileptr;
    char alpha_action[100];
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "var max iter %d\n", &settings_.VAR_MAX_ITER);
    fscanf(fileptr, "var convergence %f\n", &settings_.VAR_CONVERGED);
    fscanf(fileptr, "em max iter %d\n", &settings_.EM_MAX_ITER);
    fscanf(fileptr, "em convergence %f\n", &settings_.EM_CONVERGED);
    fscanf(fileptr, "alpha %s", alpha_action);
    if (strcmp(alpha_action, "fixed")==0)
    {
        settings_.ESTIMATE_ALPHA = 0;
    }
    else
    {
        settings_.ESTIMATE_ALPHA = 1;
    }
    fclose(fileptr);
}


/*
 * inference only
 *
 */

void infer(const char* model_root, const char* save, lda_corpus* corpus)
{
    FILE* fileptr;
    char filename[100];
    int i, d, n;
    lda_model *model;
    double **var_gamma, likelihood, **phi;
    document* doc;

    model = load_lda_model(model_root);
    var_gamma = malloc(sizeof(double*)*(corpus->num_docs));
    for (i = 0; i < corpus->num_docs; i++)
        var_gamma[i] = malloc(sizeof(double)*model->num_topics);
    sprintf(filename, "%s-lda-lhood.dat", save);
    fileptr = fopen(filename, "w");
    for (d = 0; d < corpus->num_docs; d++)
    {
        if (((d % 100) == 0) && (d>0)) printf("document %d\n",d);

        doc = &(corpus->docs[d]);
        phi = (double**) malloc(sizeof(double*) * doc->length);
        for (n = 0; n < doc->length; n++)
            phi[n] = (double*) malloc(sizeof(double) * model->num_topics);
        likelihood = lda_inference(doc, model, var_gamma[d], phi);
        for (n = 0; n < doc->length; n++)
            free(phi[n]);
        free(phi);

        fprintf(fileptr, "%5.5f\n", likelihood);
    }
    fclose(fileptr);
    sprintf(filename, "%s-gamma.dat", save);
    save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);
    free_lda_model(model);
    for (i = 0; i < corpus->num_docs; i++)
        free(var_gamma[i]);
    free(var_gamma);
}


/*
 * update sufficient statistics
 *
 */

lda_corpus* read_data(const char* data_filename)
{
    FILE *fileptr = NULL;
    int length, count, word, n, nd, nw;
    lda_corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(lda_corpus));
    if (c == NULL)
        return NULL;
    c->docs = 0;
    c->num_terms = 0;
    c->num_docs = 0;
    fileptr = fopen(data_filename, "r");
    if (fileptr == NULL) {
        perror(data_filename);
        goto error;
    }
    nd = 0; nw = 0;
    while ((fscanf(fileptr, "%10d", &length) != EOF))
    {
        c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
        c->docs[nd].length = length;
        c->docs[nd].total = 0;
        c->docs[nd].words = malloc(sizeof(int)*length);
        c->docs[nd].counts = malloc(sizeof(int)*length);
        for (n = 0; n < length; n++)
        {
            if (fscanf(fileptr, "%10d:%10d", &word, &count) != 2) {
                perror("invalid file format");
                goto error;
            }
            word = word - OFFSET;
            c->docs[nd].words[n] = word;
            c->docs[nd].counts[n] = count;
            c->docs[nd].total += count;
            if (word >= nw) { nw = word + 1; }
        }
        nd++;
    }
    if (feof(fileptr)) {
        perror("invalid file format"); // TODO: free
        goto error;
    }
    fclose(fileptr);
    c->num_docs = nd;
    c->num_terms = nw;
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
    return(c);
error:
    for (n = 0; n < nd; n++) {
        free(c->docs[n].words);
        free(c->docs[n].counts);
    }
    free(c->docs);
    free(c);
    if (fileptr)
        fclose(fileptr);
    return NULL;
}

int max_corpus_length(const lda_corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->num_docs; n++)
        if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}

void free_data(lda_corpus* c)
{
    int n;
    for (n = 0; n < c->num_docs; n++)
    {
        free(c->docs[n].words);
        free(c->docs[n].counts);
    }
    free(c->docs);
    free(c);
}

/*
 * given log(a) and log(b), return log(a + b)
 *
 */

double log_sum(double log_a, double log_b)
{
  double v;

  if (log_a < log_b)
  {
      v = log_b+log(1 + exp(log_a-log_b));
  }
  else
  {
      v = log_a+log(1 + exp(log_b-log_a));
  }
  return(v);
}

 /**
   * Proc to calculate the value of the trigamma, the second
   * derivative of the loggamma function. Accepts positive matrices.
   * From Abromowitz and Stegun.  Uses formulas 6.4.11 and 6.4.12 with
   * recurrence formula 6.4.6.  Each requires workspace at least 5
   * times the size of X.
   *
   **/

double trigamma(double x)
{
    double p;
    int i;

    x=x+6;
    p=1/(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
       *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for (i=0; i<6 ;i++)
    {
        x=x-1;
        p=1/(x*x)+p;
    }
    return(p);
}


/*
 * taylor approximation of first derivative of the log gamma function
 *
 */

double digamma(double x)
{
    double p;
    x=x+6;
    p=1/(x*x);
    p=(((0.004166666666667*p-0.003968253986254)*p+
        0.008333333333333)*p-0.083333333333333)*p;
    p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
    return p;
}


double log_gamma(double x)
{
   double z=1/(x*x);

   x=x+6;
   z=(((-0.000595238095238*z+0.000793650793651)
    *z-0.002777777777778)*z+0.083333333333333)/x;
   z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)-
   log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
   return z;
}



/*
 * make directory
 *
 */

void make_directory(const char* name)
{
    mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
}


/*
 * argmax
 *
 */

int argmax(double* x, int n)
{
    int i;
    double max = x[0];
    int argmax = 0;
    for (i = 1; i < n; i++)
    {
        if (x[i] > max)
        {
            max = x[i];
            argmax = i;
        }
    }
    return(argmax);
}


