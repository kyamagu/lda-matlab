// Matlab driver main.
//
// Kota Yamaguchi 2013 <kyamagu@cs.stonybrook.edu>

#include "lda.h"
#include "mex/arguments.h"
#include "mex/function.h"
#include "mex/mxarray.h"
#include "mex.h"
#include <numeric>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

using mex::CheckInputArguments;
using mex::CheckOutputArguments;
using mex::MxArray;

// Alias for the mex error function.
#define ERROR(...) mexErrMsgIdAndTxt("lda:error", __VA_ARGS__)

#define CHECK_NOTNULL(pointer) \
  if (pointer == NULL) ERROR("Null pointer exception.")

#define MEX_FLUSH mexEvalString("drawnow")

namespace {

// Reset random number.
void ResetRandomNumber() {
  seedMT(get_settings()->RANDOM_SEED);
}

// Read settings from a struct.
void ReadSettings(const mxArray* mxarray) {
  MxArray config(mxarray);
  lda_settings* settings = get_settings();
  if (config.isField("var_max_iter"))
    settings->VAR_MAX_ITER = config.at("var_max_iter").toInt();
  if (config.isField("var_converged"))
    settings->VAR_CONVERGED = config.at("var_converged").toDouble();
  if (config.isField("em_max_iter"))
    settings->EM_MAX_ITER = config.at("em_max_iter").toInt();
  if (config.isField("em_convergence"))
    settings->EM_CONVERGED = config.at("em_convergence").toDouble();
  if (config.isField("estimate_alpha"))
    settings->ESTIMATE_ALPHA = config.at("estimate_alpha").toInt();
  if (config.isField("initial_alpha"))
    settings->INITIAL_ALPHA = config.at("initial_alpha").toDouble();
  if (config.isField("num_topics"))
    settings->NTOPICS = config.at("num_topics").toInt();
  if (config.isField("random_seed"))
    settings->RANDOM_SEED = config.at("random_seed").toInt();
  if (config.isField("verbose"))
    settings->VERBOSE = config.at("verbose").toInt();
}

// Read data into corpus. Call free_data() after use.
lda_corpus* ReadCorpusFromMxArray(const mxArray* input) {
  if (!mxIsSparse(input) || !mxIsDouble(input))
    ERROR("Input data must be a sparse double array.");
  lda_corpus* corpus = (lda_corpus*)malloc(sizeof(corpus));
  CHECK_NOTNULL(corpus);
  // Get values from mxArray.
  vector<vector<pair<int, int> > > data(mxGetM(input));
  mwIndex* row_pointer = mxGetIr(input);
  mwIndex* column_pointer = mxGetJc(input);
  double* data_pointer = mxGetPr(input);
  for (int j = 0; j < mxGetN(input); ++j) {
    int column_elements = column_pointer[j+1] - column_pointer[j];
    for (int i = 0; i < column_elements; ++i) {
      pair<int, int> value(j, static_cast<int>(*data_pointer++));
      data[*row_pointer++].push_back(value);
    }
  }
  // Fill in the corpus.
  corpus->num_docs = data.size();
  corpus->num_terms = mxGetN(input);
  corpus->docs = (document*)malloc(sizeof(document)*data.size());
  for (int i = 0; i < data.size(); ++i) {
    const vector<pair<int, int> >& row = data[i];
    document* doc = &corpus->docs[i];
    doc->length = row.size();
    doc->total = 0;
    doc->words = (int*)malloc(sizeof(int) * row.size());
    doc->counts = (int*)malloc(sizeof(int) * row.size());
    for (int j = 0; j < row.size(); ++j) {
      doc->words[j] = row[j].first;
      doc->counts[j] = row[j].second;
      doc->total += row[j].second;
    }
  }
  return corpus;
}

// Write corpus data into an mxArray.
mxArray* WriteCorpusToMxArray(const lda_corpus* corpus) {
  CHECK_NOTNULL(corpus);
  // Get columns from corpus.
  int num_elements = 0;
  vector<vector<pair<int, int> > > data(corpus->num_terms);
  for (int i = 0; i < corpus->num_docs; ++i) {
    const document& doc = corpus->docs[i];
    for (int j = 0; j < doc.length; ++j) {
      pair<int, int> value(i, doc.counts[j]);
      data[doc.words[j]].push_back(value);
    }
    num_elements += doc.length;
  }
  // Fill in the mxArray.
  mxArray* output = mxCreateSparse(corpus->num_docs,
                                   corpus->num_terms,
                                   num_elements,
                                   mxREAL);
  CHECK_NOTNULL(output);
  mwIndex* row_pointer = mxGetIr(output);
  mwIndex* column_pointer = mxGetJc(output);
  double* data_pointer = mxGetPr(output);
  column_pointer[0] = 0;
  for (int j = 0; j < mxGetN(output); ++j) {
    const vector<pair<int, int> >& column = data[j];
    column_pointer[j+1] = column_pointer[j] + column.size();
    for (int i = 0; i < column.size(); ++i) {
      *row_pointer++ = column[i].first;
      *data_pointer++ = column[i].second;
    }
  }
  return output;
}

// Read LDA model data from mxArray.
lda_model* ReadModelFromMxArray(const mxArray* array) {
  MxArray input(array);
  lda_model* model = new_lda_model(input.at("num_terms").toInt(),
                                   input.at("num_topics").toInt());
  CHECK_NOTNULL(model);
  model->alpha = input.at("alpha").toDouble();
  const double* data = mxGetPr(input.at("beta").get());
  for (int i = 0; i < model->num_topics; ++i)
    for (int j = 0; j < model->num_terms; ++j)
      model->log_prob_w[i][j] = data[i + model->num_topics * j];
  return model;
}

// Write LDA model data into mxArray.
mxArray* WriteModelToMxArray(const lda_model* model) {
  const char* fields[] = {"num_topics", "num_terms", "alpha", "beta"};
  MxArray output = MxArray::Struct(4, fields);
  output.set("num_topics", model->num_topics);
  output.set("num_terms", model->num_terms);
  output.set("alpha", model->alpha);
  mxArray* beta = mxCreateDoubleMatrix(model->num_topics,
                                       model->num_terms,
                                       mxREAL);
  double* data = mxGetPr(beta);
  for (int i = 0; i < model->num_topics; ++i)
    for (int j = 0; j < model->num_terms; ++j)
      data[i + model->num_topics * j] = model->log_prob_w[i][j];
  output.set("beta", beta);
  return output.getMutable();
}

// Write topic gamma into mxArray.
mxArray* WriteGammaToMxArray(const vector<vector<double> >& var_gamma) {
  int rows = var_gamma.size();
  int columns = (var_gamma.empty()) ? 0 : var_gamma[0].size();
  mxArray* output = mxCreateDoubleMatrix(rows, columns, mxREAL);
  CHECK_NOTNULL(output);
  double* data = mxGetPr(output);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < columns; ++j)
      data[j + columns * i] = var_gamma[i][j];
  return output;
}

// Initialize LDA model according to the configuration.
void InitializeModel(const mxArray* model_input,
                     lda_corpus* corpus,
                     lda_model** model,
                     lda_suffstats** ss) {
  CHECK_NOTNULL(corpus);
  CHECK_NOTNULL(model);
  CHECK_NOTNULL(ss);
  const lda_settings* settings = get_settings();
  if (mxIsChar(model_input)) {
    string initialization_method = MxArray(model_input).toString();
    if (initialization_method == "seeded") {
      *model = new_lda_model(corpus->num_terms, settings->NTOPICS);
      *ss = new_lda_suffstats(*model);
      corpus_initialize_ss(*ss, *model, corpus);
      lda_mle(*model, *ss, 0);
      (*model)->alpha = settings->INITIAL_ALPHA;
    }
    else if (initialization_method == "random") {
      *model = new_lda_model(corpus->num_terms, settings->NTOPICS);
      *ss = new_lda_suffstats(*model);
      random_initialize_ss(*ss, *model);
      lda_mle(*model, *ss, 0);
      (*model)->alpha = settings->INITIAL_ALPHA;
    }
    else
      ERROR("Invalid model initialization: %s", initialization_method.c_str());
  }
  else {
    *model = ReadModelFromMxArray(model_input);
    *ss = new_lda_suffstats(*model);
  }
}

// Estimate topics and parameters of the model.
void RunEM(const mxArray* model_input,
           lda_corpus* corpus,
           mxArray** model_output,
           mxArray** gamma) {
  // Allocate variational parameters.
  lda_settings* settings = get_settings();
  vector<vector<double> > var_gamma(corpus->num_docs,
                                    vector<double>(settings->NTOPICS));
  vector<double*> phi(max_corpus_length(corpus));
  for (int i = 0; i < phi.size(); ++i)
    phi[i] = (double*)malloc(sizeof(double) * settings->NTOPICS);
  // Initialize model.
  lda_model *model = NULL;
  lda_suffstats *ss = NULL;
  InitializeModel(model_input, corpus, &model, &ss);
  // EM main loop.
  int i = 0;
  double likelihood, likelihood_old = 0, converged = 1;
  while (((converged < 0) || (converged > settings->EM_CONVERGED) || 
        (i <= 2)) && (i <= settings->EM_MAX_ITER)) {
    i++;
    if (get_settings()->VERBOSE)
      mexPrintf("**** em iteration %d ****\n", i);
    likelihood = 0;
    zero_initialize_ss(ss, model);
    // E-step.
    for (int d = 0; d < corpus->num_docs; ++d) {
      if ((d % 1000) == 0 && get_settings()->VERBOSE)
        mexPrintf("document %d\n", d);
      likelihood += doc_e_step(&(corpus->docs[d]),
                               &var_gamma[d][0],
                               &phi[0],
                               model,
                               ss);
    }
    // M-step.
    lda_mle(model, ss, settings->ESTIMATE_ALPHA);
    // check for convergence.
    converged = (likelihood_old - likelihood) / (likelihood_old);
    if (converged < 0)
      settings->VAR_MAX_ITER = settings->VAR_MAX_ITER * 2;
    likelihood_old = likelihood;
    if (get_settings()->VERBOSE)
      mexPrintf("likelihood = %10.10f converged = %5.5e\n",
                likelihood,
                converged);
    MEX_FLUSH;
  }
  // Prepare output.
  if (model_output != NULL)
    *model_output = WriteModelToMxArray(model);
  if (gamma != NULL)
    *gamma = WriteGammaToMxArray(var_gamma);
  // Cleanup.
  free_lda_model(model);
  free_lda_suffstats(ss);
  for (int i = 0; i < phi.size(); ++i)
    free(phi[i]);
}


// Run inference with a learned model.
void RunInference(const mxArray* model_input,
                  lda_corpus* corpus,
                  mxArray** gamma,
                  mxArray** likelihood_values) {
  lda_model* model = ReadModelFromMxArray(model_input);
  vector<vector<double> > var_gamma(corpus->num_docs,
                                    vector<double>(model->num_topics));
  vector<double> likelihoods(corpus->num_docs);
  for (int d = 0; d < corpus->num_docs; ++d) {
    if (((d % 1000) == 0) && (d > 0) && get_settings()->VERBOSE)
      mexPrintf("document %d\n", d);
    document* doc = &corpus->docs[d];
    vector<double*> phi(doc->length);
    for (int n = 0; n < doc->length; n++)
      phi[n] = (double*)malloc(sizeof(double) * model->num_topics);
    likelihoods[d] = lda_inference(doc, model, &var_gamma[d][0], &phi[0]);
    for (int n = 0; n < doc->length; n++)
      free(phi[n]);
  }
  if (likelihood_values != NULL) {
    *likelihood_values = mxCreateDoubleMatrix(likelihoods.size(), 1, mxREAL);
    copy(likelihoods.begin(), likelihoods.end(), mxGetPr(*likelihood_values));
  }
  if (gamma != NULL)
    *gamma = WriteGammaToMxArray(var_gamma);
}

// Main entry for corpus = load_corpus(filename).
MEX_FUNCTION(load_corpus) (int nlhs,
                           mxArray* plhs[],
                           int nrhs,
                           const mxArray* prhs[]) {
  CheckInputArguments(1, 1, nrhs);
  CheckOutputArguments(0, 1, nlhs);
  lda_corpus* corpus = read_data(MxArray(prhs[0]).toString().c_str());
  plhs[0] = WriteCorpusToMxArray(corpus);
  free_data(corpus);
}

// Main entry for [model, distribution] = estimate(corpus, model, options).
MEX_FUNCTION(estimate) (int nlhs,
                        mxArray* plhs[],
                        int nrhs,
                        const mxArray* prhs[]) {
  CheckInputArguments(3, 3, nrhs);
  CheckOutputArguments(0, 2, nlhs);
  lda_corpus* corpus = ReadCorpusFromMxArray(prhs[0]);
  ReadSettings(prhs[2]);
  ResetRandomNumber();
  RunEM(prhs[1],
        corpus,
        (nlhs < 1) ? NULL : &plhs[0],
        (nlhs < 2) ? NULL : &plhs[1]);
  free_data(corpus);
}

// Main entry for [likelihoods, distribution] = infer(corpus, model, options).
MEX_FUNCTION(infer) (int nlhs,
                     mxArray* plhs[],
                     int nrhs,
                     const mxArray* prhs[]) {
  CheckInputArguments(3, 3, nrhs);
  CheckOutputArguments(0, 2, nlhs);
  lda_corpus* corpus = ReadCorpusFromMxArray(prhs[0]);
  ReadSettings(prhs[2]);
  ResetRandomNumber();
  RunInference(prhs[1],
               corpus,
               (nlhs < 1) ? NULL : &plhs[0],
               (nlhs < 2) ? NULL : &plhs[1]);
  free_data(corpus);
}

} // namespace