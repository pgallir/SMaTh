#ifndef _SVMMATIO_H
#define _SVMMATIO_H

#ifdef __cplusplus
extern "C" {
#endif

// inclusions
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"
#include <matio.h>   
//#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Realloc(var,type,n) (type *)realloc(var,(n)*sizeof(type))
#define ABS(var) (((var) < 0) ? -(var) : (var))

// variables
extern struct svm_node *x; 
extern int max_nr_attr; 
extern struct svm_model* model; 
extern int predict_probablity; 

// functions
void exit_with_help();
char* readline(FILE *input, char* line, int max_line_len); 
void parse_VIarguments(int argc, char **argv, char *IFN, struct svm_parameter *param, double *DimFeatures, double *RipFeatures, double *Label, int *perc_sig); 
void parse_command_line(int argc, char **argv, struct svm_parameter *param, double *DimFeatures, double *RipFeatures, double *Label, int *perc_sig); 
const char* Mat_VarErrCheck(const matvar_t *PARAM[]); //matvar_t *Features, matvar_t *Labels, matvar_t *RIP, matvar_t *TrSz, matvar_t *TsSz);
void svm_parameterPrint(struct svm_parameter param);
void svm_problemPrint(struct svm_problem prob);
void do_cross_validation(struct svm_problem prob, struct svm_parameter param, int nr_fold);
void read_problem_from_file(char *filename, struct svm_problem *prob, struct svm_parameter *param);
const char* read_problem_from_variable(const matvar_t *PARAM[], int iTr, int *&TestPotentialSelection, int *LabelTsCellSize, int perc_sig, struct svm_problem *prob, struct svm_parameter *param);
void exit_input_error(int line_num);
matvar_t * Mat_VarCreate_jr(const char *NomeVar, int raws, int cols); // solo per double!
void predict_jr(matvar_t *plhs[], const matvar_t *prhs[], int *label_test_selection, int LabelTsSelSize, struct svm_model *model, const int predict_probability);
matvar_t ** test_model_on_data_subset(const matvar_t *PARAM[], int iTs, int *TestPotentialSelection, int LabelTsCellSize, struct svm_model *model, int predict_probability, int perc_sig);
matvar_t ** validate_model(const matvar_t *PARAM[], struct svm_model *model, int predict_probability);
void randperm(int n,int perm[]); // from groups.csail.mit.edu

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */


