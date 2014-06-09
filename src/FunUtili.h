#ifndef _FUN_UTIL_
#define _FUN_UTIL_

#include "svm.h"    // per fare la classe Job
#include <iostream> 
#include <matio.h>  // per fare la classe Problema
#include <string.h>
#include <math.h>
#include <time.h>   
#include <algorithm>

using namespace std; 

namespace FunUtili{
    void randperm(int n,int *perm);  
    void exit_with_help();
    //void parseArguments(int argc,char **argv,string *filename,int *DimFeatures,int *RipFeatures,int *Label,struct svm_parameter *param);
/*    void parseArguments(int argc, char **argv, 
                        string *filename, int *DimFeatures, int *RipFeatures, int *Label,bool *Print,
                        struct svm_parameter *param); */
    void parseArguments(int argc, char **argv, 
                        string *filename, int *DimFeatures, int *RipFeatures, int *Label,
                        struct svm_parameter *param, 
                        bool *Print, bool *MultiThreading);
    matvar_t* Mat_VarCreate_jr(const char *NomeVar, int raws, int cols); 
    void predict_jr(matvar_t *plhs[],double *f,size_t *FeatSize,double *l,size_t *LabelSize,      // contenitori {ingressi,uscite}
                    int *testing_idx,int testing_instance_number,                                  // pattern da predirre 
                    int *feat_idx,int feature_number,                                              // info sulle feature da usare
                    int LabelSelIdx,                                                               // info sulla label da usare
                    struct svm_model *model,const int predict_probability,                         // info sul modello da adoperare 
                    bool print); 
}


#endif //_FUN_UTIL_
