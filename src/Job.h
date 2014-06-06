#ifndef _JOB_H
#define _JOB_H

#include "Problema.h"  

namespace FunUtili{
    void randperm(int n,int perm[]);  // from groups.csail.mit.edu
    void exit_with_help();
    void parseArguments(int argc,char **argv,string *filename,double *DimFeatures,double *RipFeatures,double *Label);
}


class SVModel{
public: 
    SVModel();
    ~SVModel(); 
    void initStrutturaDati(int LabelTrSelSize=0,
                           size_t *FeatSize=NULL,
                           size_t *LabelSize=NULL,
                           int *label_training_selection=NULL,
                           double *f=NULL,
                           double *l=NULL);
    void setParam(int argc, char** argv); 
    void train(); 
    void predict(matvar_t **RES, const matvar_t *feat, const matvar_t *label, int *label_test_selection);
private: 
    bool assegnato=false,addestrato=false; 
    struct svm_node *x_space; 
    struct svm_parameter param;
    struct svm_problem prob;
    struct svm_model *model;
}; 

class DatiSimulazione{
public:
    Problema *pr=NULL; 
    int LabelTrSelSize=0,LabelTsSelSize=0,iTr=0,iTs=0, 
        *label_training_selection=NULL,*label_test_selection=NULL, *block_selection=NULL,
        *TestPotentialSelection=NULL; 
    const matvar_t *Features, *Labels, *VARIABLEs, *RIP, *idxCV, *idxVS, *TrSz, *TsSz;
    DatiSimulazione(); 
    ~DatiSimulazione(); 
    void assegnoTrainingSet(int iTr_);
    void assegnoTestSet(int iTs_);
    void assegnoProblema(Problema *pr_); 
private:
    int TOT_STEPS=0; 
    double *trS_, *tsS_; 
    const matvar_t **CellCV; 
    bool problema_assegnato=false; 
    bool TrainingSet_assegnato=false;  
    bool TestSet_assegnato=false; 
}; 

class Job{    
public: 
    int iRip; 
    DatiSimulazione ds; 
    SVModel svm_; 
    matvar_t **RES; 
    Job(int iRip_,Problema *pr_); 
    ~Job(); 
    void run(); 
    void UpdateDatiSimulazione();  
    void TrainingFromAssignedProblem(); 
    void predictTestSet();
private:     
    string nome="non assegnato"; 
    bool assegnato_svm=false; 
    bool assegnatiDatiTraining=false, 
         assegnatiDatiTest=false; 
    int TsSize,TrSize,*label_training_selection,*label_test_selection; 
    size_t *FeatSize,*LabelSize; 
    double *features, 
           *labels,
           ***acc, 
           ***mse, 
           ***scc, 
           ***ValidTrend;  
}; 

# endif // _JOB_H

