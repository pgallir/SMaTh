#ifndef _JOB_H
#define _JOB_H

#include "Problema.h"  

class SVModel{
public: 
    SVModel();
    SVModel(int LabelTrSelSize,size_t *FeatSize,size_t *LabelSize,int *label_training_selection,double *f,double *l); 
    ~SVModel(); 
    void initStrutturaDati(int LabelTrSelSize=0,
                           size_t *FeatSize=NULL,
                           size_t *LabelSize=NULL,
                           int *label_training_selection=NULL,
                           double *f=NULL,
                           double *l=NULL);
    void train(); 
private: 
    bool assegnato=false,addestrato=false; 
    struct svm_node *x_space; 
    struct svm_parameter param;
    struct svm_problem prob;
    struct svm_model *model;
}; 

class DatiSimulazione{
public:
    int LabelTrSelSize=0,LabelTsSelSize=0,iTr=0,iTs=0,//LabelTsCellSize, 
        *label_training_selection=NULL,*label_test_selection=NULL, 
        *TestPotentialSelection=NULL; 
    const matvar_t *Features, *Labels, *VARIABLEs, *RIP, *idxCV, *idxVS, *TrSz, *TsSz;
    DatiSimulazione(); 
    DatiSimulazione(Problema *pr_,int iTr_,int iTs_); 
    ~DatiSimulazione(); 
private:
    Problema *pr=NULL; 
    void randperm(int n,int perm[]);  // from groups.csail.mit.edu
    void AssegnoIlProblema();  
    bool assegnato=false; 
}; 

class Job{    
public: 
    int iRip; // numero di ripetizioni
    Job(int iRip_=-1); 
    ~Job(); 
    void run(); 
    void AssegnoDatiSimulazione(DatiSimulazione *ds_);  
    void read_problem_from_variable(); 
private:     
    SVModel svm_; 
    DatiSimulazione *ds=NULL; 
    string nome="non assegnato"; 
    bool assegnato_svm=false; 
    int TsSize,TrSize,
        *label_training_selection,*label_test_selection; 
    size_t *FeatSize,*LabelSize; 
    double *features, 
           *labels,
           ***acc, 
           ***mse, 
           ***scc, 
           ***ValidTrend;  
}; 

# endif // _JOB_H

