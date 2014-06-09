#ifndef _JOB_H
#define _JOB_H

#include "Problema.h"  

class SVModel{
public: 
    SVModel();
    ~SVModel(); 
    void initStrutturaDati(int LabelTrSelSize,int *label_training_selection,        // info sul Training Set
                           int FeatSelSize,int *feature_sel,                        // info sulle feature da usare
                           int LabelSelIdx,                                         // info sulla label da usare
                           double *f,size_t *FeatSize,double *l,size_t *LabelSize); // info su tutto il dataset 
    void updateParam(struct svm_parameter *param_); 
    void train(); 
    void predict(matvar_t **RES,                                              // contenitore per i risultati
                 bool VideoPrint,                                             // stampo a video le performances
                 int LabelSelSize,int *labelSelection,                        // pattern da predirre 
                 int FeatSelSize,int *featSelection,                          // info sulle feature da usare 
                 int LabelSelIdx,                                             // info sulla label da usare
                 double *f,size_t *FeatSize,double *l,size_t *LabelSize);     // info su tutto il dataset 
private: 
    bool assegnato,addestrato; 
    struct svm_node *x_space; 
    struct svm_parameter param;
    struct svm_problem prob;
    struct svm_model *model;
}; 


class Job{    
public: 
    DatiSimulazione ds; 
    SVModel svm_; 
    matvar_t **RES,**TRENDs,*VARIABLEs; 
    Job(); 
/*(int iRip_,Problema *pr_,
int FeatSelSize_,int FeatSelRip_,int LabelSelIdx_,
bool Print_, struct svm_parameter param_); */
    ~Job(); 
    void load_and_run(int iRip_,Problema *pr_,int FeatSelSize_,int FeatSelRip_,int LabelSelIdx_,bool Print_, struct svm_parameter param_); 
    void run(); 
    void UpdateDatiSimulazione();  
    void UpdateFeatureSelection();
    void TrainingFromAssignedProblem(int labelIdx); 
    void predictTestSet(int labelIdx);
    void predictValidationSet(double *ValidTrend,int labelIdx); 
private:     
    string nome,qualeLabel,resFile; 
    struct svm_parameter *param;
    bool assegnatoProblema,
         assegnato_svm,
         assegnatiDatiTraining, 
         assegnatiDatiTest,
         Print,
         featRandomSelection; 
    int iRip,FRip,
        TsSize,TrSize,
        TotFeatSize,FeatSelSize,
        *LabelSelIdx,LabelSelSize,
        *label_training_selection,*label_test_selection,*feature_sel;
    size_t *FeatSize,*LabelSize; 
    double *features, 
           *labels,
           ***acc, 
           ***mse, 
           ***scc, 
           ***ValidTrend;  
}; 

# endif // _JOB_H

