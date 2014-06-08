#ifndef _PROBLEMA_H
#define _PROBLEMA_H

#include "FunUtili.h" 


// debug da fare quando assegno il problema!
/*
  int label_num, label_instance, feature_number, testing_instance_number; 
  feature_number = (int)(prhs[1]->dims[1]);
  testing_instance_number = (int)(prhs[1]->dims[0]);
  label_num = (int)(prhs[0]->dims[1]); 
  label_instance = (int)(prhs[0]->dims[0]);
 
    cout << "f# " << feature_number << " l# " << label_num << " inst " << label_instance << endl; 
    
  
  if(label_instance!=testing_instance_number)
  {
    printf("Length of label vector does not match # of instances.\n");
    return;
  }
  if(label_num!=1)
  {
    printf("label (1st argument) should be a vector (# of column is 1).\n");
    return;
  }
*/




class Problema{
public: 
    string nome; 
    Problema(string f_pr); 
    int RIPD=0,TrSzD=0,TsSzD=0,ValidationDimension=0; 
    const matvar_t *Features, *Labels, *VARIABLEs, *RIP, *idxCV, *idxVS, *TrSz, *TsSz;   
    ~Problema(); 
    void PrintVar(); 
protected: 
    void SalvoRisultatiSuMatfile();    // qui salvo i risultati come si aspetta chi legge i .mat
}; 


class DatiSimulazione{
public:
    Problema *pr=NULL; 
    int LabelTrSelSize=0,LabelTsSelSize=0,iTr=0,iTs=0,NumTrBlockSel=0,NumTrTsBlockSel=0,
        *label_training_selection=NULL,*label_test_selection=NULL,*label_valid_selection=NULL, 
        *block_selection=NULL;
    const matvar_t *Features, *Labels, *VARIABLEs, *RIP, *idxCV, *idxVS, *TrSz, *TsSz;
    DatiSimulazione(); 
    ~DatiSimulazione(); 
    void assegnoTrainingSet(int iTr_);
    void assegnoTestSet(int iTs_);
    void assegnoValidationSet();
    void assegnoProblema(Problema *pr_); 
private:
    int TOT_STEPS=0; 
    double *trS_, *tsS_; 
    const matvar_t **CellCV, **CellVS; 
    bool problema_assegnato=false, 
         TrainingSet_assegnato=false,  
         TestSet_assegnato=false, 
         ValidationSet_assegnato=false;
}; 


# endif // _PROBLEMA_H

