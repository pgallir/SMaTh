#ifndef _PROBLEMA_H
#define _PROBLEMA_H

#include "FunUtili.h" 


// XXX debug da fare quando assegno il problema!
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
    int RIPD,TrSzD,TsSzD; 
    const matvar_t *Features, *Labels, *VARIABLEs, *RIP, *idxCV, *idxVS, *TrSz, *TsSz;   
    ~Problema(); 
    void PrintVar(); 
}; 


class DatiSimulazione{
public:
    Problema *pr; 
    int iTr,iTs,NumTrBlockSel,NumTrTsBlockSel,
        LabelTrSelSize,LabelTsSelSize,LabelValSelSize,
        *label_training_selection,*label_test_selection,*label_valid_selection, 
        *block_selection;
    const matvar_t *Features, *Labels, *VARIABLEs, *RIP, *idxCV, *idxVS, *TrSz, *TsSz;
    DatiSimulazione(); 
    ~DatiSimulazione(); 
    void assegnoTrainingSet(int iTr_);
    void assegnoTestSet(int iTs_);
    void assegnoValidationSet();
    void assegnoProblema(Problema *pr_); 
private:
    int TOT_STEPS; 
    double *trS_, *tsS_; 
    const matvar_t **CellCV, **CellVS; 
    bool problema_assegnato, 
         TrainingSet_assegnato,  
         TestSet_assegnato, 
         ValidationSet_assegnato;
}; 


# endif // _PROBLEMA_H

