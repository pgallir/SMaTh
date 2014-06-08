#include "Problema.h" 

//////////////////// PROBLEMA ////////////////////////

Problema::Problema(string f_pr){
    nome=f_pr; 
    // carico il .mat file
    mat_t *In_MATFILE = Mat_Open(nome.c_str(),MAT_ACC_RDONLY); 
    // genero il workspace
    Labels =  Mat_VarRead(In_MATFILE, "OUT"); // *** var in matlab WS
    Features = Mat_VarRead(In_MATFILE, "IN"); // *** var in matlab WS
    VARIABLEs = Mat_VarRead(In_MATFILE, "VARIABLEs"); // *** var in matlab WS
    RIP = Mat_VarRead(In_MATFILE, "RIP"); // *** var in matlab WS
    idxCV = Mat_VarRead(In_MATFILE, "idxCV"); // *** var in matlab WS	
    idxVS = Mat_VarRead(In_MATFILE, "idxVS"); // *** var in matlab WS	
    TrSz = Mat_VarRead(In_MATFILE, "TrSz"); // *** var in matlab WS
    TsSz = Mat_VarRead(In_MATFILE, "TsSz"); // *** var in matlab WS	 
    Mat_Close(In_MATFILE);
    // 
    double *ripd=(double*)RIP->data; 
    RIPD=*ripd; 
    TrSzD=TrSz->dims[1];
    TsSzD=TsSz->dims[1]; 
/*
    ValidationDimension=0; 
    const matvar_t **CellVS = (const matvar_t**) idxVS->data;
    for (int i=0; i<(int)idxVS->dims[1]; ++i)
        ValidationDimension+=(int) CellVS[i]->dims[1];        
*/
    // 
    if (0) // stampo a video per debug una matrice MxN
        Mat_VarPrint((matvar_t*) Features,1); 
}

Problema::~Problema(){ 
    // cancello il workspace
    Mat_VarFree((matvar_t *) Labels); 
    Mat_VarFree((matvar_t *) Features); 
    Mat_VarFree((matvar_t *) VARIABLEs); 
    Mat_VarFree((matvar_t *) RIP); 
    Mat_VarFree((matvar_t *) idxCV); 
    Mat_VarFree((matvar_t *) idxVS); 
    Mat_VarFree((matvar_t *) TrSz); 
    Mat_VarFree((matvar_t *) TsSz); 
}

void Problema::PrintVar(){
    Mat_VarPrint ((matvar_t *) Labels,1); 
    return; 
}

void Problema::SalvoRisultatiSuMatfile(){
    cout<<"non esiste questo metodo";     
    return;  
}



//////////////////// DATI SIMULAZIONE ////////////////////////

DatiSimulazione::DatiSimulazione(){
    problema_assegnato=false; 
    TrainingSet_assegnato=false;  
    TestSet_assegnato=false;  
    ValidationSet_assegnato=false; 
    LabelTrSelSize=0; 
    LabelTsSelSize=0; 
    NumTrBlockSel=0; 
    NumTrTsBlockSel=0;
}


DatiSimulazione::~DatiSimulazione(){
    /// XXX usando lo std::vector potrei evitarmelo
    if (TrainingSet_assegnato==true){
        delete [] label_training_selection;     
        delete [] block_selection; 
    }
    if (TestSet_assegnato==true)
        delete [] label_test_selection; 
    if (ValidationSet_assegnato==true)
        delete [] label_valid_selection;
}

void DatiSimulazione::assegnoProblema(Problema *pr_){
    pr=pr_;
    //
    Features=pr->Features; 
    Labels=pr->Labels; 
    VARIABLEs=pr->VARIABLEs; 
    RIP=pr->RIP; 
    idxCV=pr->idxCV; 
    idxVS=pr->idxVS; 
    TrSz=pr->TrSz; 
    TsSz=pr->TsSz; 
    //
    TOT_STEPS=idxCV->dims[1]; 
    trS_=(double*) TrSz->data;
    tsS_=(double*) TsSz->data; 
    CellCV = (const matvar_t**) idxCV->data;
    CellVS = (const matvar_t**) idxVS->data;
    //
    assegnoValidationSet(); // va chiamato una volta sola
    //
    problema_assegnato=true; 
    return;     
}


void DatiSimulazione::assegnoValidationSet(){
    // assegno il validation_set   
    if (ValidationSet_assegnato==false){ 
        LabelValSelSize=0;
        int i,j,k,VS_Cell_Dim=idxVS->dims[1]; 
        for (i=0; i<VS_Cell_Dim; ++i){
            LabelValSelSize += CellVS[i]->dims[1];        
        }
        label_valid_selection = new int [LabelValSelSize]; 
        k=0;  
        for (i=0; i<VS_Cell_Dim; ++i){
            double *cell = (double*)CellVS[i]->data;
            for (j=0; j<(int)CellVS[i]->dims[1]; ++j)
                label_valid_selection[k++] = (int)cell[j];
        }      
        ValidationSet_assegnato=true; 
    }else{
        fprintf(stderr," Problema non assegnato \n");
        exit(1);
    }
    return ; 
}


void DatiSimulazione::assegnoTrainingSet(int iTr_){
    // cancello la memoria se sto riassegnando il TrainingSet
    if (TrainingSet_assegnato==true){ 
        delete [] block_selection; 
        delete [] label_training_selection; 
        LabelTrSelSize=0; 
        TrainingSet_assegnato=false; 
    }
    // assegno il training_set   
    if (problema_assegnato){
        iTr=iTr_; // e se eccedo? deve controllarlo Job
        int i,ii,jj,k; 
        NumTrBlockSel=(int) (trS_[iTr_]*TOT_STEPS/100);
        // permute randomly the training selection 
        block_selection = new int [idxCV->dims[1]]; 
        FunUtili::randperm((int) idxCV->dims[1],block_selection); 
        // get the memory dimension of the label I need to allocate
        for (ii=0; ii<TOT_STEPS; ii++){
            i = block_selection[ii];  // recupero la cella opportuna dopo che ho fatto il mescolone
            if (ii<NumTrBlockSel)
                LabelTrSelSize += CellCV[i]->dims[1];        
        }
        label_training_selection = new int [LabelTrSelSize]; 
        k=0;  
        for (ii=0; ii<TOT_STEPS; ii++){
            i = block_selection[ii]; // recupero la cella opportuna dopo che ho fatto il mescolone
            double *cell = (double*)CellCV[i]->data;
            if (ii<NumTrBlockSel)
                for (jj=0; jj<(int)CellCV[i]->dims[1]; jj++)
                    label_training_selection[k++] = (int)cell[jj];
        }      
        TrainingSet_assegnato=true;  
    }else{
        fprintf(stderr," Problema non assegnato \n");
        exit(1);
    }
    return ; 
}



void DatiSimulazione::assegnoTestSet(int iTs_){
    // cancello la memoria se sto riassegnando il TestSet
    if (TestSet_assegnato==true){ 
        delete [] label_test_selection; 
        LabelTsSelSize=0; 
        TestSet_assegnato=false; 
    }
    if (problema_assegnato &&
        TrainingSet_assegnato){
        iTs=iTs_; // e se eccedo? deve controllarlo Job
        int i,ii,jj,k; 
        NumTrTsBlockSel=NumTrBlockSel+
                        (int) (tsS_[iTs_]*TOT_STEPS/100);
        if (NumTrTsBlockSel>TOT_STEPS){
            fprintf(stderr," Training Set + Test Set > 100 %% \n");
            exit(1);
        }
        for (ii=NumTrBlockSel; ii<NumTrTsBlockSel; ii++){
            i = block_selection[ii];  // recupero la cella opportuna dopo che ho fatto il mescolone
            LabelTsSelSize += CellCV[i]->dims[1]; 
        }
        label_test_selection = new int [LabelTsSelSize];
        k=0;  
        for (ii=NumTrBlockSel; ii<NumTrTsBlockSel; ii++){
            i = block_selection[ii];  // recupero la cella opportuna dopo che ho fatto il mescolone
            double *cell = (double*)CellCV[i]->data;
            for (jj=0; jj<(int)CellCV[i]->dims[1]; jj++)
                label_test_selection[k++] = (int)cell[jj];
        }      
        TestSet_assegnato=true;  
    }else{
        fprintf(stderr," Problema e(o) TrainingSet non assegnati(o) \n");
        exit(1);
    }
    return ; 
}



