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
    // PrintVar();  
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

