#include "Job.h" 

int main(int argc, char** argv) 
{	
    // recupero alcuni parametri dall'esterno
    string file="../data/r330_BipCRW_07_19_11_DownsampingFactor3_Valid_CTRL2.mat"; // default
    double DimFeatures=NAN, RipFeatures=NAN, Label=NAN;  
    FunUtili::parseArguments(argc,argv,&file,&DimFeatures, &RipFeatures, &Label);     

    // inizializzo il problema
    Problema pr(file), *pr_=&pr; 
    int rip=pr_->RIPD, 
        trsz=pr_->TrSzD,
        tssz=pr_->TsSzD, 
        iRip=0;    // definiscono il numero di volte che devo addestrare un nuovo modello 

    // avviso il povero utente
    cout << "Filename " << file << endl 
         << "DimFeatures " << DimFeatures << endl 
         << "RipFeatures "  << RipFeatures << endl 
         << "Label " << Label << endl 
         << "Rip " << rip << endl 
         << "TrSz " << trsz << endl 
         << "TsSz " << tssz << endl; 

    // lavoro
    for (iRip=0; iRip<rip; ++iRip){
        Job j(iRip,pr_); 
        j.run(); 
    }

/*
    Job j_[10]; 
    for (int iRip=0; iRip<10; iRip++){
        j_[iRip].iRip=iRip; 
        j_[iRip].AssegnoDatiSimulazione(ds_); 
        // parse dei parametri
    }
    // faccio roba 
    // j_[0].run(); 
    //cout << "nome " << j_[0].nome << " rip " << j_[0].iRip << endl; 
*/

    return 0; 
} 
	


