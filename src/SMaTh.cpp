#include "Job.h" 

int main(int argc, char** argv) 
{	
    // recupero alcuni parametri dall'esterno
    string file="../data/BipRW_r327_110704_var10_Best.mat";//"../data/r330_BipCRW_07_19_11_DownsampingFactor3_Valid_CTRL2.mat"; 
           
    int DimFeatures=-1, RipFeatures=-1, Label=-1;  
    struct svm_parameter param_;
    bool Print; 
    FunUtili::parseArguments(argc,argv,&file,&DimFeatures, &RipFeatures, &Label, &Print, &param_);     

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
        Job j(iRip,pr_,DimFeatures,RipFeatures,Label,Print,param_); 
//        Job j(iRip,pr_,argc,argv); 
        j.run(); 
    }

    svm_destroy_param(&param_);
    return 0; 
} 
	


