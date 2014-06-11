#include "Job.h" 
#include <boost/thread.hpp>   
#include <boost/thread/mutex.hpp> 
#include <boost/threadpool.hpp>

using namespace boost::threadpool;

int main(int argc, char** argv) 
{	
    // threadpool 
    pool tp(8); 

    // recupero alcuni parametri dall'esterno
    string file="../data/BipRW_r327_110704_var10_Best.mat"; 
           
    int DimFeatures=-1, RipFeatures=-1, Label=-1;  
    struct svm_parameter param_;
    bool Print=false,MultiThreading=false; // default 
    FunUtili::parseArguments(argc,argv,&file,&DimFeatures, &RipFeatures, &Label, &param_, &Print, &MultiThreading);     

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
         << "TsSz " << tssz << endl
         << "Print " << Print << endl
         << "MultiThreading " << MultiThreading << endl; 

    // lavoro
    Job j[rip];
    for (iRip=0; iRip<rip; ++iRip){
        if (MultiThreading==true)
            tp.schedule(boost::bind(&Job::load_and_run,j[iRip],iRip,pr_,DimFeatures,RipFeatures,Label,Print,param_));  // multithr
        else
            j[iRip].load_and_run(iRip,pr_,DimFeatures,RipFeatures,Label,Print,param_); 
    }
    if (MultiThreading==true)
        tp.wait(); 

    svm_destroy_param(&param_);
    return 0; 
} 
	


