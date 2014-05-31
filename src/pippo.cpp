#include "Job.h" 

int main() 
{	
    // trovo il file
    string file="../data/r330_BipCRW_07_19_11_DownsampingFactor3_Valid_CTRL2.mat"; 
    // inizializzo il problema
    Problema pr(file), *pr_=&pr; 
    // inizializzo il set di job
    Job j_[10]; 
    for (iRip=0; iRip<10; iRip++){
        j_[iRip].iRip=iRip; 
        j_[iRip].AssegnoDatiSimulazione(pr_); // genera errore ma ci piace cosi`
    }
    


/*    
    int iTr=0,iTs=0;  
    DatiSimulazione ds(pr_,iTr,iTs), 
                    *ds_=&ds; 

    int iRip=0; 
    Job j_[10]; 
    for (iRip=0; iRip<10; iRip++){
        j_[iRip].iRip=iRip; 
        j_[iRip].AssegnoDatiSimulazione(ds_); 
    }
    // faccio roba 
    j_[0].run(); 
    //cout << "nome " << j.nome << " rip " << j.iRip << endl; 
*/

    return 0; 
} 
	


