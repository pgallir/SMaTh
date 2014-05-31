#ifndef _PROBLEMA_H
#define _PROBLEMA_H

#include "svm.h"    // per fare la classe Job
#include <iostream> 
#include <matio.h>  // per fare la classe Problema
#include <string>
using namespace std; 

class Problema{
public: 
    string nome; 
    Problema(string f_pr); 
    const matvar_t *Features, *Labels, *VARIABLEs, *RIP, *idxCV, *idxVS, *TrSz, *TsSz;   
    ~Problema(); 
    void PrintVar(); 
protected: 
    void SalvoRisultatiSuMatfile();    // qui salvo i risultati come si aspetta chi legge i .mat
}; 


# endif // _PROBLEMA_H

