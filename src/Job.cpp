#include "Job.h" 
#include <time.h>   

//////////////////// SVModel ////////////////////////

SVModel::SVModel(){
    assegnato=false;     
    addestrato=false; 
    // valori di default 
    param.svm_type = EPSILON_SVR;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0; // 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
}

SVModel::~SVModel(){
    if (assegnato){
        delete [] prob.x; 
        delete [] prob.y; 
        delete [] x_space; 
    }
    if (addestrato){
        delete [] model; 
    }
}


SVModel::SVModel(int LabelTrSelSize,
                 size_t *FeatSize,size_t *LabelSize,
                 int *label_training_selection,
                 double *f,double *l){
    initStrutturaDati(LabelTrSelSize,FeatSize,LabelSize,label_training_selection,f,l); 
}

void SVModel::initStrutturaDati(int LabelTrSelSize,size_t *FeatSize,size_t *LabelSize,int *label_training_selection,double *f,double *l){
    assegnato=true; 
    int elements,i,ii,k; 
    size_t j,max_index; 
    // set dimensions problem
    prob.l = LabelTrSelSize;				      // #labels
    // JR: ATTENZIONE QUI!
    elements = prob.l*(1+FeatSize[1]);  //(FeatSize[0]+FeatSize[1]); // #labels (1 + #features)
    // cout << prob.l << " " << FeatSize[0] << " " << FeatSize[1] << endl; 


    max_index = FeatSize[1];    // #features  -- here only a full matrix had been considered
    // allocate enough memory
    prob.y = new double [prob.l];
    prob.x = new struct svm_node* [prob.l]; 
    x_space = new struct svm_node [elements]; 
    // riempio la struttura 
    k=0; // init # svm-s
    for (ii=0; ii<prob.l; ii++){ // run labels over label_training_selection indexes
        // get the proper pattern index
        i = label_training_selection[ii];
        // set prob fields
        prob.y[ii] = l[i];
        prob.x[ii] = &x_space[k]; 
        for (j=0; j<FeatSize[1]; j++){ // run features
            x_space[k].index = (long int)(j+1); 
            x_space[k].value = f[j*LabelSize[0]+i];
            ++k; 
        }
        x_space[k++].index = -1; // end of each little problem
    }   
    if(param.gamma == 0 && max_index > 0)
        param.gamma = 1.0/max_index;
    // if(param.kernel_type == PRECOMPUTED)
        // return "I do NOT consider a precumputed kernel_type by now";	
}


void SVModel::train(){    
    addestrato=true; 
    model=svm_train(&prob,&param); 
}



//////////////////// DATI SIMULAZIONE ////////////////////////

DatiSimulazione::DatiSimulazione(){
    assegnato=false; 
    LabelTrSelSize=0; 
    LabelTsSelSize=0; 
}

DatiSimulazione::DatiSimulazione(Problema *pr_,int iTr_,int iTs_){
    assegnato=true; 
    iTr=iTr_; 
    iTs=iTs_; 
    pr=pr_;
    AssegnoIlProblema();     
    // 
    const matvar_t **Cell = (const matvar_t**) idxCV->data;
    int i,ii,jj,k,k2,TOT_STEPS=idxCV->dims[1],
        NumTrBlockSel,NumTrTsBlockSel,block_selection[idxCV->dims[1]]; 
    double *trS_=(double*) TrSz->data, 
           *tsS_=(double*) TsSz->data; 
    NumTrBlockSel=(int) (trS_[iTr_]*TOT_STEPS/100);
    NumTrTsBlockSel=NumTrBlockSel+ (int) (tsS_[iTs_]*TOT_STEPS/100);
    if (NumTrTsBlockSel>TOT_STEPS)
        cout << endl << "QUESTO NON DEVE ACCADERE MAI!" << endl; 
    // permute randomly the training selection 
    randperm((int) idxCV->dims[1],block_selection); 
    // get the memory dimension of the label I need to allocate
    for (ii=0; ii<TOT_STEPS; ii++){
        if (ii<NumTrBlockSel)
            LabelTrSelSize += Cell[ii]->dims[1];
        else if (ii<NumTrTsBlockSel)
            LabelTsSelSize += Cell[ii]->dims[1]; 
    }
    // allocate memory abd assign block of {training,test} patterns
    // LabelTsCellSize = idxCV->dims[1] - nRPsel; // number of block remained from training phase
    label_training_selection = new int [LabelTrSelSize]; 
    label_test_selection = new int [LabelTsSelSize];
    //TestPotentialSelection = new int [LabelTsCellSize]; 
    k=0; k2=0;  
    for (ii=0; ii<(int)idxCV->dims[1]; ii++){
        i = block_selection[ii]; 
        double *cell = (double*)Cell[i]->data;
        if (ii<NumTrBlockSel)
            for (jj=0; jj<(int)Cell[i]->dims[1]; jj++)
                label_training_selection[k++] = (int)cell[jj];
        else if (ii<NumTrTsBlockSel)
            for (jj=0; jj<(int)Cell[i]->dims[1]; jj++)
                label_test_selection[k2++] = (int)cell[jj];
            //TestPotentialSelection[j++] = i;
    }    
}

DatiSimulazione::~DatiSimulazione(){
    if (assegnato){
        delete [] label_training_selection;     
        delete [] label_test_selection; 
        //delete [] TestPotentialSelection; 
    }
}

// punto il problema
void DatiSimulazione::AssegnoIlProblema(){
    Features=pr->Features; 
    Labels=pr->Labels; 
    VARIABLEs=pr->VARIABLEs; 
    RIP=pr->RIP; 
    idxCV=pr->idxCV; 
    idxVS=pr->idxVS; 
    TrSz=pr->TrSz; 
    TsSz=pr->TsSz; 
    return;     
}  

void DatiSimulazione::randperm(int n,int perm[]){  // from groups.csail.mit.edu
    srand(time(NULL)); // cambio seme
    int i, j, t;
    for(i=0; i<n; i++)
        perm[i] = i;
    for(i=0; i<n; i++){
        j = rand()%(n-i)+i;
        t = perm[j];
        perm[j] = perm[i];
        perm[i] = t;
    }  
    return ;
}

//////////////////// JOB ////////////////////////

Job::Job(int iRip_){
    assegnato_svm=false;
    iRip=iRip_;
    nome="non assegnato";
}


Job::~Job(){
}


// punto il problema
void Job::AssegnoDatiSimulazione(DatiSimulazione *ds_){
    ds=ds_;
    TrSize=ds->LabelTrSelSize;
    TsSize=ds->LabelTsSelSize;
    FeatSize=ds->Features->dims; 
    LabelSize=ds->Labels->dims; 
    label_training_selection=ds->label_training_selection;
    label_test_selection=ds->label_test_selection;
    features=(double*)ds->Features->data; 
    labels=(double*)ds->Labels->data;
    return;     
}  


void Job::read_problem_from_variable(){ 
    // leggo il problema per addestrare il modello svm 
    assegnato_svm=true; 
    svm_.initStrutturaDati(TrSize,FeatSize,LabelSize,label_training_selection,features,labels);
    return ; 
}

void Job::run(){
    read_problem_from_variable(); 
    svm_.train(); 
    return; 
}	

