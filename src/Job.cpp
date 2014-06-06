#include "Job.h" 
#include <time.h>   



//////////////////// FunUtili ////////////////////////

void FunUtili::randperm(int n,int perm[]){  // from groups.csail.mit.edu
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

void FunUtili::exit_with_help(){
    printf(
    "\E[1m"
    "Usage: ./SMath [matfile] [options] \n\n"
    "1) matfile\n\n"
    "\E[0m" 
    "the matfile MUST contain the following variables:\n"
    "IN\t\t<#occurrences,#features>\t\t\tfeatures matrix\n"
    "OUT\t\t<#occurrences,#labels>\t\t\t\tlabels matrix\n"			
    "RIP\t\t#repetitions for Cross Validation\t\t1D of the simulation cube\n"
    "TrSz\t\t<1,#training sizes>\t\t\t\t2D of the simulation cube\n"
    "TsSz\t\t<1,#test sizes>\t\t\t\t\t3D of the simulation cube\n"
    "VARIABLEs\t<#labels,1>\t\t\t\t\tlabels' name\n"
    "idxCV\t\t<1,#foldCV>\t\t\t\t\tfold division for Cross Validation\n\n"
    "idxVS\t\t<1,#foldVL>\t\t\t\t\tfold division for preselected Validation Set\n\n"
    "\E[1m"
    "2) options:\n\n"
    "\E[0m"
    "-s svm_type : set type of SVM (default 0)\n"
    " 0 -- C-SVC\n"
    " 1 -- nu-SVC\n"
    " 2 -- one-class SVM\n"
    " 3 -- epsilon-SVR\n"
    " 4 -- nu-SVR\n"
    "-t kernel_type : set type of kernel function (default 2)\n"
    " 0 -- linear: u'*v\n"
    " 1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
    " 2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
    " 3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
    " 4 -- precomputed kernel (kernel values in training_set_file)\n"
    "-d degree : set degree in kernel function (default 3)\n"
    "-g gamma : set gamma in kernel function (default 1/num_features)\n"
    "-r coef0 : set coef0 in kernel function (default 0)\n"
    "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
    "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
    "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
    "-m cachesize : set cache memory size in MB (default 100)\n"
    "-e epsilon : set tolerance of termination criterion (default 0.001)\n"
    "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
    "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
    "-q : quiet mode (no outputs)\n\n"
    "\E[1m"
    "3) advanced options\n\n"
    "\E[0m"
    "--#features : if you don't want to use all features, you can here set how many features to be used (chosen randomly)\n"
    "              NB: you need to specify how many time you want to change them, becuse you use the same features RIP time among folds\n"
    "              ex: '--#features 5:10' will select 10 times 5 features and simulate the problem defined in the matfile\n"
    "--label[i] : if you don't want to run the simulations across all variables, you can here set what label should be used\n"
    "             ex: '--label[i] 2' will select the 3rd column of OUT (see 1) matfile)\n"
    "--perc_sig : perc_sig==0 means that the signal folds are measured in steps\n" 
    "             perc_sig==1 means that are measured in percentage of the whole recorded signal\n"
    "             it cannot assume different values from {0,1}\n"
    "             ex: '--perc_sig 1' will use T{r,s}Sz values as percentage of idxCV\n"
    "\n");
    exit(1);
}



void FunUtili::parseArguments(int argc, char **argv, string *filename, double *DimFeatures, double *RipFeatures, double *Label){
    // abbastanza ingressi? 
    if(argc==1)
        cout << "------ DEFAULT:  " << endl; 
    // parse options
    for(int i=1;i<argc;i++){
        if(0) // DEBUG
            cout << "arg[" << i << "]" << argv[i][0] << endl; 
        if((argv[i][0]=='-') && (argv[i][1]=='-')){ 
            if(string(argv[i]) == "--help"){
                FunUtili::exit_with_help(); 
            }else if(string(argv[i]) == "--#features"){
                if (i+1>=argc || argv[i+1][0] == '-'){
                    fprintf(stderr,"You forgot to set value @ %s\n",argv[i]); 
                    exit(2);
                }
                char *tokens[2], *token_ptr;
                token_ptr = strtok((char*)argv[i+1],",:"); 
                for(int itok=0; itok<2 && token_ptr != NULL; itok++){
                    tokens[itok] = token_ptr;
                    token_ptr = strtok(NULL, " ");
                    }
                *DimFeatures = (double)atoi(tokens[0]); 
                *RipFeatures = (double)atoi(tokens[1]); 
            }else if(string(argv[i]) == "--matfile"){
                if (i+1>=argc || argv[i+1][0] == '-'){
                    fprintf(stderr,"You forgot to set value @ %s\n",argv[i]); 
                    exit(2);
                }
                (*filename).assign(argv[i+1]);
            }else if(string(argv[i]) == "--label[i]"){
                if (i+1>=argc || argv[i+1][0] == '-'){
                    fprintf(stderr,"You forgot to set value @ %s\n",argv[i]); 
                    exit(2);
                }
                *Label = (double)atoi(argv[i+1]);
            }else{
                fprintf(stderr,"%s is not correct. Use --help for further infos\n",argv[i]);
                exit(1);
            }
        }
    }
}


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


void SVModel::setParam(int argc, char** argv){
/*
    // parse options
    for(i=1;i<argc;i++){
        if(argv[i][0] != '-') break;
        if(++i>=argc  &&
        argv[i][0] != '-') // JR: we can now have --something
        exit_with_help();
        switch(argv[i-1][1])
        {
        case 's':
        param->svm_type = atoi(argv[i]);
        break;
        case 't':
        param->kernel_type = atoi(argv[i]);
        break;
        case 'd':
        param->degree = atoi(argv[i]);
        break;
        case 'g':
        param->gamma = atof(argv[i]);
        break;
        case 'r':
        param->coef0 = atof(argv[i]);
        break;
        case 'n':
        param->nu = atof(argv[i]);
        break;
        case 'm':
        param->cache_size = atof(argv[i]);
        break;
        case 'c':
        param->C = atof(argv[i]);
        break;
        case 'e':
        param->eps = atof(argv[i]);
        break;
        case 'p':
        param->p = atof(argv[i]);
        break;
        case 'h':
        param->shrinking = atoi(argv[i]);
        break;
        case 'b':
        param->probability = atoi(argv[i]);
        break;
        case 'q':
        print_func = &print_null;
        i--;
        break;
        case 'w':
        ++param->nr_weight;
        param->weight_label = (int *)realloc(param->weight_label,sizeof(int)*param->nr_weight);
        param->weight = (double *)realloc(param->weight,sizeof(double)*param->nr_weight);
        param->weight_label[param->nr_weight-1] = atoi(&argv[i-1][2]);
        param->weight[param->nr_weight-1] = atof(argv[i]);
        default:
        fprintf(stderr,"%s is not correct. Use --help for further infos\n",argv[i-1]);
        exit(1); 
       }
    }
*/
}

void SVModel::initStrutturaDati(int LabelTrSelSize,size_t *FeatSize,size_t *LabelSize,int *label_training_selection,double *f,double *l){
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
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // if(param.kernel_type == PRECOMPUTED)
        // return "I do NOT consider a precumputed kernel_type by now";	
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // 
    assegnato=true; 
}


void SVModel::train(){    
    model=svm_train(&prob,&param); 
    addestrato=true; 
}

void SVModel::predict(matvar_t **RES, const matvar_t *Features, const matvar_t *Labels, int *label_test_selection){
    const matvar_t *INPUTs[2]; 
    INPUTs[0] = Labels; 
    INPUTs[1] = Features; 
    //FunUtili::predict_jr(RES,INPUTs,label_test_selection,model,0); 

    cout<<"fingiamo "<< (int) sizeof(label_test_selection)/sizeof(int)<<endl;



//    predict_jr((matvar_t **) RES, (const matvar_t **)INPUTs, label_test_selection, LabelVlSelSize, model, int(0)); //JR: predict_probability sempre == 0

/*
matvar_t** validate_model(const matvar_t *PARAM[], struct svm_model *model, int predict_probability)
{
  //PARs:={Features, Labels, VARIABLEs, RIP, idxCV, idxVS, TrSz, TsSz}
  const matvar_t *Features = PARAM[0];	// features matrix
  const matvar_t *Labels = PARAM[1]; 	// label vector (matrix??) for the different variables
  //const matvar_t *VARIABLEs = PARAM[2];	// cells containing variables names
  //const matvar_t *RIP = PARAM[3]; 	// #repetition of CV
  //const matvar_t *BlockCV = PARAM[4]; 	// cell containing the CROSS VALIDATION indexes subdivision
  //const matvar_t **Cell = (const matvar_t**) BlockCV->data;
  const matvar_t *BlockVS = PARAM[5]; 	// cell containing the VALIDATION SET indexes subdivision
  const matvar_t **CellV = (const matvar_t**) BlockVS->data;
  //const matvar_t *trS = PARAM[6]; 	// # of BlockCV cells to be used for training the svm model
  //const matvar_t *tsS = PARAM[7]; 	// # of BlockCV cells to be used for test the svm model

  int i, j, k, *label_test_selection, LabelVlSelSize=0;

  // get the memory dimension of the label I need to allocate for label_test_selection
  k=0; 
  for (i=0; i<(int)BlockVS->dims[1]; i++)
  {     
    LabelVlSelSize += CellV[i]->dims[1];
  }
  label_test_selection = Malloc(int, LabelVlSelSize);
  for (i=0; i<(int)BlockVS->dims[1]; i++)
  {
    double *cell = (double*)CellV[i]->data;
    for (j=0; j<(int)CellV[i]->dims[1]; j++)
      label_test_selection[k++] = (int)cell[j];
  }

  // do the prediction
  matvar_t **RES = Malloc(matvar_t*,3);
  const matvar_t *INPUTs[2]; 
  INPUTs[0] = Labels; 
  INPUTs[1] = Features; 
  predict_jr((matvar_t **) RES, (const matvar_t **)INPUTs, label_test_selection, LabelVlSelSize, model, int(0)); //JR: predict_probability sempre == 0

  //remember to free your allocated memory
  free(label_test_selection);

  return RES;   
}


*/

}

//////////////////// DATI SIMULAZIONE ////////////////////////

DatiSimulazione::DatiSimulazione(){
    problema_assegnato=false; 
    TrainingSet_assegnato=false;  
    TestSet_assegnato=false;  
    LabelTrSelSize=0; 
    LabelTsSelSize=0; 
}


DatiSimulazione::~DatiSimulazione(){
    /// XXX usando lo std::vector potrei evitarmelo
    if (TrainingSet_assegnato==true){
        delete [] label_training_selection;     
        delete [] block_selection; 
    }
    if (TestSet_assegnato==true)
        delete [] label_test_selection; 
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
    //
    problema_assegnato=true; 
    return;     
}

void DatiSimulazione::assegnoTrainingSet(int iTr_){
    // cancello la memoria se sto riassegnando il TrainingSet
    /*
    if (TrainingSet_assegnato==true){ 
        delete [] block_selection; 
        delete [] label_training_selection; 
        cout << "FATTO"<<endl;
    }
    */
    // assegno il training_set   
    if (problema_assegnato){
        iTr=iTr_; // e se eccedo? deve controllarlo Job
        int i,ii,jj,k,NumTrBlockSel; 
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
    if (TrainingSet_assegnato==true){ 
        delete [] label_test_selection; 
    }
    if (problema_assegnato &&
        TrainingSet_assegnato){
        iTs=iTs_; // e se eccedo? deve controllarlo Job
        int i,ii,jj,k,
            NumTrBlockSel,NumTrTsBlockSel; 
        NumTrBlockSel=(int) (trS_[iTr]*TOT_STEPS/100);
        NumTrTsBlockSel=NumTrBlockSel+ (int) (tsS_[iTs_]*TOT_STEPS/100);
        if (NumTrTsBlockSel>TOT_STEPS){
            fprintf(stderr," Training Set + Test Set > 100 %% \n");
            exit(1);
        }
        for (ii=0; ii<TOT_STEPS; ii++){
            i = block_selection[ii];  // recupero la cella opportuna dopo che ho fatto il mescolone
            if (ii>=NumTrBlockSel &&
                ii<NumTrTsBlockSel)
                LabelTsSelSize += CellCV[i]->dims[1]; 
        }
        label_test_selection = new int [LabelTsSelSize];
        k=0;  
        for (ii=0; ii<TOT_STEPS; ii++){
            i = block_selection[ii];  // recupero la cella opportuna dopo che ho fatto il mescolone
            double *cell = (double*)CellCV[i]->data;
            if (ii>=NumTrBlockSel && 
                ii<NumTrTsBlockSel)
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




/* NON DOVREBBE SERVIRE PIU'
DatiSimulazione::DatiSimulazione(Problema *pr_,int iTr_,int iTs_){
    iTr=iTr_; 
    iTs=iTs_; 
    pr=pr_;
    AssegnoIlProblema();     
    assegnato=true; 
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
    FunUtili::randperm((int) idxCV->dims[1],block_selection); 
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
*/




//////////////////// JOB ////////////////////////

Job::Job(int iRip_,Problema *pr_){
    iRip=iRip_;
    ds.assegnoProblema(pr_); 
    nome=ds.pr->nome;
    assegnato_svm=false;
    assegnatiDatiTraining=false; 
    assegnatiDatiTest=false; 
    RES = new matvar_t* [3];
}


Job::~Job(){
    delete [] RES;     
}


// punto il problema
void Job::UpdateDatiSimulazione(){
    TrSize=ds.LabelTrSelSize;
    TsSize=ds.LabelTsSelSize;
    FeatSize=ds.Features->dims; 
    LabelSize=ds.Labels->dims; 
    features=(double*)ds.Features->data; 
    labels=(double*)ds.Labels->data;
    //
    assegnatiDatiTraining=true; 
    assegnatiDatiTest=true; 
    return;     
}  


void Job::TrainingFromAssignedProblem(){ 
    if (assegnatiDatiTraining){
        // leggo il problema per addestrare il modello svm 
        svm_.initStrutturaDati(TrSize,FeatSize,LabelSize,ds.label_training_selection,features,labels);
        svm_.train();
        assegnato_svm=true; 
    }else{
        fprintf(stderr,"Non so quali dati usare per fare il Training del modello \n");
        exit(1);       
    }
    return ; 
}


void Job::predictTestSet(){ 
    if (assegnato_svm){
        // leggo il problema per addestrare il modello svm 
        svm_.predict(RES,ds.Features,ds.Labels,ds.label_test_selection);
    }else{
        fprintf(stderr,"Non ho addestrato un modello svm\n");
        exit(1);       
    }
    return ; 
}


void Job::run(){
    cout << "-- new run -- iRip==" << iRip << endl;  
    UpdateDatiSimulazione(); // una volta all'inizio e basta 
    int iTr=0,trsz=ds.pr->TrSzD, 
        iTs=0,tssz=ds.pr->TsSzD;  
    for (iTr=0; iTr<1/*trsz*/; ++iTr){
        assegnato_svm=false; // devo addestrare per ogni Training Set un nuovo modello 
        ds.assegnoTrainingSet(iTr);  
        /*
        // Funziona, sembra almeno... 
        cout << " " << ds.LabelTrSelSize << endl; 
        for (int i=0; i<ds.LabelTrSelSize; ++i)
            cout << " " << ds.label_training_selection[i] << ":" << features[ds.label_training_selection[i]];
        */    
        // addestro
        TrainingFromAssignedProblem(); 
/*        
        for (iTs=0; iTs<tssz; ++iTs){
            // assegno il Test Set
            ds.assegnoTestSet(iTs);
            cout << "testo qui " << iTs<< endl;  
            predictTestSet(); 
        }
*/
    }
    return; 
}	

