#include "Job.h" 


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
    svm_destroy_param(&param);
}

void SVModel::updateParam(struct svm_parameter *param_){
    // XXX non testato per modelli diversi dal mio
    if (param_->svm_type!=-1)
        param.svm_type = param_->svm_type;
    if (param_->kernel_type!=-1)
        param.kernel_type = param_->kernel_type;
    if (param_->degree!=-1)
        param.degree = param_->degree;
    if (param_->gamma!=-1)
        param.gamma = param_->gamma; // 1/num_features
    if (param_->coef0!=-1)
        param.coef0 = param_->coef0;
    if (param_->nu!=-1)
        param.nu = param_->nu;
    if (param_->cache_size!=-1)
        param.cache_size = param_->cache_size;
    if (param_->C!=-1)
        param.C = param_->C;
    if (param_->eps!=-1)
        param.eps = param_->eps;
    if (param_->p!=-1)
        param.p = param_->p;
    if (param_->shrinking!=-1)
        param.shrinking = param_->shrinking;
    if (param_->probability!=-1)
        param.probability = param_->probability;
    if (param_->nr_weight!=-1){
        param.nr_weight = param_->nr_weight;
        param.weight_label = new int [param.nr_weight];
        param.weight = new double [param.nr_weight];
        for (int i=0; i<param.nr_weight; ++i){
            param.weight_label[i] = param_->weight_label[i];
            param.weight[i] = param_->weight[i]; 
        }
    }
}

void SVModel::initStrutturaDati(int LabelTrSelSize,int *label_training_selection,  // info sul Training Set
                                int FeatSelSize,int *feature_sel,             // info sulle feature da usare
                                int LabelSelIdx,                                         // info sulla label da usare
                                double *f,size_t *FeatSize,double *l,size_t *LabelSize){ // info su tutto il dataset
	int i, ii, j, k, kk;
	int elements, max_index, sc, label_vector_row_num, featsize;
	double *samples, *labels;

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	labels = l;
	samples = f;
	sc = (int)FeatSize[1];

	elements = 0;
	// the number of instance
	prob.l = LabelTrSelSize;
    featsize = (int)FeatSize[0];
	label_vector_row_num = (int)LabelSize[0]; 

	if(label_vector_row_num!=featsize)
	{
		fprintf(stderr,"#Labels != #Features!!!");
		exit(1);
	}
    int TotIstanze=featsize; 


	if(param.kernel_type == PRECOMPUTED)
		elements = prob.l * (sc + 1);
	else
	{
		for(ii = 0; ii < prob.l; ii++)
		{
            i=label_training_selection[ii]; 
			for(kk = 0; kk < FeatSelSize /*sc*/; kk++){
                k = feature_sel[kk]; 
				if(samples[k * prob.l + i] != 0)
					elements++;
            }
			// count the '-1' element
			elements++;
		}
	}

    prob.y = new double [prob.l]; 
    prob.x = new struct svm_node* [prob.l]; 
    x_space = new struct svm_node [elements]; 

	max_index = sc;
	j = 0;
	for(ii = 0; ii < prob.l; ii++)
	{
        i=label_training_selection[ii]; 
		prob.x[ii] = &x_space[j];
		prob.y[ii] = labels[LabelSelIdx * TotIstanze + i];

		for(kk = 0; kk < FeatSelSize /*sc*/; kk++)
		{
            k = feature_sel[kk]; 
			if(param.kernel_type == PRECOMPUTED || samples[k * prob.l + i] != 0)
			{
				x_space[j].index = k + 1;
				x_space[j].value = samples[k * TotIstanze + i]; 
				j++;
			}
		}
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(ii=0; ii<prob.l; ii++)
		{
			if((int)prob.x[ii][0].value <= 0 || (int)prob.x[ii][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}
    const char* error_msg = svm_check_parameter(&prob, &param);
    if(error_msg)
    {
        if (error_msg != NULL)
            fprintf(stderr,"Errore: %s\n", error_msg);
        exit(1); 
    }
    //     
    assegnato=true; 
}


void SVModel::train(){    
    model=svm_train(&prob,&param); 
    addestrato=true; 
}

void SVModel::predict(matvar_t **RES,                                              // contenitore per i risultati
                      bool VideoPrint,                                             // stampo a video le performances
                      int LabelSelSize,int *labelSelection,                        // pattern da predirre
                      int FeatSelSize,int *featSelection,                          // info sulle feature da usare 
                      int LabelSelIdx,                                             // info sulla label da usare
                      double *f,size_t *FeatSize,double *l,size_t *LabelSize){     // info su tutto il dataset         
    FunUtili::predict_jr(RES,f,FeatSize,l,LabelSize,           // contenitori {ingressi,uscite}
                         labelSelection,LabelSelSize,          // pattern da predirre 
                         featSelection,FeatSelSize,            // info sulle feature da usare
                         LabelSelIdx,                          // info sulla label da usare
                         model,param.probability,              // info sul modello da adoperare 
                         VideoPrint); 
}

//////////////////// JOB ////////////////////////

Job::Job(int iRip_,Problema *pr_,int FeatSelSize_,int FeatSelRip_,int LabelSelIdx_,bool Print_, struct svm_parameter param_){
    assegnato_svm=false;
    assegnatiDatiTraining=false; 
    assegnatiDatiTest=false;    
    Print=Print_; 
    //
    *param=param_;
    iRip=iRip_;
    iFRip=FeatSelRip_; 
    LabelSelIdx=LabelSelIdx_; 
    ds.assegnoProblema(pr_); 
    UpdateDatiSimulazione(); // una volta all'inizio e basta 
    // FeatSelSize
    matvar_t *feat=(matvar_t*)ds.Features; 
    FeatSelSize=feat->dims[1]; 
    int *feature_sel_=new int [FeatSelSize];
    for (int i=0; i<FeatSelSize; ++i)
        feature_sel_[i]=i;    
    if (FeatSelSize_==-1){
        feature_sel=new int[FeatSelSize]; 
        for (int i=0; i<FeatSelSize; ++i)
            feature_sel[i]=feature_sel_[i];            // le seleziono tutte in ordine 
    }else{
        FeatSelSize=FeatSelSize_; 
        feature_sel=new int [FeatSelSize];
        FunUtili::randperm(FeatSelSize_,feature_sel_); // mescolo  
        for (int i=0; i<FeatSelSize; ++i)
            feature_sel[i]=feature_sel_[i];            // seleziono un certo numero dal gruppo mescolato
    }
    delete [] feature_sel_; 
    // FeatSelRip
    if (FeatSelRip_==-1)
        iFRip=1; 
    else
        iFRip=FeatSelRip_; 
    // LabelSelIdx
    if (LabelSelIdx_==-1)
        LabelSelIdx=0; // prima label di default    
    else
        LabelSelIdx=LabelSelIdx_; 
    //
    nome=ds.pr->nome;
    //
    RES = new matvar_t* [3];
}


Job::~Job(){
    delete [] RES;     
    delete [] feature_sel; 
}


// punto il problema
void Job::UpdateDatiSimulazione(){
    // TrSize=ds.LabelTrSelSize;
    // TsSize=ds.LabelTsSelSize;
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
        // aggiorno i parametri della svm 
        svm_.updateParam(param); 
        // leggo il problema per addestrare il modello svm 
        svm_.initStrutturaDati(ds.LabelTrSelSize,ds.label_training_selection,  // info sul Training Set
                               FeatSelSize,feature_sel,                        // info sulle feature da usare
                               LabelSelIdx,                                    // info sulla label da usare
                               features,FeatSize,labels,LabelSize);            // info su tutto il dataset
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
        svm_.predict(RES,                                           // contenitore per i risultati
                     Print,                                         // stampo a video le performances
                     ds.LabelTsSelSize,ds.label_test_selection,     // info sul Test Set 
                     FeatSelSize,feature_sel,                       // info sulle feature da usare 
                     LabelSelIdx,                                   // info sulla label da usare
                     features,FeatSize,labels,LabelSize);           // info su tutto il dataset
                         
    }else{
        fprintf(stderr,"Non ho addestrato un modello svm\n");
        exit(1);       
    }
    return ; 
}


void Job::run(){
    cout << endl << "-- new run -- iRip==" << iRip << endl;  
    int iTr=0,trsz=ds.pr->TrSzD, 
        iTs=0,tssz=ds.pr->TsSzD;  
    for (iTr=0; iTr<trsz; ++iTr){
        assegnato_svm=false; // devo addestrare per ogni Training Set un nuovo modello 
        ds.assegnoTrainingSet(iTr);  
        TrainingFromAssignedProblem(); // addestro
        for (iTs=0; iTs<tssz; ++iTs){
            ds.assegnoTestSet(iTs);
            predictTestSet(); 
        }
    }
    return; 
}	

