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

    //debug
    if (0){
        cout << "param.svm_type " <<  param.svm_type << endl; 
        cout << "param.kernel_type " <<  param.kernel_type<< endl; 
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


// ROBA PER DEBUG 
if (0){
	for(ii = 0; ii < 10; ii++){
        i=label_training_selection[ii]; 
        cout << " " << labels[LabelSelIdx * TotIstanze + i]; 
    }
    cout << endl<< endl;

	for(ii = 0; ii < 10; ii++){
        i=label_training_selection[ii]; 
        cout << " " << i; 
    }
    cout << endl<< endl;
    for(ii = 0; ii < 10; ii++){
        i=label_training_selection[ii]; 
		for(kk = 0; kk < FeatSelSize /*sc*/; kk++){
            k = feature_sel[kk]; 
			if(param.kernel_type == PRECOMPUTED || samples[k * prob.l + i] != 0){
				cout << " " << samples[k * TotIstanze + i]; 
			}
		}
        cout << endl; 
	}
    // exit(1);
}
// FINE DI ROBA



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

Job::Job(){
    assegnato_svm=false;
    assegnatiDatiTraining=false; 
    assegnatiDatiTest=false;   
    assegnatoProblema=false; 
}

void Job::load_and_run(int iRip_,Problema *pr_,int FeatSelSize_,int FeatSelRip_,int LabelSelIdx_,bool Print_, struct svm_parameter param_){
    //
    Print=Print_; 
    param=&param_; 
    svm_.updateParam(param); // aggiorno i parametri della svm 
    iRip=iRip_;
    FRip=FeatSelRip_; 
    ds.assegnoProblema(pr_); 
    UpdateDatiSimulazione(); // una volta all'inizio e basta 
    // FeatSelSize
    FeatSelSize=FeatSize[1]; 
    TotFeatSize=FeatSelSize; 
    featRandomSelection=false;
    if (FeatSelSize_!=-1){
        FeatSelSize=FeatSelSize_; 
        featRandomSelection=true; 
    }
    feature_sel=new int [FeatSelSize];
    UpdateFeatureSelection(); 
    // FeatSelRip
    if (FeatSelRip_==-1)
        FRip=1; 
    else
        FRip=FeatSelRip_; 
    // LabelSelIdx
    LabelSelSize=LabelSize[1];
    if (LabelSelIdx_==-1){
        LabelSelIdx=new int [LabelSelSize]; 
        for (int i=0; i<LabelSelSize; ++i)
            LabelSelIdx[i]=i; 
    }else{
        LabelSelSize=1; 
        LabelSelIdx=new int [LabelSelSize]; 
        LabelSelIdx[0]=LabelSelIdx_; 
    }
    //
    string percorsoCompleto=ds.pr->nome; 
    nome=FunUtili::path2filename(percorsoCompleto,"/");
    path=FunUtili::path2pathtofile(percorsoCompleto,nome);
    nome.erase(nome.find(".mat"));
    resFile =path; resFile+=nome; 
    resFile+="-iRip_";  resFile+=to_string(iRip); resFile+="-"; 
    //
    RES = new matvar_t* [3];
    TRENDs = new matvar_t* [3];
    //
    assegnatoProblema=true; 
    run(); 
}


Job::~Job(){
    if (assegnatoProblema==true){
        delete [] RES;     
        delete [] TRENDs;     
        delete [] feature_sel; 
        delete [] LabelSelIdx; 
    }
}


// punto il problema
void Job::UpdateDatiSimulazione(){
    // TrSize=ds.LabelTrSelSize;
    // TsSize=ds.LabelTsSelSize;
    FeatSize=ds.Features->dims; 
    LabelSize=ds.Labels->dims; 
    features=(double*)ds.Features->data; 
    labels=(double*)ds.Labels->data;
    VARIABLEs=(matvar_t*)ds.VARIABLEs; 
    //
    assegnatiDatiTraining=true; 
    assegnatiDatiTest=true; 
    return;     
}  


void Job::UpdateFeatureSelection(){
    int i,*feature_sel_=new int [TotFeatSize];
    if (featRandomSelection){
        for (i=0; i<TotFeatSize; ++i)
            feature_sel_[i]=i;    
        FunUtili::randperm(TotFeatSize,feature_sel_); // mescolo  
        for (i=0; i<FeatSelSize; ++i)
            feature_sel[i]=feature_sel_[i]; 
    }else{
        for (int i=0; i<FeatSelSize; ++i)
            feature_sel[i]=i;    
    }
    delete [] feature_sel_;    
    return ;
}


void Job::TrainingFromAssignedProblem(int labelIdx){ 
    if (assegnatiDatiTraining){
        // leggo il problema per addestrare il modello svm 
        svm_.initStrutturaDati(ds.LabelTrSelSize,ds.label_training_selection,  // info sul Training Set
                               FeatSelSize,feature_sel,                        // info sulle feature da usare
                               labelIdx,                                       // info sulla label da usare
                               features,FeatSize,labels,LabelSize);            // info su tutto il dataset
        svm_.train();
        assegnato_svm=true; 
    }else{
        fprintf(stderr,"Non so quali dati usare per fare il Training del modello \n");
        exit(1);       
    }
    return ; 
}



void Job::predictTestSet(int labelIdx){ 
    if (assegnato_svm){
        svm_.predict(RES,                                           // contenitore per i risultati
                     Print,                                         // stampo a video le performances
                     ds.LabelTsSelSize,ds.label_test_selection,     // info sul Test Set 
                     FeatSelSize,feature_sel,                       // info sulle feature da usare 
                     labelIdx,                                      // info sulla label da usare
                     features,FeatSize,labels,LabelSize);           // info su tutto il dataset
    }else{
        fprintf(stderr,"Non ho addestrato un modello svm\n");
        exit(1);       
    }
    return ; 
}


void Job::predictValidationSet(double *ValidTrend,int labelIdx){
    if (assegnato_svm){
        if (Print)
            cout << endl << "validation" << endl; 
        svm_.predict(TRENDs,                                           // contenitore per i risultati
                     Print,                                            // stampo a video le performances
                     ds.LabelValSelSize,ds.label_valid_selection,      // info sul Test Set 
                     FeatSelSize,feature_sel,                          // info sulle feature da usare 
                     labelIdx,                                         // info sulla label da usare
                     features,FeatSize,labels,LabelSize);              // info su tutto il dataset
        double *res = (double *)((matvar_t *)TRENDs[2])->data; 
        for (int i=0; i<ds.LabelValSelSize; ++i)
            ValidTrend[i]=res[i]; 
    }else{
        fprintf(stderr,"Non ho addestrato un modello svm\n");
        exit(1);       
    }
    return ; 
}



void Job::run(){
    if (Print)
        cout << endl << "------ new run ------ iRip==" << iRip << endl << endl;  
    //
    int iTr=0,iTs=0,iVar,iV,iFRip, 
        i,ii,
        trsz=ds.pr->TrSzD, tssz=ds.pr->TsSzD, valdim=ds.LabelValSelSize; 
    double acc[trsz][tssz][FRip], mse[trsz][tssz][FRip], scc[trsz][tssz][FRip],
           featNum[FeatSelSize][FRip],
           *res,trends[valdim][trsz][FRip],actualLabel[valdim],ValidTrend[valdim]; 
    matvar_t *varnameC;
    //
    for (iVar=0; iVar<LabelSelSize; ++iVar){ 
        iV=LabelSelIdx[iVar]; // questa e` la variabile che vogliamo decodificare
        // -------------- cambio il nome del file in modo da coincidere con la variabile --------------------------------------
        string resFile_=resFile,VARNAME;            // cosi` lo scope e` locale in questo ciclo
        varnameC = Mat_VarGetCell(VARIABLEs,iV);    // recupero il nome di questa variabile
        VARNAME=(const char*)varnameC->data;        
        // segnalo dove sono a video
        if (Print)
            cout << endl << "------ iV=" << VARNAME << endl << endl;  
        // 
        resFile_+=VARNAME; resFile_+="-";           // riporto quale variabile sto esaminando nel filename
        resFile_+="nF_"; resFile_+=to_string(FeatSelSize); resFile_+="-"; // e quante features
        resFile_+="res.mat";    // aggiungo l'estensione 
        // elimino gli spazi
        resFile_.erase(remove_if(resFile_.begin(), 
                                 resFile_.end(),
                                 [](char x){return isspace(x);}),
                       resFile_.end());                              
        // --------------------------------------------------------------------------------------------------------------------
        //
        // ----------------------------  recupero l'andamento della label sul validation set ----------------------------------
        for (ii=0; ii<valdim; ++ii){
            i=ds.label_valid_selection[ii];
            actualLabel[ii]=labels[LabelSize[0]*iV+i]; // prendo l'i-esimo indice del del validation set della variabile iV
                                                       // LabelSize[0] e` il numero totale delle istanze
        }
        // ---------------------------------------------------------------------------------------------------------------------  
        //
        for (iTr=0; iTr<trsz; ++iTr){
            ds.assegnoTrainingSet(iTr);  
            for (iFRip=0; iFRip<FRip; ++iFRip){
                if (Print)
                    cout << endl << "------ iTr%=" << (double)iTr/trsz     
                         << endl << "------ iFRip%=" << (double)iFRip/FRip 
                         << endl << endl;  
                UpdateFeatureSelection(); // cambio, se devo, la selezione delle features
        //
        // -----------------------------------------  recupero gli indici delle features ---------------------------------------
                for (i=0; i<FeatSelSize; ++i) // recupero gli indici delle feature che ho usato
                    featNum[i][iFRip]=(double)(feature_sel[i]+1); // per metterla nel ws di matio
                                                                  // aggiungo 1.0 per come funziona l'indicizzazione su matlab
        // ---------------------------------------------------------------------------------------------------------------------  
        //
// per debug
if (0){
    cout << endl << "feat idx" << endl; 
    for (int i=0; i<FeatSelSize; ++i)
        cout << " " << feature_sel[i];
    cout << endl;    
}
                assegnato_svm=false; // devo riaddestrare per ogni Training Set o per ogni sottocampionamento delle features  
                TrainingFromAssignedProblem(iV); // addestro con lo stesso Training Set ma (forse) diverse Features
                predictValidationSet(ValidTrend,iV); // Test sul Validation Set

                // ------------------------------- recupero i trends sul Validation Set -----------------------------------------
                for (ii=0; ii<valdim; ++ii)
                    trends[ii][iTr][iFRip]=ValidTrend[ii]; 
                // --------------------------------------------------------------------------------------------------------------
                //
                for (iTs=0; iTs<tssz; ++iTs){
                    ds.assegnoTestSet(iTs);
                    predictTestSet(iV); 
                    // recupero le performance del modello
                    res = (double *)((matvar_t *)RES[1])->data; 
                    acc[iTr][iTs][iFRip]=res[0];
                    mse[iTr][iTs][iFRip]=res[1];
                    scc[iTr][iTs][iFRip]=res[2];
                }
            }
        }
        // -------------- salvo un file per ogni variabile | setto il workspace da salvare --------------------------------------
        int WsSize=12; 
        matvar_t *workspace[WsSize]; 
        size_t dims_3[3], dims_2[2], dims_1[1]; 
        dims_3[2]=trsz; dims_3[1]=tssz; dims_3[0]=FRip; 
        dims_2[1]=FeatSelSize; dims_2[0]=FRip;
        dims_1[0]=1;
        workspace[0] = Mat_VarCreate("acc",MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims_3,acc,0);
        workspace[1] = Mat_VarCreate("mse",MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims_3,mse,0);
        workspace[2] = Mat_VarCreate("scc",MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims_3,scc,0);
        workspace[3] = Mat_VarCreate("featIdx",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims_2,featNum,0);
        dims_2[1]=valdim; dims_2[0]=1;
        workspace[4] = Mat_VarCreate("actualLabel",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims_2,actualLabel,0);
        dims_3[2]=valdim; dims_3[1]=trsz; dims_3[0]=FRip; 
        workspace[5] = Mat_VarCreate("trends",MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims_3,trends,0);
        double FRip_ws[1], trsz_ws[1], tssz_ws[1], FeatSelSize_ws[1], validSz_ws[1]; 
        FRip_ws[0]=FRip; trsz_ws[0]=trsz; tssz_ws[0]=tssz; FeatSelSize_ws[0]=FeatSelSize; validSz_ws[0]=valdim; 
        workspace[6] = Mat_VarCreate("FeatSelectionRip",MAT_C_DOUBLE,MAT_T_DOUBLE,1,dims_1,FRip_ws,0);
        workspace[7] = Mat_VarCreate("NumFeatures",MAT_C_DOUBLE,MAT_T_DOUBLE,1,dims_1,FeatSelSize_ws,0);
        workspace[8] = Mat_VarCreate("TrSz",MAT_C_DOUBLE,MAT_T_DOUBLE,1,dims_1,trsz_ws,0);
        workspace[9] = Mat_VarCreate("TsSz",MAT_C_DOUBLE,MAT_T_DOUBLE,1,dims_1,tssz_ws,0);
        workspace[10] = Mat_VarCreate("ValidationSetSize",MAT_C_DOUBLE,MAT_T_DOUBLE,1,dims_1,validSz_ws,0);
        string help ="CONTENUTO DEL MATFILE:\n";
               help+="{acc,mse,scc}\n";
               help+="\tdimensione RxTsxTr\n";
               help+="\t(accuratezza,mean squared error ed r2\n\toutput di libsvm)\n";
 
               help+="featIdx\n";
               help+="\tdimensione RxN\n";
               help+="\t(indice delle features usate per addestrare\n\t il modello ad ogni ricampionamento)\n";
              

               help+="actualLabel\n";
               help+="\tdimensione 1xV\n";
               help+="\t(valore della label corrispondente al validation set\n\tNB: salvo un file diverso per ogni label)\n";

               help+="trends\n";
               help+="\tdimensione RxTrxV\n";
               help+="\t(predizione della label, ne ho una diverso per ogni modello\n\tNB: ho un modello diverso per ogni\n\t +ricampionamento delle features\n\t +ricampionamento del training set)\n";

               help+="***LEGENDA***\n";
               help+="\tR=FeatSelectionRip\n";
               help+="\tTs=TsSz\n";
               help+="\tTr=TrSz\n";
               help+="\tN=NumFeatures\n";
               help+="\tV=ValidationSetSize\n";
        dims_2[1]=help.size(); dims_2[0]=1;
        workspace[11] = Mat_VarCreate("README",MAT_C_CHAR,MAT_T_UINT8,2,dims_2,(char*)help.c_str(),0);

        // Apro scrivo e chiudo il matfile
        mat_t *Out_MATFILE = Mat_CreateVer((const char*)resFile_.c_str(),NULL,MAT_FT_DEFAULT);	
        for (int iWS=0; iWS<WsSize; ++iWS)
            Mat_VarWrite(Out_MATFILE, workspace[iWS], MAT_COMPRESSION_NONE);
        Mat_Close(Out_MATFILE);
        // ----------------------------------------------------------------------------------------------------------------------
    }

    return; 
}	

