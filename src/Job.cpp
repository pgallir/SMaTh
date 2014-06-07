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

matvar_t* FunUtili::Mat_VarCreate_jr(const char *NomeVar, int raws, int cols) // solo per double!
{
  size_t dims[2] = {(size_t) raws, (size_t) cols};
  double STUPIDVECTOR[dims[0]*dims[1]];
  return Mat_VarCreate(NomeVar,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,STUPIDVECTOR,0); 
}

void FunUtili::predict_jr(matvar_t *plhs[], const matvar_t *prhs[], int *CVSel, int CVSelSize, struct svm_model *model, const int predict_probability)
{
  int label_vector_row_num, label_vector_col_num;
  int feature_number, testing_instance_number;
  int instance_index;
  double *ptr_instance, *ptr_label, *ptr_predict_label; 
  double *ptr_prob_estimates, *ptr_dec_values, ptr[3];
  struct svm_node *x;

  int correct = 0;
  int total = 0;
  double error = 0;
  double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

  int svm_type=svm_get_svm_type(model);
  int nr_class=svm_get_nr_class(model);
  double *prob_estimates=NULL;

  feature_number = (int)(prhs[1]->dims[1]);
  testing_instance_number = (int)(prhs[1]->dims[0]);
  label_vector_row_num = (int)(prhs[0]->dims[0]); 
  label_vector_col_num = (int)(prhs[0]->dims[1]);
  
  if(label_vector_row_num!=testing_instance_number)
  {
    printf("Length of label vector does not match # of instances.\n");
    return;
  }
  if(label_vector_col_num!=1)
  {
    printf("label (1st argument) should be a vector (# of column is 1).\n");
    return;
  }

  ptr_instance = (double *) prhs[1]->data;//mxGetPr(prhs[1]);
  ptr_label    = (double *) prhs[0]->data;//mxGetPr(prhs[0]);

  /* JR: TO DO THINGs 
  if(mxIsParse(prhs[1]))
  {blablablablabla}
  */

  if(predict_probability)
  {
    if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
      printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
    else
      prob_estimates = (double *) malloc(nr_class*sizeof(double));
  }


  // NB: sto settando i valori di uscita?
  plhs[0] = Mat_VarCreate_jr("Out1", CVSelSize, 1); // JR: MA PORCA TROIA! testing_instance_number sti due coglioni!!

  if(predict_probability)
    // prob estimates are in plhs[2]
    if(svm_type==C_SVC || svm_type==NU_SVC)
      plhs[2] = Mat_VarCreate_jr("Out3", CVSelSize, nr_class);
    else
      plhs[2] = Mat_VarCreate_jr("Out3", 0, 0);
  else
    // decision values are in plhs[2]
    if(svm_type == ONE_CLASS ||
       svm_type == EPSILON_SVR ||
       svm_type == NU_SVR ||
       nr_class == 1) // if only one class in training data, decision values are still returned.
      plhs[2] = Mat_VarCreate_jr("Out3", CVSelSize, 1);
    else
      plhs[2] = Mat_VarCreate_jr("Out3", CVSelSize, nr_class*(nr_class-1)/2);
  
  // NB: sto facendo puntare i valori di uscita da un puntatore? (per l'assegnamento)
  ptr_predict_label = (double *) plhs[0]->data;
  ptr_prob_estimates = (double *) plhs[2]->data; 
  ptr_dec_values = (double *) plhs[2]->data;
  x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );

  // JR: VA FATTO UN WRAPPER PER LA LETTURA DEI DATI!
  for(int ii=0; ii<CVSelSize; ii++)
  //for(instance_index=0;instance_index<testing_instance_number;instance_index++)
  { 
    instance_index = CVSel[ii]; 
    int i;
    double target_label, predict_label;
    target_label = ptr_label[instance_index];
    if(0)// JR:mxIsSparse(prhs[1]) && model->param.kernel_type != PRECOMPUTED) // prhs[1]^T is still sparse
      printf("SE HO VOGLIA aggiusto per matrici sparse"); //read_sparse_instance(plhs[0], instance_index, x);
    else
    {
      for(i=0;i<feature_number;i++)
      {
	// NB: assegno ad x i valori puntatida ptr_instance
	x[i].index = i+1;
	x[i].value =  ptr_instance[testing_instance_number*i+instance_index]; // ptr_instance[testing_instance_number*i+instance_index];
      }
      x[feature_number].index = -1;
    }
    if(predict_probability)
    {
      if(svm_type==C_SVC || svm_type==NU_SVC)
      {
	predict_label = svm_predict_probability(model, x, prob_estimates);
	//ptr_predict_label[instance_index] = predict_label;
        ptr_predict_label[ii] = predict_label;
	for(i=0;i<nr_class;i++)
	  //ptr_prob_estimates[instance_index + i * testing_instance_number] = prob_estimates[i];
          ptr_prob_estimates[ii] = prob_estimates[i];
      } else {
	predict_label = svm_predict(model,x);
	//ptr_predict_label[instance_index] = predict_label;
        ptr_predict_label[ii] = predict_label;
      }
    }
    else
    {
      if(svm_type == ONE_CLASS ||
         svm_type == EPSILON_SVR ||
         svm_type == NU_SVR)
      {
	double res;
	predict_label = svm_predict_values(model, x, &res);
	//ptr_dec_values[instance_index] = res;
        ptr_dec_values[ii] = res;
      }
      else
      {
	double *dec_values = (double *) malloc(sizeof(double) * nr_class*(nr_class-1)/2);
	predict_label = svm_predict_values(model, x, dec_values);
	if(nr_class == 1) 
	  //ptr_dec_values[instance_index] = 1;
          ptr_dec_values[ii] = 1;
	else
	  for(i=0;i<(nr_class*(nr_class-1))/2;i++)
	    //ptr_dec_values[instance_index + i * testing_instance_number] = dec_values[i];
            ptr_dec_values[ii] = dec_values[i];
	free(dec_values);
      }
      //ptr_predict_label[instance_index] = predict_label;
      ptr_predict_label[ii] = predict_label;
    }
    if(predict_label == target_label)
      ++correct;
    error += (predict_label-target_label)*(predict_label-target_label);
    sump += predict_label;
    sumt += target_label;
    sumpp += predict_label*predict_label;
    sumtt += target_label*target_label;
    sumpt += predict_label*target_label;
    ++total;
  }
/*
  // print stuff
  if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
  {
    printf("Mean squared error = %g (regression)\n",error/total);
    printf("Squared correlation coefficient = %g (regression)\n",
      ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
      ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
      );
  }
  else
    printf("Accuracy = %g%% (%d/%d) (classification)\n",
      (double)correct/total*100,correct,total);
*/
  // return accuracy, mean squared error, squared correlation coefficient
  size_t dims[2];
  dims[0] = 3; 
  dims[1] = 1;
  ptr[0] = (double)correct/total*100;
  ptr[1] = (double)error/total;
  ptr[2] = (double)(((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
	            ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt)));
  plhs[1] = Mat_VarCreate("Out2",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,ptr,0); // JR: da mettere un nome vero

  // free memory
  free(x);
  ptr_label=NULL;free(ptr_label);
  ptr_predict_label=NULL;free(ptr_predict_label);
  ptr_prob_estimates=NULL;free(ptr_prob_estimates);
  ptr_dec_values=NULL;free(ptr_dec_values);

  if(prob_estimates != NULL)
    free(prob_estimates);
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
    svm_destroy_param(&param);
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
	int i, ii, j, k;
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
	if(param.kernel_type == PRECOMPUTED)
		elements = prob.l * (sc + 1);
	else
	{
		for(ii = 0; ii < prob.l; ii++)
		{
            i=label_training_selection[ii]; 
			for(k = 0; k < sc; k++)
				if(samples[k * prob.l + i] != 0)
					elements++;
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
		prob.y[ii] = labels[i];

		for(k = 0; k < sc; k++)
		{
			if(param.kernel_type == PRECOMPUTED || samples[k * prob.l + i] != 0)
			{
				x_space[j].index = k + 1;
				x_space[j].value = samples[k * prob.l + i];
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

void SVModel::predict(matvar_t **RES, const matvar_t *Features, const matvar_t *Labels, int *labelSelection, int LabelSize){
    const matvar_t *INPUTs[2]; 
    INPUTs[0] = Labels; 
    INPUTs[1] = Features; 
    FunUtili::predict_jr(RES,INPUTs,labelSelection,LabelSize,model,0); 
}

//////////////////// DATI SIMULAZIONE ////////////////////////

DatiSimulazione::DatiSimulazione(){
    problema_assegnato=false; 
    TrainingSet_assegnato=false;  
    TestSet_assegnato=false;  
    LabelTrSelSize=0; 
    LabelTsSelSize=0; 
    NumTrBlockSel=0; 
    NumTrTsBlockSel=0;
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
    if (TrainingSet_assegnato==true){ 
        delete [] block_selection; 
        delete [] label_training_selection; 
        LabelTrSelSize=0; 
        TrainingSet_assegnato=false; 
    }
    // assegno il training_set   
    if (problema_assegnato){
        iTr=iTr_; // e se eccedo? deve controllarlo Job
        int i,ii,jj,k; 
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
    if (TestSet_assegnato==true){ 
        delete [] label_test_selection; 
        LabelTsSelSize=0; 
        TestSet_assegnato=false; 
    }
    if (problema_assegnato &&
        TrainingSet_assegnato){
        iTs=iTs_; // e se eccedo? deve controllarlo Job
        int i,ii,jj,k; 
        NumTrTsBlockSel=NumTrBlockSel+
                        (int) (tsS_[iTs_]*TOT_STEPS/100);
        if (NumTrTsBlockSel>TOT_STEPS){
            fprintf(stderr," Training Set + Test Set > 100 %% \n");
            exit(1);
        }
        for (ii=NumTrBlockSel; ii<NumTrTsBlockSel; ii++){
            i = block_selection[ii];  // recupero la cella opportuna dopo che ho fatto il mescolone
            LabelTsSelSize += CellCV[i]->dims[1]; 
        }
        label_test_selection = new int [LabelTsSelSize];
        k=0;  
        for (ii=NumTrBlockSel; ii<NumTrTsBlockSel; ii++){
            i = block_selection[ii];  // recupero la cella opportuna dopo che ho fatto il mescolone
            double *cell = (double*)CellCV[i]->data;
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
        svm_.predict(RES,ds.Features,ds.Labels,ds.label_test_selection,ds.LabelTsSelSize);
    }else{
        fprintf(stderr,"Non ho addestrato un modello svm\n");
        exit(1);       
    }
    return ; 
}


void Job::run(){
    cout << endl << "-- new run -- iRip==" << iRip << endl;  
    UpdateDatiSimulazione(); // una volta all'inizio e basta 
    int iTr=0,trsz=ds.pr->TrSzD, 
        iTs=0,tssz=ds.pr->TsSzD;  

/*
    // Funziona, sembra almeno... 
    ds.assegnoTrainingSet(iTr);  
    ds.assegnoTestSet(tssz-1);  
    cout << "LabelTsSelSize " << ds.LabelTsSelSize << endl; 
    for (int i=0; i<ds.LabelTsSelSize; ++i)
        cout << " " << ds.label_test_selection[i] << ":" << features[ds.label_test_selection[i]];
    cout << endl;  
    TrainingFromAssignedProblem();    
    predictTestSet();
*/

    for (iTr=0; iTr<1/*trsz*/; ++iTr){
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

