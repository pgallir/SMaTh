#include "SvmMatio.h" 
#include <iostream>

using namespace std; 

void print_null(const char *s) {}

// JR: exit_input_error() presa pari pari da libsvm
// 
void exit_input_error(int line_num)
{
  fprintf(stderr,"Wrong input format at line %d\n", line_num);
  exit(1);
}

// JR: readline(), modificata da libsvm
//
char* readline(FILE *input, char *line, int max_line_len)
{
  int len;
  if(fgets(line,max_line_len,input) == NULL)
    return NULL;
  while(strrchr(line,'\n') == NULL)
  {
    max_line_len *= 2;
    line = (char *) realloc(line,max_line_len);
    len = (int) strlen(line);
    if(fgets(line+len,max_line_len-len,input) == NULL)
    break;
  }
  return line;
}


// JR: parse_VIarguments(), Very Important arguments are .mat filename and WhaToDo
//
void parse_VIarguments(int argc, char **argv, char *IFN, struct svm_parameter *param, double *DimFeatures, double *RipFeatures, double *Label, int *perc_sig)
{
  char *ptrS; 
  
  // check if we have enough inputs - consider that the exe counts as 1 
  if(argc<=1)
  {
    fprintf(stderr,"\nYou need at least one argument: the filename.mat or --help for instructions.\n"); 
    exit_with_help();
  }
  // check if in the last argument is a .mat file: 
  ptrS = strstr(argv[1],".mat");
  if(ptrS==NULL)
  {
    ptrS = strstr(argv[1],"help");
    if (ptrS!=NULL)
    {
      exit_with_help(); 
    }else{
      fprintf(stderr,"\nfilename=%s\nYou need to specify the .mat filename (with extension) or --help as first argument\n",argv[1]); 
    }
  }else
    IFN = strcpy(IFN,argv[1]);

  // JR: remove matfile argc,argv, then parse_command_line
  --argc; 
  for (int p=1; p<argc; p++)
    argv[p]=argv[p+1]; 
  parse_command_line(argc, argv, param, DimFeatures, RipFeatures, Label, perc_sig);
  // svm_parameterPrint(*param);

  return; 
}


// JR: checking if the variables I loaded from MATFILE are set properly
const char* Mat_VarErrCheck(const matvar_t *PARAM[])
{
  //PARs:={Features, Labels, VARIABLEs, RIP, idxCV, idxVS, TrSz, TsSz}
  const matvar_t *Features = PARAM[0];	// features matrix
  const matvar_t *Labels = PARAM[1]; 	// label vector (matrix??) for the different variables
  const matvar_t *VARIABLEs = PARAM[2];	// cells containing variables names
  const matvar_t *RIP = PARAM[3]; 	// #repetition of CV
  //const matvar_t *BlockCV = PARAM[4]; 	// cell containing the CROSS VALIDATION indexes subdivision
  //const matvar_t **Cell = (const matvar_t**) BlockCV->data;
  //const matvar_t *BlockVS = PARAM[5]; 	// cell containing the VALIDATION SET indexes subdivision
  //const matvar_t **CellV = (const matvar_t**) BlockVS->data;
  const matvar_t *trS = PARAM[6]; 	// # of BlockCV cells to be used for training the svm model
  const matvar_t *tsS = PARAM[7]; 	// # of BlockCV cells to be used for test the svm model

  // NB
  // Features: matrix of features 	(RxC1)	, R samples of C1 features
  // Labels: matrix of labels	(RxC2)	, R samples of C2 variables
  // VARIABLEs: vector of variables' name (C2x1)
  // idxCV: indexes for Cross Validation - cellarray	(1xM)	of vectors (1xNm) , M slices of data with Nm samples each 
  // idxVS: indexes of the Validation Set - cellarray    (1xB)   of vectors (1xNb) , B slices of data with Nb samples each
  // {RIP,trS,tsS}:  # CV numerosity (1x1), vectors of Training Size (1xTr) and vectors of Test Size (1xTs)   
	
  // 1) Check R rows for Features and Labels
  if (Features->dims[0]!=Labels->dims[0])
    return "Features and Labels have a different number of samples"; 
  // 2) Check C2 for Labels and VARIABLEs
  if (Labels->dims[1]<VARIABLEs->dims[0])
    return "Each tested variable needs a column of Labels";
  else if (Labels->dims[1]>VARIABLEs->dims[0])
    return "Each column of Labels needs a variable name";
  // 3) Check RIP >= 0
  double *INT = (double *) RIP->data; 
  //printf("\nRIP = %d\n", (int)(rip[0]));
  if ((int)INT[0] <=0 )
    return "RIP must be >= 0";
  // 4) Check that T{r,s}Sz combinations are monotonic increasing. 
  if  (((int) trS->dims[1]==0) || 
      ((int) tsS->dims[1]==0)    )
    return "T{r,s}Sz are empty";
  else if (((int) trS->dims[1]>1) || 
      	  ((int) tsS->dims[1]>1)    )
  {
    double *trS_ = (double *)trS->data; 
    double *tsS_ = (double *)tsS->data; 
    for (int i=1; i<(int)trS->dims[1]; i++)
      if (trS_[i]<trS_[i-1])
	return "trS must be a monotonic increasing funciton";
    for (int i=1; i<(int)tsS->dims[1]; i++)
      if (tsS_[i]<tsS_[i-1])
	return "tsS must be a monotonic increasing funciton";
  }
  return NULL;
}



// JR: parse_command_line(), modificata da libsvm  
//
void parse_command_line(int argc, char **argv, struct svm_parameter *param, double *DimFeatures, double *RipFeatures, double *Label, int *perc_sig)
{
  int i;
  void (*print_func)(const char*) = NULL; // default printing to stdout
  // default values
  param->svm_type = EPSILON_SVR;
  param->kernel_type = RBF;
  param->degree = 3;
  param->gamma = 0; // 1/num_features
  param->coef0 = 0;
  param->nu = 0.5;
  param->cache_size = 100;
  param->C = 1;
  param->eps = 1e-3;
  param->p = 0.1;
  param->shrinking = 1;
  param->probability = 0;
  param->nr_weight = 0;
  param->weight_label = NULL;
  param->weight = NULL;
  // parse options
  for(i=1;i<argc;i++)
  {
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
	break;
      // JR adds
      case '-':
	if	 (string(argv[i]) == "--help")
	{
	  fprintf(stderr,"Use --help alone\n");
	  exit(1);
	}else if (string(argv[i-1]) == "--#features"){
	  if (i>=argc || argv[i][0] == '-'){fprintf(stderr,"You forgot to set value @ %s\n",argv[i-1]); exit(2);}
	  char *tokens[2], *token_ptr;
	  token_ptr = strtok((char*)argv[i],",:"); 
	  for (int itok=0; itok<2 && token_ptr != NULL; itok++) 	  
	  {
	    tokens[itok] = token_ptr;
	    token_ptr = strtok(NULL, " ");
	  }
	  *DimFeatures = (double)atoi(tokens[0]); 
	  *RipFeatures = (double)atoi(tokens[1]); 
	}else if (string(argv[i-1]) == "--label[i]"){
	  if (i>=argc || argv[i][0] == '-'){fprintf(stderr,"You forgot to set value @ %s\n",argv[i-1]); exit(2);}
	  *Label = (double)atoi(argv[i]);
	}else if (string(argv[i-1]) == "--perc_sig"){
	  if (i>=argc || argv[i][0] == '-'){fprintf(stderr,"You forgot to set value @ %s. It can be 0 (default value) or 1. 0 means that the signal folds are measured in steps, 1 means that are measured in percentage of the whole recorded signal\n",argv[i-1]); exit(2);}
	  *perc_sig = (double)atoi(argv[i]);	 
	  if (*perc_sig != 0 && *perc_sig !=1){fprintf(stderr,"perc_sig can assume {0,1} value\n"); exit(2);} 
	}else{
	  fprintf(stderr,"%s is not correct. Use --help for further infos\n",argv[i-1]);
	  exit(1);
	}
	break;
      // end JR adds 
      default:
	fprintf(stderr,"%s is not correct. Use --help for further infos\n",argv[i-1]);
	exit(1);
    }
  }
  svm_set_print_string_function(print_func);
  // determine filenames
}


// JR: print svm_parameter
void svm_parameterPrint(struct svm_parameter param)
{
  printf( "\nsvm_type: %d"
	  "\nkernel_type: %d"
	  "\ndegree: %d"
	  "\ngamma: %f"
	  "\ncoef0: %f"
	  "\ncache_size: %f"
	  "\neps: %f"
	  "\nC: %f"
	  "\nnr_weight: %d"
	  "\nnu: %f"
	  "\np: %f"
	  "\nshrinking: %d"
	  "\nprobability: %d", param.svm_type, param.kernel_type, param.degree, param.gamma, param.coef0, param.cache_size, param.eps, param.C, param.nr_weight, param.nu, param.p, param.shrinking, param.probability); 
  return; 
};


// JR: print an svm_problem
void svm_problemPrint(struct svm_problem prob)
{
  printf("\nl: %d", prob.l);
  return; 
};


/*
// JR: do_cross_validation(), modificata da libsvm
//  - RILEGGITI BENE COSA FA QUESTA FUNZIONE!! - 
void do_cross_validation(struct svm_problem prob, struct svm_parameter param, int nr_fold)
{
  int i;
  int total_correct = 0;
  double total_error = 0;
  double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
  double *target = Malloc(double,prob.l);

  svm_cross_validation(&prob,&param,nr_fold,target);
  if(param.svm_type == EPSILON_SVR ||
     param.svm_type == NU_SVR)
  {
    for(i=0;i<prob.l;i++)
    {
      double y = prob.y[i];
      double v = target[i];
      total_error += (v-y)*(v-y);
      sumv += v;
      sumy += y;
      sumvv += v*v;
      sumyy += y*y;
      sumvy += v*y;
    }
    printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
    printf("Cross Validation Squared correlation coefficient = %g\n",
      ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
      ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
          );
  }
  else
  {
    for(i=0;i<prob.l;i++)
      if(target[i] == prob.y[i])
	  ++total_correct;
    printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
  }
  free(target);
}
*/



// JR:  modificata read_probem() da libsvm
// read in a problem (in svmlight format)
//
void read_problem_from_file(char *filename, struct svm_problem *prob, struct svm_parameter *param)
{
  int elements, max_index, inst_max_index, i, j;
  FILE *fp = fopen(filename,"r");
  char *endptr;
  char *idx, *val, *label;
  struct svm_node *x_space; 

  if(fp == NULL)
  {
    fprintf(stderr,"can't open input file %s\n",filename);
    exit(1);
  }

  prob->l = 0;
  elements = 0;

  int max_line_len = 1024;
  char *line = Malloc(char,max_line_len);
  while(readline(fp,line,max_line_len)!=NULL)
  {
    char *p = strtok(line," \t"); // label

    // features
    while(1)
    {
      p = strtok(NULL," \t");
      if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
	break;
      ++elements;
    }
    ++elements;
    ++prob->l;
  }
  rewind(fp);

  prob->y = Malloc(double,prob->l);
  prob->x = Malloc(struct svm_node *,prob->l);
  x_space = Malloc(struct svm_node,elements);

  max_index = 0;
  j=0;
  for(i=0;i<prob->l;i++)
  {
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    readline(fp,line,max_line_len);
    prob->x[i] = &x_space[j];
    label = strtok(line," \t\n");
    if(label == NULL) // empty line
      exit_input_error(i+1);

    prob->y[i] = strtod(label,&endptr);
    if(endptr == label || *endptr != '\0')
      exit_input_error(i+1);

    while(1)
    {
      idx = strtok(NULL,":");
      val = strtok(NULL," \t");

      if(val == NULL)
	break;

      errno = 0;
      x_space[j].index = (int) strtol(idx,&endptr,10);
      if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
	exit_input_error(i+1);
      else
	inst_max_index = x_space[j].index;

      errno = 0;
      x_space[j].value = strtod(val,&endptr);
      if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
	exit_input_error(i+1);

      ++j;
    }

    if(inst_max_index > max_index)
      max_index = inst_max_index;
    x_space[j++].index = -1;
  }

  if(param->gamma == 0 && max_index > 0)
    param->gamma = 1.0/max_index;

  if(param->kernel_type == PRECOMPUTED)
    for(i=0;i<prob->l;i++)
    {
      if (prob->x[i][0].index != 0)
      {
	fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
	exit(1);
      }
      if ((int)prob->x[i][0].value <= 0 || (int)prob->x[i][0].value > max_index)
      {
	fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
	exit(1);
      }
    }
  fclose(fp);
  free(line);
}


// JR: read_probem_from_variable() -- 
// read in a problem from full matrix (NB: FULL MATRIX, not sparse)
//
const char* read_problem_from_variable(const matvar_t *PARAM[], int iTr, int *&TestPotentialSelection, int *LabelTsCellSize, int perc_sig, struct svm_problem *prob, struct svm_parameter *param)
{

  //PARs:={Features, Labels, VARIABLEs, RIP, idxCV, idxVS, TrSz, TsSz}
  const matvar_t *Features = PARAM[0];	// features matrix
  const matvar_t *Labels = PARAM[1]; 	// label vector (matrix??) for the different variables
  //const matvar_t *VARIABLEs = PARAM[2];	// cells containing variables names
  //const matvar_t *RIP = PARAM[3]; 	// #repetition of CV
  const matvar_t *BlockCV = PARAM[4]; 	// cell containing the CROSS VALIDATION indexes subdivision
  const matvar_t **Cell = (const matvar_t**) BlockCV->data;
  //const matvar_t *BlockVS = PARAM[5]; 	// cell containing the VALIDATION SET indexes subdivision
  //const matvar_t **CellV = (const matvar_t**) BlockVS->data;
  const matvar_t *trS = PARAM[6]; 	// # of BlockCV cells to be used for training the svm model
  //const matvar_t *tsS = PARAM[7]; 	// # of BlockCV cells to be used for test the svm model

  int elements, max_index, i, ii, j, k, *label_training_selection, block_selection[BlockCV->dims[1]], LabelTrSelSize = 0, nRPsel;
  double *trS_;
  struct svm_node *x_space; 
  double *f = (double*) Features->data; 
  double *l = (double*) Labels->data;

  //Mat_VarPrint(trS,1);
  trS_ = (double*) trS->data; 
  if (perc_sig==0)
    nRPsel = (int)trS_[iTr]; // number of blocks to consider for CV
  else if(perc_sig==1)
    nRPsel = (int)(trS_[iTr]*BlockCV->dims[1]/100);

  // permute randomly the training selection 
  randperm((int)BlockCV->dims[1],block_selection); 

  // fill the label
  // get the memory dimension of the label I need to allocate
  for (ii=0; ii<nRPsel; ii++)
    LabelTrSelSize += Cell[ii]->dims[1];
  // allocate memory abd assign block of {training,test} patterns
  *LabelTsCellSize = BlockCV->dims[1] - nRPsel; // store in *LabelTsCellSize the number of block remained for test after training phase
  label_training_selection = Malloc(int,LabelTrSelSize);
  TestPotentialSelection = Malloc(int,*LabelTsCellSize);
  k=0; j=0; 
  for (ii=0; ii<(int)BlockCV->dims[1]; ii++)
  {
    i = block_selection[ii]; 
    double *cell = (double*)Cell[i]->data;
    if (ii<nRPsel)
      for (int jj=0; jj<(int)Cell[i]->dims[1]; jj++)
	label_training_selection[k++] = (int)cell[jj];
    else
      TestPotentialSelection[j++] = i;
  }

  // set dimensions problem
  prob->l = LabelTrSelSize;				      // #labels
  elements = prob->l + (Features->dims[0]*Features->dims[1]); // #labels (1 + #features)
  max_index = Features->dims[1];			      // #features  -- here only a full matrix had been considered
    
  // allocate enough memory
  prob->y = Malloc(double,prob->l);
  prob->x = Malloc(struct svm_node *,prob->l);
  x_space = Malloc(struct svm_node,elements);

  k=0; // init # svm-s
  for (ii=0; ii<prob->l; ii++) // run labels over label_training_selection indexes
  {
    // get the proper pattern index
    i = label_training_selection[ii];
    // set prob fields
    prob->y[ii] = l[i];
    prob->x[ii] = &x_space[k]; 
    for (j=0; j<(int)Features->dims[1]; j++) // run features
    {
      x_space[k].index = (long int)(j+1); 
      x_space[k].value = f[j*Labels->dims[0]+i];
      ++k; 
    }
    x_space[k++].index = -1; // end of each little problem
  }   
    
  if(param->gamma == 0 && max_index > 0)
    param->gamma = 1.0/max_index;

  if(param->kernel_type == PRECOMPUTED)
    return "I do NOT consider a precumputed kernel_type by now";	

  //remember to free your allocated memory
  free(label_training_selection);
  l=NULL;free(l); 
  f=NULL;free(f);
  x_space=NULL;free(x_space);
  trS_=NULL;free(trS_);

  return NULL; 
}

// JR: test_model_on_data_subset() -- 
// test a model on a subset of data defined in a full matrix (NB: FULL MATRIX, not sparse)
matvar_t** test_model_on_data_subset(const matvar_t *PARAM[], int iTs, int *TestPotentialSelection, int LabelTsCellSize, struct svm_model *model, int predict_probability, int perc_sig)
{
  //PARs:={Features, Labels, VARIABLEs, RIP, idxCV, idxVS, TrSz, TsSz}
  const matvar_t *Features = PARAM[0];	// features matrix
  const matvar_t *Labels = PARAM[1]; 	// label vector (matrix??) for the different variables
  //const matvar_t *VARIABLEs = PARAM[2];	// cells containing variables names
  //const matvar_t *RIP = PARAM[3]; 	// #repetition of CV
  const matvar_t *BlockCV = PARAM[4]; 	// cell containing the CROSS VALIDATION indexes subdivision
  const matvar_t **Cell = (const matvar_t**) BlockCV->data;
  //const matvar_t *BlockVS = PARAM[5]; 	// cell containing the VALIDATION SET indexes subdivision
  //const matvar_t **CellV = (const matvar_t**) BlockVS->data;
  //const matvar_t *trS = PARAM[6]; 	// # of BlockCV cells to be used for training the svm model
  const matvar_t *tsS = PARAM[7]; 	// # of BlockCV cells to be used for test the svm model

  int i, ii, j, k, block_selection[LabelTsCellSize], *label_test_selection, nRPsel, LabelTsSelSize=0;
  double *tsS_ = (double*) tsS->data; 

  if (perc_sig==0)
    nRPsel = (int)tsS_[iTs];  // number of blocks to consider for CV
  else if(perc_sig==1)
    nRPsel = (int)(tsS_[iTs]*BlockCV->dims[1]/100); 

  // permute randomly the test selection 
  randperm(LabelTsCellSize, block_selection);
  
  // get the memory dimension of the label I need to allocate for label_test_selection
  k=0; 
  for (ii=0; ii<nRPsel; ii++)
  {
    i = TestPotentialSelection[block_selection[ii]]; 
    LabelTsSelSize += Cell[i]->dims[1];
  }
  label_test_selection = Malloc(int, LabelTsSelSize);
  for (ii=0; ii<nRPsel; ii++)
  {
    i = TestPotentialSelection[block_selection[ii]]; 
    double *cell = (double*)Cell[i]->data;
    for (j=0; j<(int)Cell[i]->dims[1]; j++)
      label_test_selection[k++] = (int)cell[j];
  }

  // do the prediction
  matvar_t **RES = Malloc(matvar_t*,3);
  const matvar_t *INPUTs[2]; 
  INPUTs[0] = Labels; 
  INPUTs[1] = Features; 
  predict_jr((matvar_t **) RES, (const matvar_t **)INPUTs, label_test_selection, LabelTsSelSize, model, int(0)); //JR: predict_probability sempre == 0

  //remember to free your allocated memory
  free(label_test_selection);

  return RES;   
}


// JR: validate_model() -- 
// quite the same of test_model_on_data_subset()
// it tests a model on the validation set a priori chosen
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


// JR: from groups.csail.mit.edu
//
void randperm(int n,int perm[])
{
  int i, j, t;
  for(i=0; i<n; i++)
    perm[i] = i;
  for(i=0; i<n; i++) 
  {
    j = rand()%(n-i)+i;
    t = perm[j];
    perm[j] = perm[i];
    perm[i] = t;
  }
}



// JR: modified from Mat_VarCreate() 
// needed just to interface with Mat_VarCreate of matio
matvar_t* Mat_VarCreate_jr(const char *NomeVar, int raws, int cols) // solo per double!
{
  size_t dims[2] = {(size_t) raws, (size_t) cols};
  double STUPIDVECTOR[dims[0]*dims[1]];
  return Mat_VarCreate(NomeVar,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,STUPIDVECTOR,0); 
}

// JR: modified version of predict() from libsvm
//     adds: here the type matvar_t * from matio has been used
void predict_jr(matvar_t *plhs[], const matvar_t *prhs[], int *CVSel, int CVSelSize, struct svm_model *model, const int predict_probability)
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
  
/*
  if(label_vector_row_num!=testing_instance_number)
  {
    printf("Length of label vector does not match # of instances.\n");
    return;
  }
*/
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


void exit_with_help()
{
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

