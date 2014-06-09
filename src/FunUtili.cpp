#include "FunUtili.h"

//////////////////// FunUtili ////////////////////////

string FunUtili::path2pathtofile(string path, string filename){ 
    string pathtofile; 
    pathtofile.assign(path.begin(), path.end()-filename.size());
    return pathtofile; 
}


string FunUtili::path2filename(string path, const char* separator){ 
    string filename;
    size_t pos = path.find_last_of(separator);  // dipende dal sistema operativo
    if(pos != string::npos)
        filename.assign(path.begin() + pos + 1, path.end());
    else
        filename = path;
    return filename; 
}

void FunUtili::randperm(int n,int *perm){  
    // NB: meglio se n e` la dimensione di perm. 
    //     ma funziona anche se e` un intero piu` piccolo
    srand(time(NULL)); // cambio seme
    int i, j, t;
    vector <int> p; 
    for(i=0; i<n; i++)
        p.push_back(perm[i]);
    i=0; 
    while (p.size()!=0){
        j=(rand()%(int)(p.size())); // output = min + (rand() % (int)(max - min + 1))
        t=p[j];              
        perm[i++]=t;           
        p.erase(p.begin()+j); 
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
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
    "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
    "\E[1m"
    "3) advanced options\n\n"
    "\E[0m"
    "--#features       : if you don't want to use all features, you can here set how many features to be used (chosen randomly)\n"
    "                    NB: you need to specify how many time you want to change them, becuse you use the same features RIP time among folds\n"
    "                    ex: '--#features 5:10' will select 10 times 5 features and simulate the problem defined in the matfile\n"
    "--label[i]        : if you don't want to run the simulations across all variables, you can here set what label should be used\n"
    "                    ex: '--label[i] 2' will select the 3rd column of OUT (see 1) matfile)\n"
    "--print           : {0,1} enables the print of the status of the simulation on the screen (if not multithreding). default is off. \n"
    "--multiThreading  : {0,1} enables multithreading. default is off. when on, disable the print option.\n"
    "\n");
    exit(1);
}

void FunUtili::parseArguments(int argc, char **argv, 
                              string *filename, int *DimFeatures, int *RipFeatures, int *Label,
                              struct svm_parameter *param, 
                              bool *Print, bool *MultiThreading){
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
                *DimFeatures = (int)atoi(tokens[0]); 
                *RipFeatures = (int)atoi(tokens[1]); 
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
                *Label = (int)atoi(argv[i+1]);
            }else if(string(argv[i]) == "--multiThreading"){
                if (i+1>=argc || argv[i+1][0] == '-'){
                    fprintf(stderr,"You forgot to set value @ %s\n",argv[i]); 
                    exit(2);
                }
                int mlt = (int)atoi(argv[i+1]);
                if (mlt==1)
                    *MultiThreading=true; 
                else if (mlt==0)
                    *MultiThreading=false;
                else{
                    fprintf(stderr,"Wrong argument for --multiThreading. Values can be 0 or 1\n"); 
                    exit(2);
                }
            }else if(string(argv[i]) == "--print"){
                if (i+1>=argc || argv[i+1][0] == '-'){
                    fprintf(stderr,"You forgot to set value @ %s\n",argv[i]); 
                    exit(2);
                }
                int print = (int)atoi(argv[i+1]);
                if (print==1)
                    *Print=true; 
                else if (print==0)
                    *Print=false;
                else{
                    fprintf(stderr,"Wrong argument for --print. Values can be 0 or 1\n"); 
                    exit(2);
                }
            }else{
                fprintf(stderr,"%s is not correct. Use --help for further infos\n",argv[i]);
                exit(1);
            }
        }
    }

    // valori di default
    param->svm_type=-1; 
    param->kernel_type=-1;
    param->degree=-1;
    param->gamma=-1;
    param->coef0=-1;
    param->nu=-1; 
    param->cache_size=-1;
    param->C=-1;
    param->eps=-1; 
    param->p=-1; 
    param->shrinking=-1;
    param->probability=-1;
    param->nr_weight=-1;
    param->weight_label=NULL;
    param->weight=NULL;
    // parse options
    for(int ii=1;ii<argc;ii++){
        int i=ii+1; 
        if(argv[ii][0] == '-') 
        switch(argv[ii][1])
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
            case 'w':
                ++param->nr_weight;
                param->weight_label = (int *)realloc(param->weight_label,sizeof(int)*param->nr_weight);
                param->weight = (double *)realloc(param->weight,sizeof(double)*param->nr_weight);
                param->weight_label[param->nr_weight-1] = atoi(&argv[i-1][2]);
                param->weight[param->nr_weight-1] = atof(argv[i]);
       }
    }
}

matvar_t* FunUtili::Mat_VarCreate_jr(const char *NomeVar, int raws, int cols) // solo per double!
{
    size_t dims[2] = {(size_t) raws, (size_t) cols};
    double STUPIDVECTOR[dims[0]*dims[1]];
    return Mat_VarCreate(NomeVar,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,STUPIDVECTOR,0); 
}


void FunUtili::predict_jr(matvar_t *plhs[],double *f,size_t *FeatSize,double *l,size_t *LabelSize,      // contenitori {ingressi,uscite}
                         int *testing_idx,int testing_instance_number,                                  // pattern da predirre 
                         int *feat_idx,int feature_number,                                              // info sulle feature da usare
                         int LabelSelIdx,                                                               // info sulla label da usare
                         struct svm_model *model,const int predict_probability,                         // info sul modello da adoperare 
                         bool print)                         
{
    int idx,idx2,ii,jj,i,TotInstances=(int)FeatSize[0]; 


// ROBA PER DEBUG, DA CANCELLARE POI
// controllo la lettura dei dati
// direi che va bene
if (0){
    cout << endl << "TestSet" << endl; 
    for (int r=0; r<10; ++r){
        cout << " " << testing_idx[r];
    }
    cout << endl << endl << "Label" << endl; 
    for (int r=0; r<10; ++r){
        int cc=LabelSelIdx,
            rr=testing_idx[r]; 
        cout << " " << l[TotInstances * cc + rr];
    }
    cout << endl << endl << "Feat" << endl; 
    for (int r=0; r<10; ++r){
        for (int c=0; c<feature_number; ++c){
            int cc=feat_idx[c], 
                rr=testing_idx[r]; 
            cout << " " << f[TotInstances * cc + rr];
        }
        cout << endl; 
    }
    exit(1); 
}
//

    double *ptr_predict_label, *ptr_prob_estimates, *ptr_dec_values, ptr[3];
    struct svm_node *x;

    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    int svm_type=svm_get_svm_type(model);
    int nr_class=svm_get_nr_class(model);
    double *prob_estimates=NULL;

    if(predict_probability){
        if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
            printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
        else
            prob_estimates = new double [nr_class]; 
    }

    // setto l'uscita
    plhs[0] = Mat_VarCreate_jr("PredictedLabel", testing_instance_number, 1); 

    if(predict_probability){
        // prob estimates are in plhs[2]
        if(svm_type==C_SVC || svm_type==NU_SVC)
           plhs[2] = Mat_VarCreate_jr("ProbabilityEstimates", testing_instance_number, nr_class);
        else
            plhs[2] = Mat_VarCreate_jr("ProbabilityEstimates", 0, 0);
    }else{
        // decision values are in plhs[2]
        if(svm_type == ONE_CLASS ||
           svm_type == EPSILON_SVR ||
           svm_type == NU_SVR ||
           nr_class == 1) // if only one class in training data, decision values are still returned.
            plhs[2] = Mat_VarCreate_jr("DecisionValues", testing_instance_number, 1);
        else
            plhs[2] = Mat_VarCreate_jr("DecisionValues", testing_instance_number, nr_class*(nr_class-1)/2);
    }

    // punto il campo data delle variabili da salvare in uscita 
    ptr_predict_label = (double *) plhs[0]->data;
    ptr_prob_estimates = (double *) plhs[2]->data; 
    ptr_dec_values = (double *) plhs[2]->data;
    // alloco memoria alla struttura dei nodi 
    x = new struct svm_node [(feature_number+1)]; 


    for(ii=0; ii<testing_instance_number; ii++){ 
        idx = testing_idx[ii]; 
        double target_label, predict_label;
        target_label = l[TotInstances * LabelSelIdx + idx];
        // XXX non va bene per matrici sparse
        for(jj=0;jj<feature_number;jj++){
            idx2=feat_idx[jj]; 
            // NB: assegno ad x i valori della feature corrispondente 
            x[jj].index = idx2+1;
            x[jj].value =  f[TotInstances * idx2 + idx];  
        }
        x[feature_number].index = -1;

        if(predict_probability){
            if(svm_type==C_SVC || svm_type==NU_SVC){
                predict_label = svm_predict_probability(model, x, prob_estimates);
                //ptr_predict_label[idx] = predict_label;
                ptr_predict_label[ii] = predict_label;
                for(i=0;i<nr_class;i++)
                    //ptr_prob_estimates[idx + i * TotInstances] = prob_estimates[i];
                    ptr_prob_estimates[ii] = prob_estimates[i];
            }else{
                predict_label = svm_predict(model,x);
                //ptr_predict_label[idx] = predict_label;
                ptr_predict_label[ii] = predict_label;
            }
        }else{
            if(svm_type == ONE_CLASS ||
               svm_type == EPSILON_SVR ||
               svm_type == NU_SVR){
                double res;
                predict_label = svm_predict_values(model, x, &res);
                //ptr_dec_values[idx] = res;
                ptr_dec_values[ii] = res;
            }else{
                double *dec_values; 
                dec_values = new double [nr_class*(nr_class-1)/2];
                predict_label = svm_predict_values(model, x, dec_values);
                if(nr_class == 1) 
                    //ptr_dec_values[idx] = 1;
                    ptr_dec_values[ii] = 1;
                else
                    for(i=0;i<(nr_class*(nr_class-1))/2;i++)
                        //ptr_dec_values[idx + i * TotInstances] = dec_values[i];
                        ptr_dec_values[ii] = dec_values[i];
                delete [] dec_values;
            }
            //ptr_predict_label[idx] = predict_label;
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


    if (print){ 
        if(svm_type==NU_SVR || svm_type==EPSILON_SVR){
            printf("Mean squared error = %g (regression)\n",error/total);
            printf("Squared correlation coefficient = %g (regression)\n",
                ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
                ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt)));
        }
        else
            printf("Accuracy = %g%% (%d/%d) (classification)\n",
                    (double)correct/total*100,correct,total);
    }


  
    // return accuracy, mean squared error, squared correlation coefficient
    size_t dims[2];
    dims[0] = 3; 
    dims[1] = 1;
    ptr[0] = (double)correct/total*100;
    ptr[1] = (double)error/total;
    ptr[2] = (double)(((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
             ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt)));
    plhs[1] = Mat_VarCreate("ACC_MSE_SCC",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,ptr,0); // JR: da mettere un nome vero

    // libero memoria 
    delete [] x; 
    if(prob_estimates != NULL)
        delete [] prob_estimates;

    return ;    
}



