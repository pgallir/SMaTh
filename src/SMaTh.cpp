#include "SvmMatio.h" 
#include <iostream> 
#include <sstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>   
#include <boost/thread/mutex.hpp> 
#include <boost/threadpool.hpp>

using namespace std;
using namespace boost::threadpool;
using namespace boost::algorithm; 

// define the class job that represent the enqueued task to the threadpool duties
// in this specific case, one job is to create an svm model with X% of training data and test it with {Y1,Y2,...,Yn}% of test data
class job
{
private: 
  struct svm_problem prob;
  struct svm_model *model;
  int iTr, iTs, iR;
  const char *error_msg;
  

public:
  static struct svm_parameter param;
  static int perc_sig;
  static const matvar_t *Features, *Labels, *VARIABLEs, *RIP, *idxCV, *idxVS, *TrSz, *TsSz;
  static double	***acc, ***mse, ***scc, ***ValidTrend;

  job(int ir, int itr) {iR=ir; iTr=itr;}
  ~job() {}

  void run()
  {
    int *TsPSel=NULL, LabelTsCellSize, predict_probability=0;
    const matvar_t *PARs[8] = {Features, Labels, VARIABLEs, RIP, idxCV, idxVS, TrSz, TsSz}; // fill PARs vector
    matvar_t **VAL, **STAT;

    // read the problem, malloc TsPSel and set it (& catch errors)
    error_msg = read_problem_from_variable(PARs, iTr, TsPSel, &LabelTsCellSize, perc_sig, &prob, &param); 
    if(error_msg)
    {
      fprintf(stderr,"ERROR: %s\n",error_msg);
      exit(0);
    }

    // train svm-model
    model = svm_train(&prob,&param);

    // test on the Validation Set
    {
      // predict the OUT{...} using IN{...} and the calculated svm model (& catch errors)
      VAL = validate_model(PARs, model, predict_probability);
      // fill ValidTrend
      double *trend = (double *)((matvar_t *)VAL[0])->data; 
      
      for (int iSample=0; iSample<(int)VAL[0]->dims[0]; iSample++) 
        ValidTrend[iTr][iR][iSample] = trend[iSample];
    }
     
    // cycle Test Sizes
    for (iTs=0; iTs<(int)TsSz->dims[1]; iTs++)
    {
      // check if I can do it: TrSz[iTr]+TsSz[iTs]<= #folds of signal
      double *trS = (double *)TrSz->data; 
      double *tsS = (double *)TsSz->data; 
      bool LessThanWholeSignal=0; 
	
      if (perc_sig==0) // 0 means that the signal folds are measured in steps
	LessThanWholeSignal=(int)trS[iTr]+(int)tsS[iTs] > (int)idxCV->dims[1];
      else if(perc_sig==1) // 1 means that are measured in percentage of the whole recorded signal 
	LessThanWholeSignal=(int)trS[iTr]+(int)tsS[iTs] > 100;
    
      if (LessThanWholeSignal)
      {
	acc[iTr][iTs][iR]=NAN;
        mse[iTr][iTs][iR]=NAN;
        scc[iTr][iTs][iR]=NAN;
	continue; 
      }
      // predict the OUT{...} using IN{...} and the calculated svm model (& catch errors)
      STAT = test_model_on_data_subset(PARs, iTs, TsPSel, LabelTsCellSize, model, predict_probability, perc_sig);
      double *r = (double *)((matvar_t *)STAT[1])->data; 
      // printf(" [%f,%f,%f]\n", r[0], r[1], r[2]);
      acc[iTr][iTs][iR]=r[0];
      mse[iTr][iTs][iR]=r[1];
      scc[iTr][iTs][iR]=r[2];
    }

    // END) free memory			
    Mat_VarFree(STAT[0]); 
    Mat_VarFree(STAT[1]); 
    Mat_VarFree(STAT[2]); 
    svm_destroy_param(&param);
    svm_free_and_destroy_model(&model);
    free(TsPSel);
    free(prob.y);
    free(prob.x);
    return; 
  }	
};

struct svm_parameter job::param;
int job::perc_sig;
const matvar_t	*job::Features, 
		*job::Labels, 
		*job::VARIABLEs, 
		*job::RIP,
		*job::idxCV,
		*job::idxVS,
		*job::TrSz,
		*job::TsSz;
double	***job::acc, 
	***job::mse, 
	***job::scc,
	***job::ValidTrend; 


int main(int argc, char** argv) 
{	
  // Create a thread pool.
  pool tp(8); 
  // change the seed	
  srand(time(NULL));
  // declare variables 
  struct svm_parameter param;
  const char *error_msg;
  int iR, iTr; 
  int perc_sig=1; // 0 means signals fold are in steps, 1 means in percentage of whole recorded signal. by default, it is 0. 
  double DimFeatures=NAN, RipFeatures=NAN, Label=NAN, *iRip; 
  bool SelFeatures=0; 
  char IFN[1024], *OFN=Malloc(char,1024); 
  matvar_t *cell_row[3], *TrendCell[1], *varnameC, **CellV; 	
  string VARNAME; 
  
  ////////////////////////////////////////////
  ////////////	prepare the simulation
  ////////////////////////////////////////////
  
  // parse arguments from outside 
  parse_VIarguments(argc, argv, IFN, &param, &DimFeatures, &RipFeatures, &Label, &perc_sig);

  // load from IFN, which must be a *.mat file, the variables I need
  mat_t   *In_MATFILE = Mat_Open(IFN,MAT_ACC_RDONLY),  		// INPUT.MAT
	  *Out_MATFILE;						// OUTPUT(s).MAT

  matvar_t  *Features, *act_features,  // matrix of features (RxC1), R samples of C1 features
	    *Labels, *act_label,  // matrix of labels (RxC2), R samples of C2 labels
	    *idxCV,		  // indexes for Cross Validation - cellarray (1xM) of vectors (1xN), M slices of data with N samples each
	    *idxVS,		  // indexes for Validation Set - cellarray (1xM) of vectors (1xN), M slices of data with N samples each
	    *RIP, *TrSz, *TsSz,	  // # CV numerosity (1x1), vectors of {Training,Test} Size (1xK)   
	    *RESULTs,		  // here is where the output stats will be stored
            *TRENDs,              // here is where the output trends will be stored
	    *VARIABLEs;		  // list of variables
  Labels =  Mat_VarRead(In_MATFILE, "OUT"); // *** var in matlab WS
  Features = Mat_VarRead(In_MATFILE, "IN"); // *** var in matlab WS
  VARIABLEs = Mat_VarRead(In_MATFILE, "VARIABLEs"); // *** var in matlab WS
  RIP = Mat_VarRead(In_MATFILE, "RIP"); // *** var in matlab WS
  idxCV = Mat_VarRead(In_MATFILE, "idxCV"); // *** var in matlab WS	
  idxVS = Mat_VarRead(In_MATFILE, "idxVS"); // *** var in matlab WS	
  TrSz = Mat_VarRead(In_MATFILE, "TrSz"); // *** var in matlab WS
  TsSz = Mat_VarRead(In_MATFILE, "TsSz"); // *** var in matlab WS	
  iRip = (double*) RIP->data; 
  const matvar_t *PARs[8] = {Features, Labels, VARIABLEs, RIP, idxCV, idxVS, TrSz, TsSz}; // fill PARs vector

  // prepare output matvar_t* variables 
  size_t dims[2], dims_1c[3], dims_vl[3]; 
  // RESULTS VARIABLE DIM
  dims[1] = 1; dims[0] = 3;
  RESULTs = Mat_VarCreate("RESULTs",MAT_C_CELL,MAT_T_CELL,2,dims,NULL,0);
  dims[1] = 1; dims[0] = 1;
  TRENDs  = Mat_VarCreate("TRENDs",MAT_C_CELL,MAT_T_CELL,2,dims,NULL,0);
  // RESULTS Stats DIM
  dims_1c[0] = (int)TrSz->dims[1]; dims_1c[1] = (int)TsSz->dims[1]; dims_1c[2] = (int)(*iRip); 
  // RESULTS Trends DIM
  int VlSelSize=0; CellV=(matvar_t**) idxVS->data; 
  for (int i=0; i<(int)idxVS->dims[1]; i++)
      VlSelSize += CellV[i]->dims[1];
  dims_vl[0] = (int)TrSz->dims[1]; dims_vl[1] = (int)(*iRip); dims_vl[2] = VlSelSize; 
  // ALLOCATION
  double ***acc_, ***mse_, ***scc_, ***ValidTrend_;   
  acc_ = Malloc(double**,dims_1c[0]); 
  mse_ = Malloc(double**,dims_1c[0]); 
  scc_ = Malloc(double**,dims_1c[0]); 
  ValidTrend_ = Malloc(double**,dims_vl[0]); 
  for(int i=0; i<(int)dims_1c[0]; i++)
  {
    acc_[i] = Malloc(double*,dims_1c[1]); 
    mse_[i] = Malloc(double*,dims_1c[1]); 
    scc_[i] = Malloc(double*,dims_1c[1]); 
    ValidTrend_[i] = Malloc(double*,dims_vl[1]); //dims_1c[0] == dims_vl[0]
    for(int j=0; j<(int)dims_1c[1]; j++)
    {
      acc_[i][j] = Malloc(double,dims_1c[2]);
      mse_[i][j] = Malloc(double,dims_1c[2]);
      scc_[i][j] = Malloc(double,dims_1c[2]);
    }
    for(int j=0; j<(int)dims_vl[1]; j++)  //dims_1c[1] != dims_vl[1]
      ValidTrend_[i][j] = Malloc(double,dims_vl[2]); 
  }
  double acc[(int)dims_1c[0]]
	    [(int)dims_1c[1]]
	    [(int)dims_1c[2]], 
	 mse[(int)dims_1c[0]]
	    [(int)dims_1c[1]]
	    [(int)dims_1c[2]], 
	 scc[(int)dims_1c[0]]
	    [(int)dims_1c[1]]
	    [(int)dims_1c[2]],
         ValidTrend[(int)dims_vl[0]]
                   [(int)dims_vl[1]]
                   [(int)dims_vl[2]], 
	 act_label_[(int) Labels->dims[0]]; 	
  int feature_sel[(int)Features->dims[1]]; 

  // check if I have everything I need  
  error_msg = Mat_VarErrCheck(PARs);
  if(error_msg)
  {
    fprintf(stderr,"ERROR: %s\n",error_msg);
    exit(1);
  }

  /////////////////
  /////  start 
  /////////////////

  // do I want to select only part of the features?
  int nRFeat=1;
  if (DimFeatures==DimFeatures) // if DimFeatures is not NAN
  {
    if (RipFeatures==RipFeatures) // if RipFeatures is not NAN
    {
      nRFeat=RipFeatures;
      SelFeatures=1; 
      // check if DimFeatures is not bigger than #Features 
      if (DimFeatures > (int)Features->dims[1])
      {
	fprintf(stderr,"DimFeatures is bigger than possible. #Features==%d\n",(int)Features->dims[1]);
	exit(1); 
      }
   }else{
      fprintf(stderr,"DimFeatures has been set, but RipFeatures has not been set. This is wrong. Check --help for further info\n");
      exit(1); 
    }  
  }
 
  // set all job::*stuff* but for Labels and Features.
  job::param = param;
  job::perc_sig = perc_sig;
  job::VARIABLEs=PARs[2]; 
  job::RIP=PARs[3]; 
  job::idxCV=PARs[4];
  job::idxVS=PARs[5];
  job::TrSz=PARs[6];
  job::TsSz=PARs[7];
  job::acc=acc_;
  job::mse=mse_;
  job::scc=scc_;
  job::ValidTrend=ValidTrend_;

  // overwrite PARs[1] to the actual label
  double *pLabels = (double*)Labels->data;

  for (int iRFeat=0; iRFeat<nRFeat; iRFeat++)
  {
    // set the actual features
    if(SelFeatures) 
    {
      double act_features_[(int)DimFeatures][(int)Features->dims[0]]; 
      double *pFeat = (double*) Features->data; 

      // select DimFeatures features randomly 
      randperm((int)Features->dims[1],feature_sel); 
      // and set them in job::Features
      for (int l=0; l<(int)DimFeatures; l++)
	for (int i=0; i<(int)Features->dims[0]; i++)
	  act_features_[l][i] = pFeat[feature_sel[l]*(int)Features->dims[0]+i]; 
      dims[0] = (int)Features->dims[0]; dims[1] = (int)DimFeatures; 
      act_features = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,act_features_,0); 
      //cout << "#" << feature_sel[0]+1 << "| #" << feature_sel[1]+1;
      //Mat_VarPrint(act_features, 1);
      // set job::Features
      job::Features=act_features; 
    }else{
      job::Features=PARs[0];      
    }

    for (int VARIABLE=0; VARIABLE<(int)Labels->dims[1]; VARIABLE++)
    {
      if (Label==Label) // if not NAN
	if (VARIABLE!=(int)Label)	// check if VARIABLE is the right one
	  continue; 

      // set the actual label
      for (int i=0; i<(int)Labels->dims[0]; i++)
	act_label_[i] = pLabels[VARIABLE*(int)Labels->dims[0]+i];
      dims[0] = (int)Labels->dims[0]; dims[1] = 1;
      act_label = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,act_label_,0); 
      //Mat_VarPrint(act_label, 1);
      job::Labels=act_label; 

      // get variable name ("trimmed" with underscores)
      varnameC = Mat_VarGetCell(VARIABLEs,VARIABLE);
      VARNAME=(const char*)varnameC->data;
      replace_all(VARNAME," ","_");    
      // naming output matfile
      memcpy(OFN,IFN,1024); 
      strtok(OFN,"."); // get input filename without mat extension
      strcat(OFN,"-");
      strcat(OFN,VARNAME.c_str());
      if(SelFeatures)
      {
	strcat(OFN,"-SFsize-");
	std::ostringstream ostr;
	ostr << DimFeatures;
	strcat(OFN,ostr.str().c_str());
	
	strcat(OFN,"-SFrip-");
	ostr.clear(); ostr.str("");
	ostr << iRFeat;
	strcat(OFN,ostr.str().c_str());
      }
      strcat(OFN,"-res.mat");
      Out_MATFILE = Mat_CreateVer(OFN,NULL,MAT_FT_DEFAULT);	
   
      //cout<<OFN<<"\n";
      //exit(1);
      ////////////////////////////////////////////
      ////////////	sim on actual label
      ////////////////////////////////////////////

      cout<<"\ti="<<VARIABLE<<"\tworking on '"<<OFN<<"'\n"; fflush(stdout);

      iR=0; 
      // cycle repetition-s for Cross Validation
      while(iR<(*iRip))  
      {
	// cycle Training Sizes
	for (iTr=0; iTr<(int)TrSz->dims[1]; iTr++)
	{
	  // Add task to the pool
	  job j(iR, iTr);
	  tp.schedule(boost::bind(&job::run,j));  // multithr
          //j.run(); cout<<"****************"<<endl<<"NOT MULTITHREADING"<<endl<<"****************"<<endl; fflush(stdout);
	}
	++iR;
      }
      tp.wait();
  
      ////////////////////////////////////////////
      ////////////	    sim ended
      ////////////////////////////////////////////
 	
      // copy results
      iR=0; 
      // cycle repetition-s for Cross Validation
      while(iR<(*iRip))  
      {
	// cycle Training Sizes
	for (iTr=0; iTr<(int)TrSz->dims[1]; iTr++)
        {
	  for (int iTs=0;  iTs<(int)TsSz->dims[1]; iTs++)
	  {
	    acc[iTr][iTs][iR]=job::acc[iTr][iTs][iR];
	    mse[iTr][iTs][iR]=job::mse[iTr][iTs][iR];
	    scc[iTr][iTs][iR]=job::scc[iTr][iTs][iR];
	  }
          for (int iSample=0; iSample<VlSelSize; iSample++)
            ValidTrend[iTr][iR][iSample] = job::ValidTrend[iTr][iR][iSample];
        }
	++iR;
      }

      // create a cell that contains accuracy (acc), mean squared error (mse) and squared correlation coefficient (scc)
      cell_row[0] = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims_1c,acc,0);
      cell_row[1] = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims_1c,mse,0);
      cell_row[2] = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims_1c,scc,0);
      Mat_VarSetCell(RESULTs,0,cell_row[0]); 
      Mat_VarSetCell(RESULTs,1,cell_row[1]); 
      Mat_VarSetCell(RESULTs,2,cell_row[2]); 
      // create a cell that contains accuracy (acc), mean squared error (mse) and squared correlation coefficient (scc)
      TrendCell[0] = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,3,dims_vl,ValidTrend,0);
      Mat_VarSetCell(TRENDs,0,TrendCell[0]); 

      // associate {RESULTs, VARIABLEs} to Out_MATFILE
      Mat_VarWrite(Out_MATFILE, RESULTs, MAT_COMPRESSION_NONE);
      Mat_VarWrite(Out_MATFILE, TRENDs, MAT_COMPRESSION_NONE);
    }
  }
  /////////////////
  /////  end 
  /////////////////

  // free allocated memory		
  // Mat_*variables
  Mat_VarFree(Features); 
  if (SelFeatures) {Mat_VarFree(act_features);}  
  Mat_VarFree(Labels);  
  Mat_VarFree(act_label);  
  Mat_VarFree(idxCV); 
  Mat_VarFree(idxVS);
  Mat_VarFree(RIP); 
  Mat_VarFree(TrSz); 
  Mat_VarFree(TsSz); 
  Mat_VarFree(RESULTs);
  Mat_VarFree(TRENDs); 
  Mat_VarFree(VARIABLEs); 
  // Mat_*files
  Mat_Close(In_MATFILE);
  Mat_Close(Out_MATFILE);
  // char *
  free(OFN);
  // Malloc-ed variables
  free(acc_); 
  free(mse_);
  free(scc_); 
  free(ValidTrend_);
  return 0;
} 
	


