CXX = g++
CFLAGS = -std=c++0x -Wall -Wl,--copy-dt-needed-entries #-g -v -Q -da
LIB = -lmatio -lz  -lboost_thread -lm 
all: clean SMaTh 
SMaTh: svm.o Problema.o Job.o FunUtili.o
	$(CXX) $(CFLAGS) SMaTh.cpp svm.o Job.o Problema.o FunUtili.o $(LIB) -o SMaTh 
svm.o:  
	$(CXX) $(CFLAGS) -c svm.cpp
SvmMatio.o:
	$(CXX) $(CFLAGS) -c SvmMatio.cpp
Problema.o:  
	$(CXX) $(CFLAGS) -c Problema.cpp  
Job.o:  
	$(CXX) $(CFLAGS) -c Job.cpp 
FunUtili.o:
	$(CXX) $(CFLAGS) -c FunUtili.cpp 
clean:
	rm -f *~ *.o *.cpp.* SMaTh 
reopen: 
	pkill gvim; make clean; gvim -p * 


# how to highlight errors:
# make 2>&1 | grep -E --color=always 'error|$'
