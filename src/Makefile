CXX = gcc
CFLAGS = -Wall -Wl,--copy-dt-needed-entries #-g -v -Q -da
LIB = -lmatio -lz  -lboost_thread 
all: SvmMatio.o SMaTh
SMaTh: SMaTh.cpp svm.o SvmMatio.o
	$(CXX) $(CFLAGS) SMaTh.cpp svm.o SvmMatio.o $(LIB) -o SMaTh -lm
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
SvmMatio.o : SvmMatio.cpp SvmMatio.h svm.o
	$(CXX) $(CFLAGS) -c SvmMatio.cpp
clean:
	rm -f *~ *.o *.cpp.* SMaTh
