##----------##
## Makefile ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2017.11.21                               ##
##-------------------------------------------------------##

include ../Makefile.global

CC   := g++
NVCC := nvcc

BUILD_PATH   := ../build/lib
PROJECT_NAME := BoraBase

HEADER_PATH  := header
SOURCE_PATH  := source
OBJECT_PATH  := object

INCLUDES := -I.
INCLUDES += -I./header
INCLUDES += -I./unittest
INCLUDES += -I/usr/local/include
INCLUDES += -I/usr/local/include/eigen3
INCLUDES += -I/usr/local/cuda/include
INCLUDES += -I/usr/include/python2.7
INCLUDES += -I$(HOU_INSTALL_PATH)/toolkit/include

LINKS := -L.
LINKS += -L/usr/lib64
LINKS += -L/usr/local/lib
LINKS += -lglfw -lGLEW
LINKS += -L/usr/local/cuda/lib64
LINKS += -lHalf -ltbb -lcudart -lcublas
LINKS += -lboost_system -lboost_filesystem -lboost_iostreams
LINKS += -L$(HOU_INSTALL_PATH)/dsolib
LINKS += -lHoudiniGEO -lHoudiniOP1 -lHoudiniOP2 -lHoudiniOP3 -lHoudiniOPZ -lHoudiniPRM
LINKS += -lHoudiniRAY -lHoudiniSIM -lHoudiniUI -lHoudiniUT
LINKS += -lopenvdb_sesi

DEFINES := $(HDEFINES)
DEFINES += -DGLM_FORCE_CUDA 
DEFINES += -DGLM_ENABLE_EXPERIMENTAL

CCFLAGS := -Wall -W -O3 -m64 -fPIC -fopenmp -std=c++11
CUFLAGS := -m64 -arch=compute_61 -Xcompiler -fPIC -O3 -std=c++11 --disable-warnings --expt-extended-lambda

CPP_SOURCES  := $(wildcard $(SOURCE_PATH)/*.cpp)
HDR_SOURCES  := $(wildcard $(HEADER_PATH)/*.h)
CPP_LIST     := $(notdir $(CPP_SOURCES))						 # *.cpp
OBJ_LIST     := $(addprefix $(OBJECT_PATH)/,$(CPP_LIST:.cpp=.o)) # *.o

CU_SOURCES   := $(wildcard $(SOURCE_PATH)/*.cu)
CU_LIST      := $(notdir $(CU_SOURCES))							 # *.cu
OBJ_LIST     += $(addprefix $(OBJECT_PATH)/,$(CU_LIST:.cu=.o))	 # *.o

DEP_LIST     := $(addprefix $(OBJECT_PATH)/,$(CPP_LIST:.cpp=.d)) # .d
DEP_LIST     += $(addprefix $(OBJECT_PATH)/,$(CPP_LIST:.cu=.d))	 # .d

RPATHS       := -Wl,-rpath=.
RPATHS       += -Wl,-rpath=$(BUILD_PATH)
RPATHS       += -Wl,-rpath=$(HOU_INSTALL_PATH)/dsolib

.SILENT:

all: comm $(OBJECT_PATH)/lib$(PROJECT_NAME).so

comm:
	@echo -e '\e[1;32m Building $(PROJECT_NAME)... \e[m'
	chmod 755 source/*.*
	chmod 755 header/*.*

# .o -> .so
$(OBJECT_PATH)/lib$(PROJECT_NAME).so: $(OBJ_LIST)
	$(CC) $(CCFLAGS) $(LINKS) -shared $(OBJ_LIST) -o $@
	mkdir -p -m 755 $(BUILD_PATH)
	cp -rf $@ $(BUILD_PATH)	

# .cpp -> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cpp
	mkdir -p -m 755 $(OBJECT_PATH)
	$(CC) $(CCFLAGS) $(DEFINES) $(INCLUDES) $(SWITCHES) -MMD -c $< -o $@

# .cu -> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cu $(HDR_SOURCES)
	mkdir -p -m 755 $(OBJECT_PATH)
	$(NVCC) $(CUFLAGS) $(DEFINES) $(INCLUDES) -c $< -o $@
    
-include $(DEP_LIST)

unit:
	$(CC) $(CCFLAGS) $(DEFINES) $(INCLUDES) $(LINKS) -L$(OBJECT_PATH) -l$(PROJECT_NAME) $(RPATHS) $(SWITCHES) ./unittest/test.cpp -o $(OBJECT_PATH)/test
	$(OBJECT_PATH)/test

exec:
	$(CC) $(CCFLAGS) $(DEFINES) $(INCLUDES) $(LINKS) -lGL -lGLU -lglut -L$(OBJECT_PATH) -l$(PROJECT_NAME) $(RPATHS) $(SWITCHES) ./main.cpp -o $(OBJECT_PATH)/main
	$(OBJECT_PATH)/main
	
clean:
	rm -rf $(OBJECT_PATH)/*
	rm -rf $(BUILD_PATH)/*

